"""A track frame must bind one frozen tracker identity to its original pixels."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from frigate.api import media
from frigate.api.auth import require_admin_by_default
from frigate.camera.state import FrozenFace, FrozenVehicle
from frigate.models import Event
from frigate.test import test_tracked_object_publication as tracked_fixture
from frigate.test.http_api.base_http_test import AuthTestClient
from frigate.test.test_live_track_snapshot import LivePersonFixture

CAMERA = tracked_fixture.CAMERA
TRACK_ID = tracked_fixture.EVENT_ID
MAX = media.MAX_FRAME_FACES


class TestHttpTrackFrame(unittest.TestCase):
    def setUp(self):
        self.fixture = tracked_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = self.fixture.state
        self.state.camera_config.enabled = True
        self.state.camera_config.detect.enabled = True
        # A uniform native image makes overlays and crops observable after JPEG decode.
        image = self.fixture.processor.frame_manager.get_captured_frame.return_value
        image[:] = 128
        image[:240] = 96
        self.fixture.start_stationary_track()
        self.app = FastAPI(dependencies=[Depends(require_admin_by_default())])
        self.app.include_router(media.router)
        self.app.frigate_config = self.fixture.processor.config
        self.app.detected_frames_processor = self.fixture.processor

    def capture(self, client, camera=CAMERA, track_id=TRACK_ID, age=0.1, **kwargs):
        """Exercise real media/auth code while rejecting any Event-table access."""
        with (
            patch(
                "frigate.api.media.time.time",
                return_value=self.state.current_frame_time + age,
            ),
            patch.object(Event, "select", side_effect=AssertionError("Event DB read")),
            patch.object(Event, "get", side_effect=AssertionError("Event DB read")),
        ):
            return client.get(f"/{camera}/tracks/{track_id}/frame.jpg", **kwargs)

    def assert_frame(self, response, label, frame_time, vehicles, faces=None):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(
            json.loads(response.headers["x-frigate-track"]),
            {"id": TRACK_ID, "camera": CAMERA, "label": label, "end_time": None},
        )
        self.assertEqual(float(response.headers["x-frame-time"]), frame_time)
        calibration = json.loads(response.headers["x-calibration-frame"])
        expected = {
            "state": "matched",
            "capturedAtMs": int(frame_time * 1000 + 0.5),
            "width": 320,
            "height": 240,
            "vehicles": vehicles,
        }
        if faces is not None:
            expected["faces"] = faces
        self.assertEqual(calibration, expected)
        decoded = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded.shape, (240, 320, 3))
        return decoded

    def fresh_vehicle(self, frame_time):
        """The vehicle entry for a track the model measured on that frame."""
        return {
            "box": [100, 100, 200, 200],
            "id": TRACK_ID,
            "detectorObservedAtMs": int(frame_time * 1000 + 0.5),
        }

    def test_vehicle_returns_full_native_pixels_and_only_frozen_track_metadata(self):
        frame, frame_time, track, boxes, faces = self.state.get_track_frame(TRACK_ID)
        self.assertEqual(track, (TRACK_ID, "car", None))
        self.assertEqual(
            boxes, (FrozenVehicle((100, 100, 200, 200), TRACK_ID, frame_time),)
        )
        self.assertIsNone(faces)
        expected = frame.copy()
        frame[:] = 0
        np.testing.assert_array_equal(self.state.get_track_frame(TRACK_ID)[0], expected)
        obj = self.state.tracked_objects[TRACK_ID]
        obj.obj_data.update(label="person", end_time=999, box=[0, 0, 20, 20])
        obj.path_data.extend([((0.5, 0.5), 100.0)] * 10000)
        with AuthTestClient(self.app) as client:
            response = self.capture(client)
        decoded = self.assert_frame(
            response, "car", frame_time, [self.fresh_vehicle(frame_time)]
        )
        self.assertLessEqual(
            int(np.abs(decoded.astype(int) - expected.astype(int)).max()), 2
        )

    def test_stationary_unrecorded_person_can_supply_a_frame(self):
        person = LivePersonFixture(position_changes=0)
        self.addCleanup(person.close)
        person.state.camera_config.record.enabled = True
        person.state.camera_config.snapshots.enabled = True
        person.start()
        obj = person.state.tracked_objects[TRACK_ID]
        self.assertFalse(obj.false_positive)
        self.assertFalse(obj.is_active())
        self.assertFalse(person.processor.should_save_snapshot(CAMERA, obj))
        self.assertFalse(person.processor.should_retain_recording(CAMERA, obj))
        self.app.detected_frames_processor = person.processor
        self.app.frigate_config = person.processor.config
        self.state = person.state
        with AuthTestClient(self.app) as client:
            response = self.capture(client)
        self.assert_frame(response, "person", self.state.current_frame_time, [])

    def test_next_tracker_update_cannot_relabel_or_replace_the_prior_published_frame(
        self,
    ):
        before_time = self.state.current_frame_time
        self.fixture.processor.frame_manager.get_captured_frame.return_value[:240] = 160
        observed = []
        with AuthTestClient(self.app) as client:

            def during_update(*_args):
                self.state.tracked_objects[TRACK_ID].obj_data["label"] = "truck"
                observed.append(self.capture(client))

            self.state.on("camera_activity", during_update)
            self.fixture.next_frame()
            after = self.capture(client)
        self.assertEqual(len(observed), 1)
        before_pixels = self.assert_frame(
            observed[0], "car", before_time, [self.fresh_vehicle(before_time)]
        )
        after_pixels = self.assert_frame(
            after,
            "truck",
            self.state.current_frame_time,
            [self.fresh_vehicle(self.state.current_frame_time)],
        )
        self.assertGreater(float(after_pixels.mean() - before_pixels.mean()), 50)

    def test_actual_end_is_410_but_fresh_missing_is_404_and_stale_end_is_unknown(self):
        self.state.update("ended", self.state.current_frame_time + 1, {}, [], [])
        ended_at = self.state.current_frame_time
        self.assertEqual(
            self.state.get_track_frame(TRACK_ID)[2], (TRACK_ID, "car", ended_at)
        )
        with AuthTestClient(self.app) as client:
            self.assertEqual(self.capture(client).status_code, 410)
            self.assertEqual(self.capture(client, age=5.01).status_code, 503)
            self.state.finished(TRACK_ID)
            self.state.update("empty", ended_at + 1, {}, [], [])
            self.assertEqual(self.capture(client).status_code, 404)

    def test_false_positive_and_unsupported_tracks_never_return_pixels(self):
        person = LivePersonFixture()
        self.addCleanup(person.close)
        person.start(score=0.1)
        self.app.detected_frames_processor = person.processor
        self.app.frigate_config = person.processor.config
        self.state = person.state
        with AuthTestClient(self.app) as client:
            self.assertEqual(self.capture(client).status_code, 404)
        self.app.detected_frames_processor = self.fixture.processor
        self.app.frigate_config = self.fixture.processor.config
        self.state = self.fixture.state
        self.state.on(
            "camera_activity",
            lambda *_args: self.state.tracked_objects[TRACK_ID].obj_data.update(
                label="dog"
            ),
        )
        self.fixture.next_frame()
        with AuthTestClient(self.app) as client:
            self.assertEqual(self.capture(client).status_code, 404)
        with patch(
            "frigate.camera.state.np.copy",
            side_effect=AssertionError("unexpected frame copy"),
        ):
            self.assertIsNone(self.state.get_track_frame(TRACK_ID)[0])
            self.assertIsNone(self.state.get_track_frame("missing")[0])

    def test_disabled_missing_stale_future_and_unpublished_sources_are_503(self):
        with AuthTestClient(self.app) as client:
            for age in (-0.01, 5.01):
                with self.subTest(age=age):
                    self.assertEqual(self.capture(client, age=age).status_code, 503)
            for target in (self.state.camera_config, self.state.camera_config.detect):
                with self.subTest(target=type(target).__name__):
                    target.enabled = False
                    self.assertEqual(self.capture(client).status_code, 503)
                    target.enabled = True
            self.fixture.processor.camera_states.pop(CAMERA)
            self.assertEqual(self.capture(client).status_code, 503)
            self.fixture.processor.camera_states[CAMERA] = self.state
            self.fixture.processor.frame_manager.get_captured_frame.return_value = None
            self.fixture.next_frame()
            self.assertIsNone(self.state.get_track_frame(TRACK_ID))
            self.assertEqual(self.capture(client).status_code, 503)

    def test_wrong_camera_cannot_borrow_pixels_and_unauthorized_camera_is_not_read(
        self,
    ):
        config = self.fixture.processor.config
        config.cameras["other"] = self.state.camera_config.model_copy(deep=True)
        self.fixture.processor.create_camera_state("other")
        other = self.fixture.processor.camera_states["other"]
        other.update("other-empty", self.state.current_frame_time, {}, [], [])
        config.auth.roles["limited"] = [CAMERA]
        headers = {"remote-user": "guard", "remote-role": "limited"}
        with AuthTestClient(self.app) as client:
            self.assertEqual(self.capture(client, camera="other").status_code, 404)
            self.assertEqual(self.capture(client, headers=headers).status_code, 200)
            with patch.object(
                other,
                "get_track_frame",
                side_effect=AssertionError("unauthorized read"),
            ):
                self.assertEqual(
                    self.capture(client, camera="other", headers=headers).status_code,
                    403,
                )
        with TestClient(self.app) as client:
            self.assertEqual(self.capture(client).status_code, 401)

    def test_encoder_failure_is_unavailable_without_jpeg(self):
        with (
            patch("frigate.api.media.cv2.imencode", return_value=(False, None)),
            AuthTestClient(self.app) as client,
        ):
            response = self.capture(client)
        self.assertEqual(response.status_code, 503)
        self.assertNotEqual(response.headers.get("content-type"), "image/jpeg")


class TestTrackFrameFaceHints(TestHttpTrackFrame):
    """An optional hint may be withheld; it may never be quietly incomplete."""

    def publish(self, faces, observed=True):
        """Publish one frame carrying this face pass, and read the header back."""
        self.fixture.next_frame(face_regions=faces, observed=observed)
        with AuthTestClient(self.app) as client:
            response = self.capture(client)
        self.assertEqual(response.status_code, 200)
        return json.loads(response.headers["x-calibration-frame"])

    def face(self, box=(140, 118, 172, 156), score=0.86, at=None):
        return FrozenFace(box, score, self.fixture.frame_time + 1 if at is None else at)

    def test_a_same_frame_region_is_published_with_its_own_clock(self):
        metadata = self.publish((self.face(),))
        clock = int(self.state.current_frame_time * 1000 + 0.5)

        self.assertEqual(
            metadata["faces"],
            [
                {
                    "box": [140, 118, 172, 156],
                    "score": 0.86,
                    "detectorObservedAtMs": clock,
                }
            ],
        )
        self.assertEqual(metadata["capturedAtMs"], clock)

    def test_an_empty_pass_and_a_missing_pass_stay_distinguishable(self):
        self.assertEqual(self.publish(())["faces"], [])
        self.assertNotIn("faces", self.publish(None))

    def test_more_regions_than_the_bound_withholds_the_hint_but_keeps_the_frame(self):
        crowd = tuple(
            self.face(box=(x, 10, x + 8, 20)) for x in range(0, 8 * (MAX + 1), 8)
        )
        metadata = self.publish(crowd)

        # The image and the vehicles a capture needs are still there. Only the
        # optional hint is gone, because a truncated list would read as whole.
        self.assertNotIn("faces", metadata)
        self.assertEqual(
            metadata["vehicles"],
            [self.fresh_vehicle(self.state.current_frame_time)],
        )

    def test_exactly_the_bound_is_still_published(self):
        crowd = tuple(self.face(box=(x, 10, x + 8, 20)) for x in range(0, 8 * MAX, 8))
        self.assertEqual(len(self.publish(crowd)["faces"]), MAX)

    def test_one_unusable_region_withholds_the_whole_list(self):
        for name, bad in (
            ("out of frame", self.face(box=(300, 200, 340, 260))),
            ("inverted", self.face(box=(180, 150, 140, 118))),
            ("stale clock", self.face(at=self.fixture.frame_time)),
            ("future clock", self.face(at=self.fixture.frame_time + 2)),
        ):
            with self.subTest(name=name):
                metadata = self.publish((self.face(), bad))
                self.assertNotIn("faces", metadata)

    def test_a_later_tracker_mutation_cannot_change_an_already_frozen_frame(self):
        frozen = self.publish((self.face(),))
        obj = self.state.tracked_objects[TRACK_ID]
        obj.obj_data.update(box=[0, 0, 20, 20], label="person")
        with AuthTestClient(self.app) as client:
            again = json.loads(self.capture(client).headers["x-calibration-frame"])
        self.assertEqual(again, frozen)

    def test_a_reused_stationary_box_is_never_named_as_a_target(self):
        """Its pixels are still current; its measurement is not this frame's."""
        metadata = self.publish((self.face(),), observed=False)

        self.assertEqual(metadata["vehicles"], [{"box": [100, 100, 200, 200]}])
        self.assertEqual(len(metadata["faces"]), 1)

    def test_a_header_over_budget_yields_the_hint_rather_than_the_frame(self):
        self.fixture.next_frame(face_regions=(self.face(),))
        self.state._current_frame_vehicles = tuple(
            FrozenVehicle(
                (1, 1, 2, 2), f"{'x' * 160}-{index}", self.state.current_frame_time
            )
            for index in range(32)
        )
        with AuthTestClient(self.app) as client:
            response = self.capture(client)
        metadata = json.loads(response.headers["x-calibration-frame"])

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("faces", metadata)
        self.assertEqual(len(metadata["vehicles"]), 32)

    def test_the_calibration_leg_never_carries_faces_or_identity(self):
        self.fixture.next_frame(face_regions=(self.face(),))
        with (
            AuthTestClient(self.app) as client,
            patch(
                "frigate.api.media.time.time",
                return_value=self.state.current_frame_time + 0.1,
            ),
        ):
            response = client.get(f"/{CAMERA}/calibration.jpg")
        metadata = json.loads(response.headers["x-calibration-frame"])

        self.assertNotIn("faces", metadata)
        self.assertEqual(metadata["vehicles"], [{"box": [100, 100, 200, 200]}])
