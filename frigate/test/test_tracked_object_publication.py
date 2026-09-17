"""OCR metadata updates must reach the ordinary tracked-object event callback."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from peewee import DoesNotExist

from frigate.config import FrigateConfig
from frigate.track.object_processing import TrackedObjectProcessor

CAMERA = "front"
EVENT_ID = "100-synthetic"


class TestRecognizedPlatePublication(unittest.TestCase):
    def setUp(self) -> None:
        config = FrigateConfig(
            mqtt={"enabled": False},
            detectors={"test": {"type": "onnx"}},
            model={
                "labelmap_path": str(
                    Path(__file__).resolve().parents[2] / "labelmap.txt"
                )
            },
            cameras={
                CAMERA: {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://camera.invalid/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {"width": 320, "height": 240},
                    "objects": {"track": ["car"]},
                    "zones": {
                        "approach": {"coordinates": "0,0,1,0,1,1,0,1", "inertia": 1}
                    },
                }
            },
        )
        # Keep the real camera callbacks and metadata setter, without starting
        # the processor's sockets, thread, camera or database connections.
        self.processor = TrackedObjectProcessor.__new__(TrackedObjectProcessor)
        self.processor.config = config
        self.processor.dispatcher = MagicMock()
        self.processor.event_sender = MagicMock()
        self.processor.requestor = MagicMock()
        self.processor.ptz_autotracker_thread = MagicMock()
        self.processor.frame_manager = MagicMock()
        self.processor.frame_manager.get_captured_frame.return_value = np.zeros(
            config.cameras[CAMERA].frame_shape_yuv, dtype=np.uint8
        )
        self.processor.camera_states = {}
        self.processor.camera_activity = {}
        self.processor.create_camera_state(CAMERA)
        self.state = self.processor.camera_states[CAMERA]
        self.frame_time = 99.0
        self.observed_at = 99.0
        self.score = 0.9
        event_get = patch(
            "frigate.track.object_processing.Event.get", side_effect=DoesNotExist
        )
        event_get.start()
        self.addCleanup(event_get.stop)

    def next_frame(
        self,
        step: float = 1,
        *,
        observed: bool = True,
        attributes: list[dict] | None = None,
        face_regions=None,
    ) -> None:
        """Publish one frame for the tracked car.

        Args:
            observed: Whether the model measured this box on this frame. False
                is the stationary case, where the tracker carries an older
                measurement forward onto the current frame clock.
            attributes: Per-frame attribute regions assigned to this track.
            face_regions: Raw face regions frozen with this frame, or None when
                the frame carries no face observation pass.
        """
        self.frame_time += step
        detection = {
            "id": EVENT_ID,
            "label": "car",
            "frame_time": self.frame_time,
            "detector_observed_at": self.frame_time if observed else self.observed_at,
            "start_time": 100.0,
            "score": self.score,
            "box": (100, 100, 200, 200),
            "centroid": (150, 150),
            "area": 10000,
            "ratio": 1.0,
            "region": (0, 0, 320, 240),
            "motionless_count": 100,
            "position_changes": 1,
            "attributes": attributes or [],
        }
        if observed:
            self.observed_at = self.frame_time
        if EVENT_ID not in self.state.tracked_objects:
            detection["score_history"] = [self.score] * 3
        self.state.update(
            f"frame-{self.frame_time}",
            self.frame_time,
            {EVENT_ID: detection},
            [],
            [],
            face_regions,
        )

    def start_stationary_track(self, score: float = 0.9) -> None:
        self.score = score
        # Let initial true-positive, zone and path updates publish normally.
        for _ in range(4):
            self.next_frame()
        obj = self.state.tracked_objects[EVENT_ID]
        self.assertTrue(obj.is_stationary())
        self.processor.dispatcher.reset_mock()
        self.next_frame()
        if not obj.false_positive:
            self.processor.dispatcher.publish.assert_called_once()
        self.processor.dispatcher.reset_mock()

    def set_plate(self, value: str | None, score: float | None) -> None:
        self.processor.set_object_attribute(
            EVENT_ID, "recognized_license_plate", value, score
        )

    def test_plate_setter_publishes_on_next_frame_once(self) -> None:
        self.start_stationary_track()
        self.set_plate("TEST123", 0.9)
        self.processor.dispatcher.publish.assert_not_called()

        self.next_frame()

        self.processor.dispatcher.publish.assert_called_once()
        topic, payload = self.processor.dispatcher.publish.call_args.args
        self.assertEqual(topic, "events")
        self.assertEqual(
            self.processor.dispatcher.publish.call_args.kwargs, {"retain": False}
        )
        event = json.loads(payload)
        self.assertEqual(event["type"], "update")
        self.assertIsNone(event["before"]["recognized_license_plate"])
        self.assertEqual(event["after"]["recognized_license_plate"], ["TEST123", 0.9])
        self.assertEqual(event["after"]["id"], EVENT_ID)
        self.assertEqual(event["after"]["frame_time"], self.frame_time)
        self.assertEqual(event["after"]["current_zones"], ["approach"])
        self.assertEqual(event["after"]["entered_zones"], ["approach"])
        self.assertTrue(event["after"]["stationary"])

        self.set_plate("TEST123", 0.9)
        self.next_frame(0.1)
        self.next_frame(0.1)
        self.processor.dispatcher.publish.assert_called_once()

    def test_changed_score_plate_and_clear_each_publish_once(self) -> None:
        self.start_stationary_track()
        self.set_plate("TEST123", 0.8)
        self.next_frame()
        for value, score in [("TEST123", 0.9), ("TEST456", 0.9), (None, None)]:
            with self.subTest(value=value, score=score):
                self.processor.dispatcher.reset_mock()
                self.set_plate(value, score)
                self.next_frame()
                self.processor.dispatcher.publish.assert_called_once()
                event = json.loads(self.processor.dispatcher.publish.call_args.args[1])
                self.assertEqual(
                    event["after"]["recognized_license_plate"], [value, score]
                )
                self.next_frame(0.1)
                self.processor.dispatcher.publish.assert_called_once()

    def test_plate_update_does_not_publish_false_positive(self) -> None:
        self.start_stationary_track(score=0.1)
        self.assertTrue(self.state.tracked_objects[EVENT_ID].false_positive)
        self.set_plate("TEST123", 0.9)
        self.next_frame()
        self.processor.dispatcher.publish.assert_not_called()
