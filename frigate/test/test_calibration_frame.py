"""Calibration pixels and vehicle geometry must describe one published frame."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from frigate.api.media import calibration_frame
from frigate.test import test_tracked_object_publication as tracked_fixture


class TestCalibrationFrame(unittest.TestCase):
    def setUp(self):
        self.fixture = tracked_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.start_stationary_track()
        self.state = self.fixture.state
        self.state.camera_config.detect.enabled = True
        self.request = SimpleNamespace(
            app=SimpleNamespace(detected_frames_processor=self.fixture.processor)
        )

    def capture(self, now=None):
        with patch(
            "frigate.api.media.time.time",
            return_value=now
            if now is not None
            else self.state.current_frame_time + 0.1,
        ):
            return asyncio.run(calibration_frame(self.request, tracked_fixture.CAMERA))

    def test_boxes_are_frozen_before_later_tracker_mutation(self):
        image, capture, boxes = self.state.get_calibration_frame()
        obj = self.state.tracked_objects[tracked_fixture.EVENT_ID]
        obj.obj_data["box"] = [10, 20, 30, 40]
        obj.obj_data["frame_time"] += 1
        image2, capture2, boxes2 = self.state.get_calibration_frame()
        self.assertEqual(boxes, ((100, 100, 200, 200),))
        self.assertEqual(boxes2, boxes)
        self.assertEqual(capture2, capture)
        np.testing.assert_array_equal(image2, image)
        image2[:] = 0
        np.testing.assert_array_equal(self.state.get_calibration_frame()[0], image)

    def test_frame_and_metadata_advance_together_at_publication(self):
        before = self.state.get_calibration_frame()
        self.fixture.processor.frame_manager.get_captured_frame.return_value[:] = 128
        # The callback happens after live objects mutate but before the new image
        # is published. Reading here must still return the previous complete pair.
        observed = []
        self.state.callbacks["camera_activity"].append(
            lambda *args: observed.append(self.state.get_calibration_frame())
        )
        self.fixture.next_frame()
        self.assertTrue(observed)
        np.testing.assert_array_equal(observed[0][0], before[0])
        self.assertEqual(observed[0][1:], before[1:])
        after = self.state.get_calibration_frame()
        self.assertGreater(after[1], before[1])
        self.assertFalse(np.array_equal(after[0], before[0]))

    def test_jpeg_header_has_the_exact_capture_box_and_no_identity(self):
        response = self.capture()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        metadata = json.loads(response.headers["x-calibration-frame"])
        self.assertEqual(
            metadata,
            {
                "state": "matched",
                "capturedAtMs": int(self.state.current_frame_time * 1000),
                "width": 320,
                "height": 240,
                "vehicles": [{"box": [100, 100, 200, 200]}],
            },
        )
        decoded = cv2.imdecode(np.frombuffer(response.body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (240, 320))
        self.assertEqual(
            float(response.headers["x-frame-time"]), self.state.current_frame_time
        )

    def test_empty_frame_is_a_valid_drawing_frame(self):
        self.state.update("empty", self.state.current_frame_time + 1, {}, [], [])
        metadata = json.loads(self.capture().headers["x-calibration-frame"])
        self.assertEqual(metadata["vehicles"], [])
        self.assertEqual(metadata["state"], "matched")

    def test_false_positive_does_not_supply_a_vehicle_point(self):
        obj = self.state.tracked_objects[tracked_fixture.EVENT_ID]
        obj.false_positive = True
        obj.score_history = [0.1, 0.1, 0.1]
        self.fixture.score = 0.1
        self.fixture.next_frame()
        self.assertEqual(
            json.loads(self.capture().headers["x-calibration-frame"])["vehicles"], []
        )

    def test_stale_future_disabled_and_over_budget_capture_are_refused(self):
        for age in (-1, 6):
            with self.subTest(age=age):
                self.assertEqual(
                    self.capture(self.state.current_frame_time + age).status_code, 503
                )
        self.state.camera_config.detect.enabled = False
        self.assertEqual(self.capture().status_code, 404)
        self.state.camera_config.detect.enabled = True
        self.state._current_frame_vehicles = tuple([(1, 1, 2, 2)] * 33)
        self.assertEqual(self.capture().status_code, 503)

    def test_half_millisecond_rounding_matches_the_consumer(self):
        self.state.current_frame_time = 100.0005
        self.assertEqual(
            json.loads(self.capture().headers["x-calibration-frame"])["capturedAtMs"],
            100001,
        )
