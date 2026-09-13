"""Selected-camera cadence uses actual detector samples through real publication."""

import json
import unittest

from pydantic import ValidationError

from frigate.config.camera.detect import DetectConfig
from frigate.test import test_tracked_object_publication as fixtures


class TestVehicleDetectorUpdates(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.start_stationary_track()
        self.state = self.fixture.state
        self.config = self.state.camera_config.detect
        self.config.vehicle_detector_updates = True
        self.initial = dict(self.state.tracked_objects[fixtures.EVENT_ID].obj_data)

    def frame(self, time, detector=None, stored=None, box=None):
        data = dict(self.initial)
        data.update(
            frame_time=time if stored is None else stored, detector_observed_at=detector
        )
        if box is not None:
            data["box"] = box
            data["centroid"] = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        self.state.update(f"frame-{time}", time, {fixtures.EVENT_ID: data}, [], [])

    def events(self):
        return [
            json.loads(call.args[1])
            for call in self.fixture.processor.dispatcher.publish.call_args_list
            if call.args[0] == "events"
        ]

    def test_stopped_over_ten_seconds_and_resumption_keep_real_event_identity(self):
        for tick in range(60):
            time = 110 + tick / 5
            self.frame(time, time)
        self.frame(122, 122, box=(110, 100, 210, 200))
        events = self.events()
        clocks = [event["after"]["detector_observed_at"] for event in events]
        self.assertGreater(len(clocks), 30)
        self.assertLessEqual(max(b - a for a, b in zip(clocks, clocks[1:])), 0.401)
        self.assertEqual(clocks[-1], 122)
        self.assertTrue(
            all(event["after"]["id"] == fixtures.EVENT_ID for event in events)
        )
        self.assertTrue(
            all(
                event["after"]["frame_time"] == event["after"]["detector_observed_at"]
                for event in events
            )
        )
        self.assertEqual(events[-1]["after"]["box"], [110, 100, 210, 200])

    def test_detector_trigger_is_bounded_to_five_hz(self):
        for tick in range(100):
            time = 110 + tick / 100
            self.frame(time, time)
        self.assertLessEqual(len(self.events()), 5)
        self.assertGreaterEqual(len(self.events()), 4)

    def test_missing_stale_predicted_or_invalid_clock_does_not_publish(self):
        for time, clock, stored in [
            (110, None, None),
            (111, 110, None),
            (112, 110, 110),
            (113, float("nan"), None),
            (114, True, None),
        ]:
            self.frame(time, clock, stored)
        self.assertEqual(self.events(), [])

    def test_disabled_camera_keeps_existing_stationary_publication(self):
        self.config.vehicle_detector_updates = False
        for tick in range(60):
            time = 110 + tick / 5
            self.frame(time, time)
        self.assertEqual(self.events(), [])

    def test_nonvehicle_and_false_positive_do_not_get_detector_trigger(self):
        for label, score in [("person", 0.9), ("car", 0.1)]:
            with self.subTest(label=label, score=score):
                obj = self.state.tracked_objects[fixtures.EVENT_ID]
                obj.false_positive = score < 0.5
                obj.computed_score = score
                obj.score_history = [score] * 10
                self.initial["label"] = label
                self.initial["score"] = score
                self.initial["score_history"] = [score] * 10
                obj.obj_data["label"] = label
                obj.previous["label"] = label
                self.frame(110, 110)
                self.assertEqual(self.events(), [])

    def test_repeated_detector_clock_cannot_publish_again(self):
        self.frame(110, 110)
        self.assertEqual(len(self.events()), 1)
        self.frame(111, 110)
        self.frame(112, 110, stored=110)
        self.assertEqual(len(self.events()), 1)

    def test_enabled_requires_five_fps_without_changing_default(self):
        self.assertFalse(DetectConfig().vehicle_detector_updates)
        self.assertEqual(DetectConfig(fps=15, vehicle_detector_updates=True).fps, 15)
        with self.assertRaises(ValidationError):
            DetectConfig(fps=4, vehicle_detector_updates=True)
