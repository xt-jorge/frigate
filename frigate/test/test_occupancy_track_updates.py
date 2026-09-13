"""Selected-camera cadence uses actual detector samples through real publication."""

import json
import unittest
from itertools import pairwise

from frigate.test import test_tracked_object_publication as fixtures


class TestOccupancyTrackUpdates(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.start_stationary_track()
        self.state = self.fixture.state
        self.config = self.state.camera_config.detect
        self.config.occupancy_zones = ["approach"]
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
        for tick in range(48):
            time = 110 + tick / 4
            self.frame(time, time)
        self.frame(122, 122, box=(110, 100, 210, 200))
        events = self.events()
        clocks = [event["after"]["detector_observed_at"] for event in events]
        self.assertGreater(len(clocks), 20)
        self.assertLessEqual(max(b - a for a, b in pairwise(clocks)), 0.501)
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

    def test_current_track_trigger_is_bounded_to_two_hz(self):
        for tick in range(100):
            time = 110 + tick / 100
            self.frame(time, time)
        self.assertLessEqual(len(self.events()), 2)
        self.assertGreaterEqual(len(self.events()), 1)

    def test_predicted_frame_does_not_publish_presence(self):
        for time, clock, stored in [
            (112, 110, 110),
        ]:
            self.frame(time, clock, stored)
        self.assertEqual(self.events(), [])

    def test_release_only_camera_keeps_current_stationary_presence(self):
        self.config.occupancy_zones = []
        for tick in range(48):
            time = 110 + tick / 4
            self.frame(time, time)
        self.assertGreater(len(self.events()), 20)

    def test_false_positive_does_not_get_detector_trigger(self):
        obj = self.state.tracked_objects[fixtures.EVENT_ID]
        obj.false_positive = True
        obj.computed_score = 0.1
        obj.score_history = [0.1] * 10
        self.initial["score"] = 0.1
        self.initial["score_history"] = [0.1] * 10
        self.frame(110, 110)
        self.assertEqual(self.events(), [])

    def test_current_stationary_capture_publishes_without_rejuvenating_detector_clock(
        self,
    ):
        self.frame(110, 110)
        self.assertEqual(len(self.events()), 1)
        self.frame(111, 110)
        self.frame(112, 110, stored=110)
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(self.events()[-1]["after"]["frame_time"], 111)
        self.assertEqual(self.events()[-1]["after"]["detector_observed_at"], 110)
