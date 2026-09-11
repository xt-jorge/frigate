"""OCR publications retain the accepted sample clock instead of tracker tick time."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from frigate.models import Event
from frigate.test import test_tracked_object_publication as tracked_fixture

PLATE_FIELD = "recognized_license_plate"
CLOCK_FIELD = "recognized_license_plate_frame_time"
EVENT_ID = tracked_fixture.EVENT_ID


class TestLprSampleMetadata(unittest.TestCase):
    def setUp(self):
        self.fixture = tracked_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.processor.frame_manager.get_captured_frame.return_value = (
            self.fixture.processor.frame_manager.get.return_value
        )
        self.fixture.start_stationary_track()
        self.processor = self.fixture.processor
        self.obj = self.fixture.state.tracked_objects[EVENT_ID]

    def sample(self, clock, text="TEST123", score=0.9):
        self.processor.set_object_attribute(
            EVENT_ID, PLATE_FIELD, text, score, source_frame_time=clock
        )

    def published(self):
        self.processor.dispatcher.publish.assert_called_once()
        topic, payload = self.processor.dispatcher.publish.call_args.args
        self.assertEqual(topic, "events")
        self.assertEqual(
            self.processor.dispatcher.publish.call_args.kwargs, {"retain": False}
        )
        return json.loads(payload)

    def test_sample_at_track_start_updates_plate_and_clock_in_one_publication(self):
        clock = self.obj.obj_data["start_time"]
        self.sample(clock)
        self.assertEqual(self.obj.obj_data[PLATE_FIELD], ("TEST123", 0.9))
        self.assertEqual(self.obj.obj_data[CLOCK_FIELD], clock)
        self.processor.dispatcher.publish.assert_not_called()
        self.fixture.next_frame()
        event = self.published()
        self.assertEqual(event["after"][PLATE_FIELD], ["TEST123", 0.9])
        self.assertEqual(event["after"][CLOCK_FIELD], clock)
        self.assertEqual(event["after"]["id"], EVENT_ID)
        self.assertGreater(event["after"]["frame_time"], clock)
        self.assertIsNone(event["before"].get(CLOCK_FIELD))

    def test_same_text_and_confidence_with_new_clock_is_a_significant_update_once(self):
        first_clock = self.obj.obj_data["frame_time"] - 0.5
        self.sample(first_clock)
        self.fixture.next_frame()
        self.processor.dispatcher.reset_mock()
        second_clock = self.obj.obj_data["frame_time"]
        self.sample(second_clock)
        self.fixture.next_frame()
        event = self.published()
        self.assertEqual(event["before"][PLATE_FIELD], event["after"][PLATE_FIELD])
        self.assertEqual(event["before"][CLOCK_FIELD], first_clock)
        self.assertEqual(event["after"][CLOCK_FIELD], second_clock)
        self.fixture.next_frame()
        self.fixture.next_frame()
        self.processor.dispatcher.publish.assert_called_once()
        self.assertEqual(self.obj.obj_data[CLOCK_FIELD], second_clock)

    def test_invalid_duplicate_and_regressing_samples_leave_track_and_event_unchanged(
        self,
    ):
        clock = self.obj.obj_data["frame_time"] - 1
        self.sample(clock)
        before = copy.deepcopy(self.obj.obj_data)
        invalid = (
            float("nan"),
            float("inf"),
            float("-inf"),
            0,
            -1,
            self.obj.obj_data["start_time"] - 0.01,
            self.obj.obj_data["frame_time"] + 0.01,
            clock,
            clock - 0.01,
        )
        for rejected in invalid:
            with self.subTest(clock=rejected):
                event = Mock(label="car", data={"existing": "unchanged"})
                with patch.object(Event, "get", return_value=event):
                    self.sample(rejected, "REJECTED", 0.1)
                self.assertEqual(self.obj.obj_data, before)
                self.assertEqual(event.data, {"existing": "unchanged"})
                event.save.assert_not_called()
        self.sample(clock + 0.5, "NEW456", 0.8)
        self.assertEqual(self.obj.obj_data[PLATE_FIELD], ("NEW456", 0.8))
        self.assertEqual(self.obj.obj_data[CLOCK_FIELD], clock + 0.5)

    def test_vehicle_sample_requires_a_live_true_positive_track_with_a_persisted_event(
        self,
    ):
        for unavailable in ("missing", "false-positive", "ended"):
            with self.subTest(state=unavailable):
                if unavailable == "missing":
                    self.fixture.state.tracked_objects.pop(EVENT_ID)
                elif unavailable == "false-positive":
                    self.obj.false_positive = True
                else:
                    self.obj.obj_data["end_time"] = self.obj.obj_data["frame_time"]
                before = copy.deepcopy(self.obj.obj_data)
                event = Mock(label="car", data={"existing": "unchanged"})
                with patch.object(Event, "get", return_value=event):
                    self.sample(self.obj.obj_data["frame_time"] - 0.5)
                self.assertEqual(self.obj.obj_data, before)
                self.assertEqual(event.data, {"existing": "unchanged"})
                event.save.assert_not_called()
                self.fixture.state.tracked_objects[EVENT_ID] = self.obj
                self.obj.false_positive = False
                self.obj.obj_data.pop("end_time", None)

    def test_manual_edit_clears_sample_clock_without_resetting_the_sample_watermark(
        self,
    ):
        clock = self.obj.obj_data["frame_time"] - 1
        self.sample(clock)
        self.fixture.next_frame()
        self.processor.dispatcher.reset_mock()
        self.processor.set_object_attribute(EVENT_ID, PLATE_FIELD, "MANUAL", 0.7)
        self.assertEqual(self.obj.obj_data[PLATE_FIELD], ("MANUAL", 0.7))
        self.assertIsNone(self.obj.obj_data[CLOCK_FIELD])
        self.fixture.next_frame()
        event = self.published()
        self.assertEqual(event["after"][PLATE_FIELD], ["MANUAL", 0.7])
        self.assertIsNone(event["after"][CLOCK_FIELD])
        for rejected in (clock, clock - 0.5):
            with self.subTest(clock=rejected):
                self.sample(rejected, "REJECTED", 0.2)
                self.assertEqual(self.obj.obj_data[PLATE_FIELD], ("MANUAL", 0.7))
                self.assertIsNone(self.obj.obj_data[CLOCK_FIELD])
        self.sample(clock + 0.5, "NEXT123", 0.8)
        self.assertEqual(self.obj.obj_data[PLATE_FIELD], ("NEXT123", 0.8))
        self.assertEqual(self.obj.obj_data[CLOCK_FIELD], clock + 0.5)
        self.processor.set_object_attribute(EVENT_ID, PLATE_FIELD, None, None)
        self.sample(clock + 0.5, "REJECTED", 0.2)
        self.assertEqual(self.obj.obj_data[PLATE_FIELD], (None, None))
        self.assertIsNone(self.obj.obj_data[CLOCK_FIELD])

    def test_periodic_tracker_publication_does_not_refresh_the_ocr_sample_clock(self):
        clock = self.obj.obj_data["frame_time"] - 0.25
        self.sample(clock)
        self.fixture.next_frame()
        self.processor.dispatcher.reset_mock()
        self.fixture.frame_time += 61
        self.fixture.next_frame()
        self.fixture.next_frame()
        event = self.published()
        self.assertGreater(event["after"]["frame_time"], clock + 60)
        self.assertEqual(event["before"][CLOCK_FIELD], clock)
        self.assertEqual(event["after"][CLOCK_FIELD], clock)
        self.assertEqual(self.obj.to_dict()[CLOCK_FIELD], clock)
