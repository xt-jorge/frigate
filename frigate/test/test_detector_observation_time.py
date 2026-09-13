"""Detector clocks survive real tracker prediction and stationary refreshes."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frigate.test import test_tracked_object_publication as publication_fixture
from frigate.track.norfair_tracker import NorfairTracker
from frigate.video.detect import process_frames


class TestDetectorObservationTime(unittest.TestCase):
    def setUp(self):
        self.fixture = publication_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        config = self.fixture.processor.config.cameras["front"]
        config.detect.min_initialized = 0
        self.tracker = NorfairTracker(
            config,
            SimpleNamespace(autotracker_enabled=SimpleNamespace(value=False)),
        )
        self.tracker.frame_manager = MagicMock()
        self.detection = (
            "car",
            0.95,
            (100, 100, 200, 200),
            10000,
            1.0,
            (0, 0, 320, 240),
        )

    def observe(self, frame, original):
        self.tracker.match_and_update(
            "fixture", frame, [self.detection], detector_observed_at=[original]
        )
        self.assertEqual(len(self.tracker.tracked_objects), 1)
        return next(iter(self.tracker.tracked_objects.values()))

    def test_refresh_prediction_and_real_detection_have_distinct_clocks(self):
        obj = self.observe(100.0, 100.0)
        identity = obj["id"]
        self.tracker.update_frame_times("fixture", 101.0)
        obj = self.tracker.tracked_objects[identity]
        self.assertEqual(obj["frame_time"], 101.0)
        self.assertEqual(obj["detector_observed_at"], 100.0)
        self.tracker.match_and_update("fixture", 102.0, [])
        self.assertEqual(obj["detector_observed_at"], 100.0)
        self.assertEqual(self.tracker.disappeared[identity], 1)
        obj = self.observe(103.0, 103.0)
        self.assertEqual(obj["id"], identity)
        self.assertEqual(obj["detector_observed_at"], 103.0)
        obj = self.observe(
            104.0, 103.0
        )  # stationary seed in a frame with other detections
        self.assertEqual(obj["frame_time"], 104.0)
        self.assertEqual(obj["detector_observed_at"], 103.0)

    def test_missing_provenance_does_not_invent_a_detector_clock(self):
        self.tracker.match_and_update("fixture", 100.0, [self.detection])
        obj = next(iter(self.tracker.tracked_objects.values()))
        self.assertIsNone(obj["detector_observed_at"])
        self.tracker.update_frame_times("fixture", 101.0)
        self.assertIsNone(obj["detector_observed_at"])

    def test_clock_is_exposed_by_actual_tracked_event_serialization(self):
        self.fixture.start_stationary_track()
        obj = self.fixture.state.tracked_objects["100-synthetic"]
        obj.obj_data["detector_observed_at"] = 99.5
        self.assertEqual(obj.to_dict()["detector_observed_at"], 99.5)

    def test_pipeline_keeps_stationary_seed_clock_with_new_detector_result(self):
        old = self.observe(100.0, 100.0)
        old["motionless_count"] = 10000
        fresh = ("car", 0.95, (10, 10, 50, 50), 1600, 1.0, (0, 0, 320, 240))
        frames = Queue()
        frames.put(("fixture", 101.0))
        output = Queue()
        motion = MagicMock()
        motion.detect.return_value = []
        motion.is_calibrating.return_value = True
        stop = MagicMock()
        stop.is_set.return_value = False
        config = self.fixture.processor.config
        config.cameras["front"].detect.enabled = True
        with (
            patch("frigate.video.detect.CameraConfigUpdateSubscriber") as subscriber,
            patch("frigate.video.detect.get_cluster_candidates", return_value=[]),
            patch(
                "frigate.video.detect.get_startup_regions",
                return_value=[(0, 0, 320, 240)],
            ),
            patch("frigate.video.detect.detect", return_value=[fresh]),
        ):
            subscriber.return_value.check_for_updates.return_value = []
            process_frames(
                MagicMock(),
                frames,
                (240, 320),
                config.model,
                config.cameras["front"],
                self.fixture.processor.frame_manager,
                motion,
                MagicMock(),
                self.tracker,
                output,
                MagicMock(),
                stop,
                MagicMock(),
                [],
                exit_on_empty=True,
            )
        published = output.get_nowait()[3]
        clocks = {
            tuple(obj["box"]): obj["detector_observed_at"] for obj in published.values()
        }
        self.assertEqual(clocks, {self.detection[2]: 100.0, fresh[2]: 101.0})
