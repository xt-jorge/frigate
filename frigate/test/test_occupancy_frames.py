"""Existing-zone coverage requires real inference, including quiet empty frames."""

import json
import unittest
from queue import Queue
from unittest.mock import MagicMock, patch

from frigate.test import test_tracked_object_publication as fixtures
from frigate.video.detect import process_frames


class TestOccupancyFrames(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.processor.config
        self.camera = self.config.cameras["front"]
        self.camera.detect.enabled = True
        self.camera.detect.occupancy_zones = ["approach"]

    def run_frames(
        self,
        times,
        detections=(),
        success=True,
        moving=False,
        regions=(),
        candidates=(),
    ):
        frames, output = Queue(), Queue()
        for time in times:
            frames.put(("fixture", time))
        tracker = MagicMock()
        tracker.tracked_objects = {}
        tracker.occupancy_tracks.return_value = []
        tracker.untracked_object_boxes = []
        detector = MagicMock()
        detector.last_detection_successful = success
        motion = MagicMock()
        motion.detect.return_value = []
        motion.is_calibrating.return_value = True
        stop = MagicMock()
        stop.is_set.return_value = False

        def inference(*args):
            if args[-1] is not None:
                args[-1].extend([*detections, *candidates])
            return list(detections)

        with (
            patch("frigate.video.detect.CameraConfigUpdateSubscriber") as subscriber,
            patch(
                "frigate.video.detect.get_startup_regions", return_value=list(regions)
            ),
            patch("frigate.video.detect.ptz_moving_at_frame_time", return_value=moving),
            patch("frigate.video.detect.detect", side_effect=inference) as infer,
        ):
            subscriber.return_value.check_for_updates.return_value = []
            process_frames(
                MagicMock(),
                frames,
                (240, 320),
                self.config.model,
                self.camera,
                self.fixture.processor.frame_manager,
                motion,
                detector,
                tracker,
                output,
                MagicMock(),
                stop,
                MagicMock(),
                [],
                2,
                exit_on_empty=True,
            )
        published = []
        while not output.empty():
            row = output.get_nowait()
            if row[6] is not None:
                published.append(row[6])
        return published, infer.call_count

    def test_quiet_empty_is_fresh_complete_and_bounded(self):
        frames, scans = self.run_frames([100 + tick / 100 for tick in range(101)])
        self.assertEqual(scans, 5)
        self.assertEqual(
            [f["frame_time"] for f in frames], [100, 100.25, 100.5, 100.75, 101]
        )
        self.assertTrue(all(f["complete"] and f["objects"] == [] for f in frames))
        self.assertTrue(all(f["coverage"] for f in frames))

    def test_normal_covering_region_avoids_extra_scan(self):
        frames, scans = self.run_frames([100], regions=[(0, 0, 320, 240)])
        self.assertEqual(scans, 1)
        self.assertTrue(frames[0]["complete"])

    def test_no_selected_zone_has_no_added_scan_or_publication(self):
        self.camera.detect.occupancy_zones = []
        self.assertEqual(self.run_frames([100, 101]), ([], 0))

    def test_raw_uninitialized_stationary_object_is_never_dropped(self):
        detection = ("car", 0.9, (10, 10, 80, 90), 5600, 0.875, (0, 0, 320, 240))
        frames, _ = self.run_frames([100, 100.25], detections=[detection])
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(
                frame["objects"],
                [
                    {
                        "label": "car",
                        "box": [10, 10, 80, 90],
                        "detector_observed_at": frame["frame_time"],
                    }
                ],
            )

    def test_timeout_cannot_publish_complete_empty(self):
        frames, _ = self.run_frames([100], success=False)
        self.assertFalse(frames[0]["complete"])
        self.assertEqual(frames[0]["coverage"], [])

    def test_weak_candidate_only_sets_region_uncertainty_in_existing_mqtt_frame(self):
        weak = ("car", 0.2, (10, 10, 80, 90), 5600, 0.875, (0, 0, 320, 240))
        frames, scans = self.run_frames([100, 100.25], candidates=[weak])
        self.assertEqual(scans, 2)
        for frame in frames:
            self.assertTrue(frame["complete"])
            self.assertEqual(frame["objects"], [])
            self.assertEqual(frame["tracks"], [])
            self.assertTrue(frame["regions"][0]["uncertain"])

    def test_disabled_detection_missing_zone_and_ptz_are_unknown(self):
        for disabled, missing, moving in [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]:
            with self.subTest(disabled=disabled, missing=missing, moving=moving):
                self.camera.detect.enabled = not disabled
                self.camera.detect.occupancy_zones = [
                    "missing" if missing else "approach"
                ]
                frames, scans = self.run_frames([100], moving=moving)
                self.assertFalse(frames[0]["complete"])
                self.assertEqual(scans, 0)

    def test_repeated_or_backwards_capture_cannot_renew_coverage(self):
        frames, scans = self.run_frames([100, 100, 99, 100.25])
        self.assertEqual([f["frame_time"] for f in frames], [100, 100.25])
        self.assertEqual(scans, 2)

    def test_complete_empty_frame_reaches_nonretained_mqtt_publication(self):
        frames, _ = self.run_frames([100])
        processor = self.fixture.processor
        processor.stop_event = MagicMock()
        processor.stop_event.is_set.side_effect = [False, True, True]
        processor.camera_config_subscriber = MagicMock()
        processor.camera_config_subscriber.check_for_updates.return_value = {}
        processor.sub_label_subscriber = MagicMock()
        processor.sub_label_subscriber.check_for_update.return_value = None
        processor.event_end_subscriber = MagicMock()
        processor.detection_publisher = MagicMock()
        processor.last_motion_detected = {}
        processor.tracked_objects_queue = Queue()
        processor.tracked_objects_queue.put(
            ("front", "fixture", 100, {}, [], [], frames[0], None)
        )
        processor.run()
        messages = [
            call
            for call in processor.dispatcher.publish.call_args_list
            if call.args[0] == "occupancy_frames"
        ]
        self.assertEqual(len(messages), 1)
        self.assertEqual(json.loads(messages[0].args[1]), frames[0])
        self.assertEqual(messages[0].kwargs, {"retain": False})
