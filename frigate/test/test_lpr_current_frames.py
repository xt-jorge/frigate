"""Current detector packets drive bounded OCR with original sample custody."""

import math
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import frigate.embeddings
from frigate.config.camera.camera import CameraTypeEnum
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)
from frigate.embeddings.maintainer import EmbeddingMaintainer
from frigate.events.types import EventStateEnum, EventTypeEnum


def camera_config(dedicated=False):
    return SimpleNamespace(
        enabled=True,
        type=CameraTypeEnum.lpr if dedicated else CameraTypeEnum.generic,
        detect=SimpleNamespace(
            enabled=not dedicated, fps=15, stationary=SimpleNamespace(threshold=150)
        ),
        objects=SimpleNamespace(track=[] if dedicated else ["license_plate"]),
        lpr=SimpleNamespace(enabled=True, min_area=1, expire_time=10),
        frame_shape=(32, 32),
        frame_shape_yuv=(48, 32),
        motion=SimpleNamespace(rasterized_mask=np.ones((32, 32))),
    )


def packet(camera="a", frame_time=1000.0, ids=("track",)):
    objects = [
        {
            "id": track,
            "camera": camera,
            "frame_time": frame_time,
            "label": "car",
            "false_positive": False,
            "end_time": None,
            "position_changes": 1,
            "stationary": False,
            "box": [1, 1, 20, 20],
        }
        for track in ids
    ]
    return (camera, f"{camera}_frame0", frame_time, objects, [[1, 1, 2, 2]], [])


class TestCurrentLprScheduling(unittest.TestCase):
    def setUp(self):
        self.assertIs(frigate.embeddings.EmbeddingMaintainer, EmbeddingMaintainer)
        self.owner = EmbeddingMaintainer.__new__(EmbeddingMaintainer)
        self.owner.config = SimpleNamespace(
            cameras={"a": camera_config(), "b": camera_config(True)}
        )
        self.owner._latest_lpr_frames = {}
        self.owner._lpr_camera_attempt = {}
        self.owner._lpr_track_attempt = {}
        self.owner._last_lpr_attempt = -math.inf
        self.processor = MagicMock(spec=LicensePlateRealTimeProcessor)
        self.processor.lp_objects = ["car"]
        self.processor.stationary_scan_duration = 5
        self.owner.realtime_processors = [self.processor]
        self.owner.frame_manager = MagicMock()
        self.owner.frame_manager.get_captured_frame.return_value = np.zeros(
            (48, 32), dtype=np.uint8
        )
        self.queue = deque()
        self.owner.detection_subscriber = MagicMock()
        self.owner.detection_subscriber.check_for_update.side_effect = lambda **_: (
            ("video", self.queue.popleft()) if self.queue else (None, None)
        )
        self.now = 1000.0
        self.tick = 0.0
        clock = patch("frigate.embeddings.maintainer.datetime")
        self.clock = clock.start()
        self.clock.datetime.now.return_value.timestamp.side_effect = lambda: self.now
        self.addCleanup(clock.stop)
        timer = patch(
            "frigate.embeddings.maintainer.time.monotonic",
            side_effect=lambda: self.tick,
        )
        timer.start()
        self.addCleanup(timer.stop)

    def run_packet(self, value):
        self.queue.append(value)
        self.owner._process_frame_updates()

    def test_first_eligible_sample_is_immediate_without_event_or_thumbnail(self):
        self.run_packet(packet())
        self.assertEqual(self.processor.process_frame.call_count, 1)
        self.assertEqual(
            self.processor.process_frame.call_args.kwargs, {"source_frame_time": 1000.0}
        )
        self.owner.frame_manager.get_captured_frame.assert_called_once_with(
            "a_frame0", (48, 32), 1000.0
        )

    def test_lifecycle_callback_does_not_run_ocr_again(self):
        self.owner.config.semantic_search = SimpleNamespace(enabled=False)
        self.owner.post_processors = []
        self.owner.event_subscriber = MagicMock()
        self.owner.event_subscriber.check_for_update.return_value = (
            EventTypeEnum.tracked_object,
            EventStateEnum.update,
            "a",
            "a_frame0",
            packet()[3][0],
        )
        self.owner.frame_manager.get.return_value = np.zeros((48, 32), np.uint8)
        self.owner._process_updates()
        self.processor.process_frame.assert_not_called()
        self.run_packet(packet())
        self.assertEqual(self.processor.process_frame.call_count, 1)

    def test_latest_only_and_bounded_drain_under_continuous_input(self):
        self.queue.extend(packet(frame_time=997 + i / 100) for i in range(70))
        self.owner._process_frame_updates()
        self.assertEqual(len(self.queue), 38)
        self.assertEqual(self.processor.process_frame.call_count, 1)
        self.assertEqual(
            self.processor.process_frame.call_args.kwargs["source_frame_time"], 997.31
        )
        self.assertLessEqual(len(self.owner._latest_lpr_frames), 2)

    def test_global_camera_ceilings_and_dedicated_coexistence(self):
        self.queue.extend([packet(), packet("b")])
        self.owner._process_frame_updates()
        self.tick = 0.249
        self.owner._process_frame_updates()
        self.assertEqual(self.processor.process_frame.call_count, 1)
        self.tick = 0.250
        self.owner._process_frame_updates()
        self.assertEqual(
            self.processor.process_frame.call_args.args[0], {"camera": "b"}
        )
        self.assertTrue(self.processor.process_frame.call_args.args[2])
        self.tick = 0.499
        self.run_packet(packet())
        self.assertEqual(self.processor.process_frame.call_count, 2)
        self.tick = 0.500
        self.owner._process_frame_updates()
        self.assertEqual(self.processor.process_frame.call_count, 3)

    def test_slow_inference_returns_to_other_camera_then_least_sampled_track(self):
        self.processor.process_frame.side_effect = lambda *_args, **_kwargs: setattr(
            self, "tick", self.tick + 2
        )
        self.queue.extend([packet(ids=("one", "two")), packet("b")])
        self.owner._process_frame_updates()
        self.run_packet(packet(ids=("one", "two")))
        self.owner._process_frame_updates()
        calls = self.processor.process_frame.call_args_list
        self.assertEqual(
            [
                calls[0].args[0]["id"],
                calls[1].args[0]["camera"],
                calls[2].args[0]["id"],
            ],
            ["one", "b", "two"],
        )

    def test_missing_overwritten_and_stale_frames_never_infer(self):
        self.owner.frame_manager.get_captured_frame.return_value = None
        self.run_packet(packet())
        self.processor.process_frame.assert_not_called()
        self.assertFalse(self.owner._latest_lpr_frames)
        self.assertEqual(self.owner._last_lpr_attempt, -math.inf)
        self.owner.frame_manager.get_captured_frame.reset_mock()
        for clock in [990.0, 1000.1, float("nan")]:
            self.run_packet(packet(frame_time=clock))
        self.owner.frame_manager.get_captured_frame.assert_not_called()

    def test_packet_does_not_make_old_false_positive_ended_or_wrong_scope_current(self):
        for replacement in [
            {"false_positive": True},
            {"end_time": 999},
            {"camera": "b"},
            {"frame_time": 999},
            {"label": "person"},
            {"box": [-1, 1, 20, 20]},
            {"position_changes": 0},
            {"stationary": True, "motionless_count": 226},
        ]:
            with self.subTest(replacement=replacement):
                value = packet()
                value[3][0].update(replacement)
                self.run_packet(value)
        self.processor.process_frame.assert_not_called()


class TestLprSamplePair(unittest.TestCase):
    def setUp(self):
        self.processor = LicensePlateProcessingMixin.__new__(
            LicensePlateProcessingMixin
        )
        p = self.processor
        p.config = SimpleNamespace(cameras={"a": camera_config()})
        p.metrics = MagicMock()
        p.plates_rec_second = MagicMock()
        p.plates_det_second = MagicMock()
        p.plate_rec_speed = MagicMock()
        p.plate_det_speed = MagicMock()
        p.lp_objects = ["car"]
        p.stationary_scan_duration = 5
        p.cluster_threshold = 0.8
        p.similarity_threshold = 0.8
        p.lpr_config = SimpleNamespace(
            recognition_threshold=0.7,
            min_plate_length=3,
            format=None,
            known_plates={},
            match_distance=0,
        )
        p.detected_license_plates = {}
        p.camera_current_cars = {}
        p.sub_label_publisher = MagicMock()
        p.event_metadata_publisher = MagicMock()
        p.requestor = MagicMock()
        p._process_license_plate = MagicMock()
        self.obj = {
            "camera": "a",
            "id": "track",
            "label": "license_plate",
            "position_changes": 1,
            "stationary": False,
            "box": [5, 5, 25, 15],
        }
        self.frame = np.zeros((48, 32), np.uint8)

    def sample(self, text, confidence, clock):
        self.processor._process_license_plate.return_value = (
            [text],
            [[confidence] * len(text)],
            [100],
        )
        self.processor.lpr_process(self.obj, self.frame, source_frame_time=clock)

    def test_same_text_uses_new_actual_confidence_and_capture_not_best_or_processing_time(
        self,
    ):
        self.sample("ABC123", 0.99, 100.0)
        self.sample("ABC123", 0.80, 101.0)
        value = self.processor.sub_label_publisher.publish.call_args.args[0]
        self.assertEqual(value[:3], ("track", "recognized_license_plate", "ABC123"))
        self.assertAlmostEqual(value[3], 0.80)
        self.assertEqual(value[4], 101.0)
        self.assertEqual(
            self.processor.detected_license_plates["track"]["plates"][-1]["timestamp"],
            101.0,
        )

    def test_old_cluster_representative_cannot_borrow_new_clock(self):
        self.sample("ABC123", 0.99, 100.0)
        self.processor.sub_label_publisher.reset_mock()
        self.sample("ABC128", 0.80, 101.0)
        self.processor.sub_label_publisher.publish.assert_not_called()
        self.assertEqual(
            self.processor.detected_license_plates["track"][
                "recognized_license_plate_frame_time"
            ],
            100.0,
        )

    def test_failed_or_duplicate_sample_never_refreshes_the_previous_pair(self):
        self.sample("ABC123", 0.9, 100.0)
        self.processor.sub_label_publisher.reset_mock()
        self.sample("ABC123", 0.5, 101.0)
        self.sample("ABC123", 0.95, 100.0)
        self.processor.sub_label_publisher.publish.assert_not_called()
        self.assertEqual(
            self.processor.detected_license_plates["track"][
                "recognized_license_plate_frame_time"
            ],
            100.0,
        )
        self.sample("ABC123", 0.92, 101.0)
        self.assertEqual(
            self.processor.sub_label_publisher.publish.call_args.args[0][4], 101.0
        )

    def test_dedicated_mode_publishes_the_same_actual_source_clock(self):
        self.processor.config.cameras["a"] = camera_config(True)
        self.processor._detect_license_plate = MagicMock(return_value=(5, 5, 25, 15))
        self.processor._process_license_plate.return_value = (
            ["ABC123"],
            [[0.9] * 6],
            [100],
        )
        self.processor.lpr_process(
            {"camera": "a"}, self.frame, True, source_frame_time=100.0
        )
        created = self.processor.event_metadata_publisher.publish.call_args.args[0]
        self.assertEqual(created[0], 100.0)
        samples = [
            call.args[0]
            for call in self.processor.sub_label_publisher.publish.call_args_list
            if len(call.args[0]) == 5 and call.args[0][1] == "recognized_license_plate"
        ]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][4], 100.0)
        self.assertAlmostEqual(samples[0][3], 0.9)

    def test_missing_capture_is_not_replaced_by_wall_clock(self):
        for clock in [None, 0, -1, True, float("nan"), float("inf")]:
            with self.subTest(clock=clock):
                self.processor.lpr_process(
                    self.obj, self.frame, source_frame_time=clock
                )
        self.processor._process_license_plate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
