"""The pixels a detection was measured on must survive the whole handoff.

Detection takes longer than the capture ring's lifetime, so by the time the
tracker, CameraState or OCR run, the camera has already reused the capture slot
the frame arrived in. These tests force that reuse and assert every consumer
either reads the exact original pixels or refuses.

Publication retention is bounded, not a lease. A consumer that arrives after
its generation has rolled out of the ring must refuse, never read a newer one.

``test_capture_frame_custody`` covers the shared-memory primitive itself on
Linux. These tests drive the real production call paths against a frame manager
that reproduces the same contract in process, so the ownership wiring is
exercised on any platform.
"""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.camera.maintainer import CameraMaintainer
from frigate.const import MAX_PUBLICATION_FRAMES, MIN_CAPTURE_FRAMES
from frigate.test import test_tracked_object_publication as publication_fixture
from frigate.track.norfair_tracker import NorfairTracker
from frigate.util.image import (
    FrameManager,
    capture_frame_name,
    partition_frame_slots,
    publication_frame_name,
)
from frigate.video.detect import process_frames

CAMERA = "front"
FRAME_SHAPE = (240, 320)
YUV_SHAPE = (360, 320)


class RingFrameManager(FrameManager):
    """In-process stand-in with the production slot contract.

    A slot exists only if it was created, a write must advance the clock, and a
    read returns an owned copy only when the caller names the exact clock the
    slot currently holds.
    """

    def __init__(self) -> None:
        self.slots: dict[str, tuple[bytes, float]] = {}
        self.sizes: dict[str, int] = {}
        self.rejected_writes = 0

    def create_captured_frame(self, name: str, size: int) -> None:
        self.sizes[name] = size
        self.slots[name] = (bytes(size), 0.0)

    def write_captured_frame(self, name: str, pixels: bytes, frame_time: float) -> bool:
        if name not in self.slots or len(pixels) != self.sizes[name]:
            self.rejected_writes += 1
            return False
        if not isinstance(frame_time, (int, float)) or frame_time <= 0:
            self.rejected_writes += 1
            return False
        if frame_time <= self.slots[name][1]:
            self.rejected_writes += 1
            return False
        self.slots[name] = (bytes(pixels), frame_time)
        return True

    def get_captured_frame(self, name: str, shape, frame_time: float):
        slot = self.slots.get(name)
        if slot is None or frame_time <= 0:
            return None
        pixels, held = slot
        if held != frame_time or len(pixels) != int(np.prod(shape)):
            return None
        return np.frombuffer(pixels, dtype=np.uint8).reshape(shape).copy()

    # unused by these paths, but the port requires them
    def create(self, name: str, size: int):
        raise NotImplementedError

    def write(self, name: str):
        raise NotImplementedError

    def get(self, name: str, timeout_ms: int = 0):
        raise NotImplementedError

    def close(self, name: str) -> None:
        pass

    def delete(self, name: str) -> None:
        self.slots.pop(name, None)
        self.sizes.pop(name, None)

    def cleanup(self) -> None:
        self.slots.clear()
        self.sizes.clear()


def yuv(fill: int) -> np.ndarray:
    return np.full(YUV_SHAPE, fill, dtype=np.uint8)


class TestSlotPartition(unittest.TestCase):
    """The split reallocates the budget a camera already had."""

    def test_deployed_and_runtime_budgets(self):
        self.assertEqual(partition_frame_slots(16), (10, 6))
        self.assertEqual(partition_frame_slots(10), (4, 6))

    def test_small_budgets_shrink_publication_before_capture(self):
        self.assertEqual(partition_frame_slots(8), (4, 4))
        self.assertEqual(partition_frame_slots(5), (4, 1))
        self.assertEqual(partition_frame_slots(2), (1, 1))

    def test_a_budget_too_small_for_both_families_allocates_neither(self):
        self.assertEqual(partition_frame_slots(1), (0, 0))
        self.assertEqual(partition_frame_slots(0), (0, 0))

    def test_partition_never_exceeds_the_existing_budget(self):
        for total in range(0, 65):
            capture, publication = partition_frame_slots(total)
            with self.subTest(total=total):
                self.assertGreaterEqual(capture, 0)
                self.assertGreaterEqual(publication, 0)
                self.assertLessEqual(capture + publication, total)
                self.assertLessEqual(publication, MAX_PUBLICATION_FRAMES)
                if publication:
                    self.assertGreaterEqual(capture, 1)
                if total >= MIN_CAPTURE_FRAMES + MAX_PUBLICATION_FRAMES:
                    self.assertEqual(publication, MAX_PUBLICATION_FRAMES)
                    self.assertEqual(capture, total - MAX_PUBLICATION_FRAMES)


class TestCameraFrameSlotLifecycle(unittest.TestCase):
    """Both families exist before a frame can publish, and both are unlinked."""

    def maintainer(self, shm_count: int) -> CameraMaintainer:
        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.frame_manager = MagicMock()
        maintainer.frame_manager.shm_store = {}
        maintainer.shm_count = shm_count
        return maintainer

    def created(self, maintainer: CameraMaintainer) -> list[str]:
        return [
            call.args[0]
            for call in maintainer.frame_manager.create_captured_frame.call_args_list
        ]

    def config(self) -> SimpleNamespace:
        return SimpleNamespace(name=CAMERA, frame_shape_yuv=YUV_SHAPE)

    def test_both_families_are_created_within_the_budget(self):
        maintainer = self.maintainer(16)
        partition = maintainer._CameraMaintainer__create_camera_frame_slots(
            self.config(), False
        )
        self.assertEqual(partition, (10, 6))
        names = self.created(maintainer)
        self.assertEqual(len(names), 16)
        self.assertEqual(
            names,
            [capture_frame_name(CAMERA, i) for i in range(10)]
            + [publication_frame_name(CAMERA, i) for i in range(6)],
        )

    def test_runtime_camera_gets_the_runtime_budget(self):
        maintainer = self.maintainer(16)
        partition = maintainer._CameraMaintainer__create_camera_frame_slots(
            self.config(), True
        )
        self.assertEqual(partition, (4, 6))
        self.assertEqual(len(self.created(maintainer)), 10)

    def test_a_budget_too_small_creates_nothing_and_publishes_nothing(self):
        maintainer = self.maintainer(1)
        partition = maintainer._CameraMaintainer__create_camera_frame_slots(
            self.config(), False
        )
        self.assertEqual(partition, (0, 0))
        self.assertEqual(self.created(maintainer), [])

    def test_no_shm_budget_creates_nothing(self):
        maintainer = self.maintainer(0)
        self.assertEqual(
            maintainer._CameraMaintainer__create_camera_frame_slots(
                self.config(), False
            ),
            (0, 0),
        )
        self.assertEqual(self.created(maintainer), [])

    def test_unlink_removes_both_families_and_nothing_else(self):
        maintainer = self.maintainer(16)
        maintainer.frame_manager.shm_store = {
            capture_frame_name(CAMERA, 0): object(),
            capture_frame_name(CAMERA, 1): object(),
            publication_frame_name(CAMERA, 0): object(),
            publication_frame_name(CAMERA, 1): object(),
            capture_frame_name("side", 0): object(),
            publication_frame_name("side", 0): object(),
            # detector request/response buffers, sized by the model
            CAMERA: object(),
            f"out-{CAMERA}": object(),
        }
        maintainer._CameraMaintainer__unlink_camera_frame_slots(CAMERA)
        deleted = sorted(
            call.args[0] for call in maintainer.frame_manager.delete.call_args_list
        )
        self.assertEqual(
            deleted,
            [
                capture_frame_name(CAMERA, 0),
                capture_frame_name(CAMERA, 1),
                publication_frame_name(CAMERA, 0),
                publication_frame_name(CAMERA, 1),
            ],
        )

    def test_slots_exist_before_the_detector_process_is_started(self):
        """The detector publishes; its slots cannot be created after it runs."""
        maintainer = self.maintainer(16)
        maintainer.config = SimpleNamespace(
            model=SimpleNamespace(merged_labelmap={}),
            logger=None,
        )
        maintainer.detection_queue = MagicMock()
        maintainer.detected_frames_queue = MagicMock()
        maintainer.camera_metrics = {CAMERA: MagicMock()}
        maintainer.ptz_metrics = {CAMERA: MagicMock()}
        maintainer.region_grids = {CAMERA: []}
        maintainer.camera_processes = {}
        maintainer.camera_stop_events = {}
        config = SimpleNamespace(
            name=CAMERA, frame_shape_yuv=YUV_SHAPE, enabled_in_config=True
        )
        order: list[str] = []
        maintainer.frame_manager.create_captured_frame.side_effect = lambda name, size: (
            order.append(f"create:{name}")
        )
        with patch("frigate.camera.maintainer.CameraTracker") as tracker:
            tracker.side_effect = lambda *a, **k: order.append("start") or MagicMock()
            maintainer._CameraMaintainer__start_camera_processor(CAMERA, config)

        self.assertEqual(order[-1], "start")
        self.assertIn(f"create:{publication_frame_name(CAMERA, 0)}", order)
        # the detector is told how many publication slots it owns
        self.assertEqual(tracker.call_args.args[8], 6)


class DetectorPipelineCase(unittest.TestCase):
    """Drive the real detector loop over a ring the camera keeps reusing."""

    CAPTURE_SLOTS = 2
    PUBLICATION_SLOTS = 2

    def setUp(self):
        self.fixture = publication_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.processor.config
        self.camera_config = self.config.cameras[CAMERA]
        self.camera_config.detect.min_initialized = 0
        self.camera_config.detect.enabled = True
        self.camera_config.detect.occupancy_zones = []
        # the deployed posture: the classifier wants the frame every call
        self.camera_config.detect.stationary.classifier = True

        self.frames = RingFrameManager()
        for i in range(self.CAPTURE_SLOTS):
            self.frames.create_captured_frame(
                capture_frame_name(CAMERA, i), int(np.prod(YUV_SHAPE))
            )
        for i in range(self.PUBLICATION_SLOTS):
            self.frames.create_captured_frame(
                publication_frame_name(CAMERA, i), int(np.prod(YUV_SHAPE))
            )

        self.tracker = NorfairTracker(
            self.camera_config,
            SimpleNamespace(autotracker_enabled=SimpleNamespace(value=False)),
        )
        self.detection = (
            "car",
            0.95,
            (100, 100, 200, 200),
            10000,
            1.0,
            (0, 0, 320, 240),
        )

    def capture(self, index: int, frame_time: float, fill: int) -> str:
        name = capture_frame_name(CAMERA, index)
        self.assertTrue(
            self.frames.write_captured_frame(name, yuv(fill).tobytes(), frame_time)
        )
        return name

    def run_detector(
        self,
        queued,
        *,
        detections=None,
        reuse=None,
        queue_full=False,
        startup_regions=True,
    ):
        """Run process_frames once per queued (name, clock).

        ``reuse`` is called during inference, which is exactly when the camera
        overwrites the capture slot the detector is still working from.
        """
        frame_queue = Queue()
        for item in queued:
            frame_queue.put(item)
        output = MagicMock()
        output.full.return_value = queue_full
        self.published: list[tuple] = []
        output.put.side_effect = self.published.append

        motion = MagicMock()
        motion.detect.return_value = []
        motion.is_calibrating.return_value = True
        stop = MagicMock()
        stop.is_set.return_value = False

        def inference(*args, **kwargs):
            if reuse is not None:
                reuse()
            return list(detections if detections is not None else [self.detection])

        with (
            patch("frigate.video.detect.CameraConfigUpdateSubscriber") as subscriber,
            patch("frigate.video.detect.ptz_moving_at_frame_time", return_value=False),
            patch("frigate.video.detect.get_cluster_candidates", return_value=[]),
            patch(
                "frigate.video.detect.get_startup_regions",
                return_value=[(0, 0, 320, 240)] if startup_regions else [],
            ),
            patch("frigate.video.detect.detect", side_effect=inference),
        ):
            subscriber.return_value.check_for_updates.return_value = []
            process_frames(
                MagicMock(),
                frame_queue,
                FRAME_SHAPE,
                self.config.model,
                self.camera_config,
                self.frames,
                motion,
                MagicMock(),
                self.tracker,
                output,
                MagicMock(),
                stop,
                MagicMock(),
                [],
                self.PUBLICATION_SLOTS,
                exit_on_empty=True,
            )
        return self.published


class TestForcedCaptureReuse(DetectorPipelineCase):
    def test_tracker_advances_when_the_capture_slot_is_reused_mid_detection(self):
        name = self.capture(0, 100.0, 11)
        published = self.run_detector(
            [(name, 100.0)],
            # the camera laps the 2-slot ring while inference runs
            reuse=lambda: (
                self.capture(1, 100.1, 22),
                self.capture(0, 100.2, 33),
            ),
        )
        self.assertIsNone(self.frames.get_captured_frame(name, YUV_SHAPE, 100.0))
        self.assertEqual(len(self.tracker.tracked_objects), 1)
        obj = next(iter(self.tracker.tracked_objects.values()))
        self.assertEqual(obj["frame_time"], 100.0)
        self.assertEqual(obj["detector_observed_at"], 100.0)
        self.assertEqual(len(published), 1)

    def test_published_pixels_are_the_originals_not_the_reused_slot(self):
        name = self.capture(0, 100.0, 11)
        published = self.run_detector(
            [(name, 100.0)], reuse=lambda: self.capture(0, 100.2, 33)
        )
        published_name, published_time = published[0][1], published[0][2]
        self.assertEqual(published_name, publication_frame_name(CAMERA, 0))
        self.assertEqual(published_time, 100.0)
        np.testing.assert_array_equal(
            self.frames.get_captured_frame(published_name, YUV_SHAPE, 100.0), yuv(11)
        )

    def test_empty_frame_still_advances_the_tracker_and_publishes(self):
        name = self.capture(0, 100.0, 11)
        published = self.run_detector(
            [(name, 100.0)],
            detections=[],
            reuse=lambda: self.capture(0, 100.2, 33),
        )
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][3], {})

    def test_stationary_refresh_advances_without_renewing_the_detector_clock(self):
        first = self.capture(0, 100.0, 11)
        self.run_detector([(first, 100.0)])
        obj = next(iter(self.tracker.tracked_objects.values()))
        identity = obj["id"]
        obj["motionless_count"] = 10000

        # a frame with no regions takes the update_frame_times path
        second = self.capture(1, 101.0, 22)
        update_frame_times = self.tracker.update_frame_times

        def reuse_before_refresh(frame, frame_time):
            self.capture(1, 101.5, 44)
            return update_frame_times(frame, frame_time)

        with patch.object(
            self.tracker, "update_frame_times", side_effect=reuse_before_refresh
        ) as refresh:
            published = self.run_detector(
                [(second, 101.0)],
                detections=[],
                startup_regions=False,
            )
        refresh.assert_called_once()
        np.testing.assert_array_equal(refresh.call_args.args[0], yuv(22))
        carried = self.tracker.tracked_objects[identity]
        self.assertEqual(carried["frame_time"], 101.0)
        # the reused box is not a new measurement
        self.assertEqual(carried["detector_observed_at"], 100.0)
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][2], 101.0)

    def test_disabled_stationary_classifier_stays_disabled(self):
        self.camera_config.detect.stationary.classifier = False
        classifier = self.tracker.stationary_classifier
        with (
            patch.object(classifier, "ensure_anchor") as anchor,
            patch.object(classifier, "evaluate", return_value=True) as evaluate,
        ):
            for tick in range(12):
                self.tracker.match_and_update(
                    yuv(11),
                    100.0 + tick,
                    [self.detection],
                    detector_observed_at=[100.0 + tick],
                )
                for obj in self.tracker.tracked_objects.values():
                    obj["motionless_count"] = 10000
            anchor.assert_not_called()
            evaluate.assert_not_called()

    def test_detection_disabled_still_advances_the_tracker(self):
        self.camera_config.detect.enabled = False
        name = self.capture(0, 100.0, 11)
        published = self.run_detector(
            [(name, 100.0)], reuse=lambda: self.capture(0, 100.2, 33)
        )
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][2], 100.0)


class TestPublicationHandoffBounds(DetectorPipelineCase):
    def test_a_full_queue_publishes_no_descriptor_and_writes_no_slot(self):
        name = self.capture(0, 100.0, 11)
        published = self.run_detector([(name, 100.0)], queue_full=True)
        self.assertEqual(published, [])
        self.assertEqual(self.frames.slots[publication_frame_name(CAMERA, 0)][1], 0.0)

    def test_a_refused_publication_write_queues_nothing(self):
        name = self.capture(0, 100.0, 11)
        # a newer generation already occupies the slot this frame would take
        self.assertTrue(
            self.frames.write_captured_frame(
                publication_frame_name(CAMERA, 0), yuv(99).tobytes(), 200.0
            )
        )
        published = self.run_detector([(name, 100.0)])
        self.assertEqual(published, [])
        self.assertEqual(self.frames.rejected_writes, 1)

    def test_no_publication_slots_means_nothing_is_published(self):
        self.PUBLICATION_SLOTS = 0
        name = self.capture(0, 100.0, 11)
        self.assertEqual(self.run_detector([(name, 100.0)]), [])
        # the tracker still ran against the frame it owned
        self.assertEqual(len(self.tracker.tracked_objects), 1)

    def test_the_publication_ring_rotates_and_retains_recent_generations(self):
        self.CAPTURE_SLOTS = 3
        self.frames.create_captured_frame(
            capture_frame_name(CAMERA, 2), int(np.prod(YUV_SHAPE))
        )
        queued = [
            (self.capture(slot, 100.0 + tick, fill), 100.0 + tick)
            for tick, (slot, fill) in enumerate([(0, 11), (1, 22), (2, 33)])
        ]
        published = self.run_detector(queued)
        self.assertEqual(
            [row[1] for row in published],
            [
                publication_frame_name(CAMERA, 0),
                publication_frame_name(CAMERA, 1),
                publication_frame_name(CAMERA, 0),
            ],
        )
        # the ring is bounded: the oldest generation is gone, not preserved
        self.assertIsNone(
            self.frames.get_captured_frame(
                publication_frame_name(CAMERA, 0), YUV_SHAPE, 100.0
            )
        )
        np.testing.assert_array_equal(
            self.frames.get_captured_frame(
                publication_frame_name(CAMERA, 0), YUV_SHAPE, 102.0
            ),
            yuv(33),
        )


class TestCameraStateReadsThePublishedFrame(DetectorPipelineCase):
    def setUp(self):
        super().setUp()
        self.state = self.fixture.state
        self.state.frame_manager = self.frames

    def publish_one(self, frame_time: float, slot: int, fill: int) -> tuple[str, float]:
        name = self.capture(slot, frame_time, fill)
        published = self.run_detector(
            [(name, frame_time)], reuse=lambda: self.capture(slot, frame_time + 0.2, 77)
        )
        self.assertEqual(len(published), 1)
        return published[0][1], published[0][2]

    def detection_payload(self, frame_time: float) -> dict:
        return {
            publication_fixture.EVENT_ID: {
                "id": publication_fixture.EVENT_ID,
                "label": "car",
                "frame_time": frame_time,
                "detector_observed_at": frame_time,
                "start_time": frame_time,
                "score": 0.9,
                "score_history": [0.9] * 3,
                "box": (100, 100, 200, 200),
                "centroid": (150, 150),
                "area": 10000,
                "ratio": 1.0,
                "region": (0, 0, 320, 240),
                "motionless_count": 100,
                "position_changes": 1,
                "attributes": [],
            }
        }

    def test_calibration_frame_is_the_exact_published_pixels_and_clock(self):
        name, clock = self.publish_one(100.0, 0, 11)
        self.state.update(name, clock, self.detection_payload(clock), [], [], None)
        frame, frame_time, _ = self.state.get_calibration_frame()
        self.assertEqual(frame_time, 100.0)
        # BGR of a flat Y=11 frame; assert the state kept the published frame
        np.testing.assert_array_equal(self.state._current_frame, yuv(11))
        self.assertEqual(frame.shape, (240, 320, 3))

    def test_a_reused_publication_slot_leaves_the_previous_frame_and_clock(self):
        name, clock = self.publish_one(100.0, 0, 11)
        self.state.update(name, clock, self.detection_payload(clock), [], [], None)

        # a later generation takes that slot before this consumer arrives
        self.assertTrue(
            self.frames.write_captured_frame(name, yuv(88).tobytes(), 105.0)
        )
        self.state.update(name, clock, self.detection_payload(clock), [], [], None)

        self.assertEqual(self.state.current_frame_time, 100.0)
        np.testing.assert_array_equal(self.state._current_frame, yuv(11))
        self.assertNotIn(88, np.unique(self.state._current_frame))

    def test_event_callbacks_carry_the_clock_of_the_frame_they_name(self):
        seen: list[tuple] = []
        self.state.on("start", lambda *args: seen.append(args))
        name, clock = self.publish_one(100.0, 0, 11)
        self.state.update(name, clock, self.detection_payload(clock), [], [], None)
        self.assertEqual(len(seen), 1)
        camera, _obj, frame_name, frame_time = seen[0]
        self.assertEqual(camera, CAMERA)
        self.assertEqual(frame_name, name)
        self.assertEqual(frame_time, 100.0)
        np.testing.assert_array_equal(
            self.frames.get_captured_frame(frame_name, YUV_SHAPE, frame_time), yuv(11)
        )


class TestDownstreamConsumersUseTheirOwnClock(unittest.TestCase):
    """Every reader names the clock that belongs to its own message."""

    def setUp(self):
        self.frames = RingFrameManager()
        self.name = publication_frame_name(CAMERA, 0)
        self.frames.create_captured_frame(self.name, int(np.prod(YUV_SHAPE)))
        self.assertTrue(
            self.frames.write_captured_frame(self.name, yuv(11).tobytes(), 100.0)
        )
        self.camera_config = SimpleNamespace(
            frame_shape_yuv=YUV_SHAPE,
            enabled=True,
            detect=SimpleNamespace(fps=15),
        )

    def maintainer(self):
        from frigate.embeddings.maintainer import EmbeddingMaintainer

        owner = EmbeddingMaintainer.__new__(EmbeddingMaintainer)
        owner.config = SimpleNamespace(
            cameras={CAMERA: self.camera_config},
            semantic_search=SimpleNamespace(enabled=False),
        )
        owner.frame_manager = self.frames
        owner.realtime_processors = [MagicMock()]
        owner.post_processors = []
        owner.event_subscriber = MagicMock()
        return owner

    def event(self, frame_time: float):
        from frigate.events.types import EventStateEnum, EventTypeEnum

        return (
            EventTypeEnum.tracked_object,
            EventStateEnum.update,
            CAMERA,
            self.name,
            frame_time,
            {"id": "1", "label": "car"},
        )

    def test_event_update_reads_the_frame_its_message_names(self):
        owner = self.maintainer()
        owner.event_subscriber.check_for_update.return_value = self.event(100.0)
        owner._process_updates()
        processor = owner.realtime_processors[0]
        processor.process_frame.assert_called_once()
        np.testing.assert_array_equal(
            processor.process_frame.call_args.args[1], yuv(11)
        )

    def test_event_update_refuses_a_slot_that_moved_to_a_newer_generation(self):
        owner = self.maintainer()
        owner.event_subscriber.check_for_update.return_value = self.event(100.0)
        self.assertTrue(
            self.frames.write_captured_frame(self.name, yuv(88).tobytes(), 101.0)
        )
        owner._process_updates()
        owner.realtime_processors[0].process_frame.assert_not_called()

    def test_a_manual_event_names_no_frame_and_reads_none(self):
        owner = self.maintainer()
        owner.event_subscriber.check_for_update.return_value = (
            self.event(100.0)[0],
            self.event(100.0)[1],
            CAMERA,
            "",
            0.0,
            {"id": "1", "label": "car"},
        )
        owner._process_updates()
        owner.realtime_processors[0].process_frame.assert_not_called()


class TestOcrUsesTheExactPublishedFrame(unittest.TestCase):
    """OCR eligibility and the crop must agree on one generation."""

    def setUp(self):
        from frigate.test import test_lpr_current_frames as lpr_fixture

        self.lpr_fixture = lpr_fixture
        self.case = lpr_fixture.TestCurrentLprScheduling("run")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.owner = self.case.owner
        self.frames = RingFrameManager()
        self.owner.frame_manager = self.frames
        self.name = "a_published0"
        self.frames.create_captured_frame(self.name, 48 * 32)
        self.original = np.full((48, 32), 11, dtype=np.uint8)
        self.assertTrue(
            self.frames.write_captured_frame(self.name, self.original.tobytes(), 1000.0)
        )

    def packet(self, frame_time=1000.0):
        camera, _old_name, clock, objects, motion, regions = self.lpr_fixture.packet(
            frame_time=frame_time
        )
        return (camera, self.name, clock, objects, motion, regions)

    def test_ocr_crops_the_exact_published_generation(self):
        self.case.run_packet(self.packet())
        processor = self.case.processor
        processor.process_frame.assert_called_once()
        self.assertEqual(
            processor.process_frame.call_args.kwargs, {"source_frame_time": 1000.0}
        )
        np.testing.assert_array_equal(
            processor.process_frame.call_args.args[1], self.original
        )

    def test_ocr_refuses_a_publication_slot_that_already_rolled_over(self):
        self.case.queue.append(self.packet())
        # the detector publishes a newer generation into the same slot before
        # OCR reaches this packet
        self.assertTrue(
            self.frames.write_captured_frame(
                self.name, np.full((48, 32), 99, dtype=np.uint8).tobytes(), 1001.0
            )
        )
        self.owner._process_frame_updates()
        self.owner._process_current_lpr()
        self.case.processor.process_frame.assert_not_called()


if __name__ == "__main__":
    unittest.main()
