"""An empty detector result is useful only when inference actually completed."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.object_detection.base import RemoteObjectDetector


class TestDetectorCompletion(unittest.TestCase):
    def test_completion_distinguishes_empty_timeout_refusal_and_stop(self):
        for result_kind in ["empty", "timeout", "refused", "stopped", "locked"]:
            with self.subTest(result_kind=result_kind):
                detector = RemoteObjectDetector.__new__(RemoteObjectDetector)
                detector.name = "test"
                detector.labels = {0: "car"}
                detector.input_shape = (1, 2, 2, 3)
                detector.stop_event = threading.Event()
                detector.detection_queue = MagicMock()
                detector.detector_subscriber = MagicMock()
                detector.fps = MagicMock()
                detector.last_detection_successful = True
                if result_kind == "stopped":
                    detector.stop_event.set()

                def respond(envelope, result_kind=result_kind, detector=detector):
                    raw = np.zeros((20, 6)) if result_kind == "empty" else None
                    detector.detector_subscriber.check_for_update.return_value = (
                        None if result_kind == "timeout" else (envelope[1], raw)
                    )

                detector.detection_queue.put.side_effect = respond
                shm = SimpleNamespace(
                    _fd=1, size=28, buf=bytearray(28), close=lambda: None
                )
                with (
                    patch(
                        "frigate.object_detection.base.UntrackedSharedMemory",
                        return_value=shm,
                    ),
                    patch(
                        "frigate.object_detection.base.fcntl.flock",
                        side_effect=BlockingIOError
                        if result_kind == "locked"
                        else None,
                    ),
                ):
                    self.assertEqual(
                        detector.detect(np.zeros((1, 2, 2, 3), np.uint8)), []
                    )
                self.assertEqual(
                    detector.last_detection_successful, result_kind == "empty"
                )
