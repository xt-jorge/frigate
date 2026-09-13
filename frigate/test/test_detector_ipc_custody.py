"""Detector results must retain request identity across timeout and buffer reuse."""

import fcntl
import multiprocessing as mp
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import zmq

from frigate.comms.object_detector_signaler import (
    ObjectDetectorPublisher,
    ObjectDetectorSubscriber,
)
from frigate.object_detection.base import (
    AsyncDetectorRunner,
    RemoteObjectDetector,
    read_detector_input,
)
from frigate.util.builtin import EventsPerSecond
from frigate.util.image import UntrackedSharedMemory


def delayed_worker(endpoint, requests, copied):
    context = zmq.Context()
    publisher = ObjectDetectorPublisher.__new__(ObjectDetectorPublisher)
    publisher.topic = publisher.topic_base
    publisher.socket = context.socket(zmq.PAIR)
    publisher.socket.bind(endpoint)
    camera, first = requests.get(timeout=3)
    original = read_detector_input(camera, first, (1, 2, 2, 3))
    copied.set()
    camera, second = requests.get(timeout=3)
    current = read_detector_input(camera, second, (1, 2, 2, 3))
    for request_id, tensor in ((first, original), (second, current)):
        result = np.zeros((20, 6), dtype=np.float32)
        result[0] = [0, 0.9, int(tensor[0, 0, 0, 0]), 0, 1, 1]
        publisher.publish(camera, request_id, result)
    publisher.socket.close()
    context.term()


@unittest.skipUnless(
    sys.platform == "linux", "Production POSIX SHM flock requires Linux"
)
class TestDetectorIpcCustody(unittest.TestCase):
    def setUp(self):
        self.name = "dt-" + uuid4().hex[:20]
        self.shm = UntrackedSharedMemory(name=self.name, create=True, size=16 + 12)
        self.addCleanup(self.shm.unlink)
        self.addCleanup(self.shm.close)

    def remote(self, subscriber, requests):
        detector = RemoteObjectDetector.__new__(RemoteObjectDetector)
        detector.name = self.name
        detector.labels = {0: "car"}
        detector.input_shape = (1, 2, 2, 3)
        detector.stop_event = threading.Event()
        detector.detection_queue = requests
        detector.detector_subscriber = subscriber
        detector.fps = EventsPerSecond()
        return detector

    def test_old_queued_generation_cannot_read_replaced_input(self):
        first, second = uuid4().hex, uuid4().hex
        self.shm.buf[:16] = bytes.fromhex(first)
        self.shm.buf[16:28] = bytes([1] * 12)
        original = read_detector_input(self.name, first, (1, 2, 2, 3))
        self.shm.buf[:16] = bytes.fromhex(second)
        self.shm.buf[16:28] = bytes([2] * 12)
        self.assertIsNone(read_detector_input(self.name, first, (1, 2, 2, 3)))
        self.assertTrue((original == 1).all())
        self.assertTrue(
            (read_detector_input(self.name, second, (1, 2, 2, 3)) == 2).all()
        )

    def test_real_process_late_result_cannot_satisfy_next_request(self):
        context = mp.get_context("fork")
        requests, copied = context.Queue(), context.Event()
        with tempfile.TemporaryDirectory() as directory:
            endpoint = "ipc://" + str(Path(directory) / "result")
            worker = context.Process(
                target=delayed_worker, args=(endpoint, requests, copied)
            )
            worker.start()
            zcontext = zmq.Context()
            subscriber = ObjectDetectorSubscriber.__new__(ObjectDetectorSubscriber)
            subscriber.topic = f"object_detector/{self.name}/"
            subscriber.socket = zcontext.socket(zmq.PAIR)
            subscriber.socket.connect(endpoint)
            detector = self.remote(subscriber, requests)
            try:
                # A's caller times out only after the worker owns its input copy.
                def timeout(**kwargs):
                    self.assertTrue(copied.wait(3))
                    return None

                with patch.object(subscriber, "check_for_update", side_effect=timeout):
                    self.assertEqual(
                        detector.detect(np.ones((1, 2, 2, 3), np.uint8)), []
                    )
                result = detector.detect(np.full((1, 2, 2, 3), 2, np.uint8))
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0][2][0], 2)
                worker.join(3)
                self.assertEqual(worker.exitcode, 0)
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                subscriber.socket.close()
                zcontext.term()
                requests.close()

    def test_matching_missing_result_returns_no_detection(self):
        requests = MagicMock()
        subscriber = MagicMock()
        requests.put.side_effect = lambda envelope: setattr(
            subscriber.check_for_update, "return_value", (envelope[1], None)
        )
        self.assertEqual(
            self.remote(subscriber, requests).detect(np.zeros((1, 2, 2, 3), np.uint8)),
            [],
        )

    def test_async_none_publishes_failure_for_exact_request(self):
        runner = AsyncDetectorRunner.__new__(AsyncDetectorRunner)
        runner.stop_event = MagicMock()
        runner.stop_event.is_set.side_effect = [False, True]
        runner._detector = MagicMock()
        runner._detector.async_receive_output.return_value = ("request", None)
        runner.pending = {self.name: ("request", 0.0)}
        runner.pending_lock = threading.Lock()
        runner._publisher = MagicMock()
        runner.avg_speed = SimpleNamespace(value=0.0)
        runner.start_time = SimpleNamespace(value=0.0)
        runner._result_worker()
        runner._publisher.publish.assert_called_once_with(self.name, "request", None)
        self.assertEqual(runner.pending, {})

    def test_publisher_missing_output_sends_failure_without_old_payload(self):
        publisher = ObjectDetectorPublisher.__new__(ObjectDetectorPublisher)
        publisher.topic = publisher.topic_base
        publisher.socket = MagicMock()
        output = np.ones((20, 6), np.float32)
        publisher.publish(self.name, "first", output)
        publisher.publish(self.name, "second", None)
        self.assertEqual(
            publisher.socket.send_multipart.call_args.args[0][1:], [b"second", b""]
        )

    def test_async_reordered_old_result_does_not_remove_current_request(self):
        runner = AsyncDetectorRunner.__new__(AsyncDetectorRunner)
        runner.stop_event = MagicMock()
        runner.stop_event.is_set.side_effect = [False, False, True]
        runner._detector = MagicMock()
        runner._detector.async_receive_output.side_effect = [
            ("old", np.ones((20, 6))),
            ("current", None),
        ]
        runner.pending = {self.name: ("current", 0.0)}
        runner.pending_lock = threading.Lock()
        runner._publisher = MagicMock()
        runner.avg_speed = SimpleNamespace(value=0.0)
        runner.start_time = SimpleNamespace(value=0.0)
        runner._result_worker()
        runner._publisher.publish.assert_called_once_with(self.name, "current", None)

    def test_locked_input_is_refused_without_enqueuing(self):
        requests, subscriber = MagicMock(), MagicMock()
        fcntl.flock(self.shm._fd, fcntl.LOCK_EX)
        try:
            self.assertIsNone(read_detector_input(self.name, uuid4().hex, (1, 2, 2, 3)))
            self.assertEqual(
                self.remote(subscriber, requests).detect(
                    np.zeros((1, 2, 2, 3), np.uint8)
                ),
                [],
            )
            requests.put.assert_not_called()
        finally:
            fcntl.flock(self.shm._fd, fcntl.LOCK_UN)

    def test_unrelated_results_do_not_extend_original_deadline(self):
        subscriber, requests = MagicMock(), MagicMock()
        subscriber.check_for_update.return_value = ("old", np.ones((20, 6)))
        with patch(
            "frigate.object_detection.base.time.monotonic", side_effect=[0, 1, 4, 5]
        ):
            self.assertEqual(
                self.remote(subscriber, requests).detect(
                    np.zeros((1, 2, 2, 3), np.uint8)
                ),
                [],
            )
        self.assertEqual(
            [
                call.kwargs["timeout"]
                for call in subscriber.check_for_update.call_args_list
            ],
            [4, 1],
        )
