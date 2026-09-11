"""Capture publishes only complete frames accepted with their original clock."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from frigate.util.image import FrameManager
from frigate.video.ffmpeg import capture_frames


class TestCaptureFramesPublication(unittest.TestCase):
    def setUp(self):
        self.ffmpeg = Mock()
        self.ffmpeg.poll.return_value = None
        self.frame_manager = Mock(spec=FrameManager)
        self.frame_manager.write_captured_frame.return_value = True
        self.frame_queue = Mock()
        self.stop_event = threading.Event()
        self.operations = Mock()
        self.operations.attach_mock(self.ffmpeg.stdout.read, "read")
        self.operations.attach_mock(self.frame_manager.write_captured_frame, "write")

    def capture(self, packets, clocks, stop_after_last=True):
        packet_index = 0

        def read(size):
            nonlocal packet_index
            self.assertEqual(size, 6)
            packet = packets[packet_index]
            packet_index += 1
            if stop_after_last and packet_index == len(packets):
                self.stop_event.set()
            return packet

        self.ffmpeg.stdout.read.side_effect = read
        with (
            patch("frigate.video.ffmpeg.datetime") as clock,
            patch("frigate.video.ffmpeg.CameraConfigUpdateSubscriber") as subscriber,
            patch("frigate.video.ffmpeg.logger"),
        ):
            # Exhaustion fails an unexpected extra loop instead of spinning on EOF.
            clock.now.return_value.timestamp.side_effect = clocks
            capture_frames(
                self.ffmpeg,
                SimpleNamespace(name="front", enabled=True),
                2,
                0,
                (2, 3),
                self.frame_manager,
                self.frame_queue,
                SimpleNamespace(value=0.0),
                SimpleNamespace(value=0.0),
                SimpleNamespace(value=0.0),
                self.stop_event,
            )
            subscriber.return_value.stop.assert_called_once_with()

    def test_eof_from_dead_ffmpeg_exits_without_writing_or_enqueuing(self):
        self.ffmpeg.poll.return_value = 1
        self.capture([b""], [100.25], stop_after_last=False)
        self.ffmpeg.stdout.read.assert_called_once_with(6)
        self.ffmpeg.poll.assert_called_once_with()
        self.frame_manager.write_captured_frame.assert_not_called()
        self.frame_queue.put.assert_not_called()
        self.frame_manager.close.assert_not_called()

    def test_incomplete_read_retries_live_ffmpeg_then_publishes_original_clock(self):
        self.capture([b"short", b"second"], [100.25, 101.5])
        self.ffmpeg.poll.assert_called_once_with()
        self.assertEqual(
            self.operations.mock_calls,
            [
                call.read(6),
                call.read(6),
                call.write("front_frame0", b"second", 101.5),
            ],
        )
        self.frame_queue.put.assert_called_once_with(("front_frame0", 101.5), False)
        self.frame_manager.close.assert_called_once_with("front_frame0")

    def test_write_contention_drops_packet_and_drains_next_before_publication(self):
        self.frame_manager.write_captured_frame.side_effect = [False, True]
        # A refused write must not consume the source-clock watermark.
        self.capture([b"first!", b"second"], [100.25, 100.25])
        self.assertEqual(
            self.operations.mock_calls,
            [
                call.read(6),
                call.write("front_frame0", b"first!", 100.25),
                call.read(6),
                call.write("front_frame0", b"second", 100.25),
            ],
        )
        self.frame_queue.put.assert_called_once_with(("front_frame0", 100.25), False)
        self.frame_manager.close.assert_called_once_with("front_frame0")

    def test_duplicate_and_regressing_clocks_never_write_or_enqueue(self):
        self.capture(
            [b"first!", b"dupe!!", b"older!", b"newest"],
            [100.25, 100.25, 99.5, 101.75],
        )
        self.assertEqual(self.ffmpeg.stdout.read.call_count, 4)
        self.assertEqual(
            self.frame_manager.write_captured_frame.call_args_list,
            [
                call("front_frame0", b"first!", 100.25),
                call("front_frame1", b"newest", 101.75),
            ],
        )
        self.assertEqual(
            self.frame_queue.put.call_args_list,
            [
                call(("front_frame0", 100.25), False),
                call(("front_frame1", 101.75), False),
            ],
        )
