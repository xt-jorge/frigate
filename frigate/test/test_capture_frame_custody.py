"""Linux capture-ring locking must bind queued clocks to owned pixel copies."""

import fcntl
import multiprocessing
import sys
import threading
import unittest
import uuid

import numpy as np

from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory


def hold_slot(name, exclusive, acquired, release):
    shm = UntrackedSharedMemory(name=name)
    try:
        fcntl.flock(shm._fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        acquired.set()
        if not release.wait(5):
            raise RuntimeError("Test did not release capture slot")
    finally:
        shm.close()


@unittest.skipUnless(
    sys.platform == "linux", "Production POSIX SHM flock requires Linux"
)
class TestCaptureFrameCustody(unittest.TestCase):
    def setUp(self):
        self.name = f"frigate-test-{uuid.uuid4().hex}"
        self.writer = SharedMemoryFrameManager()
        self.reader = SharedMemoryFrameManager()
        self.writer.create_captured_frame(self.name, 16)
        self.addCleanup(self.writer.cleanup)
        self.addCleanup(self.reader.cleanup)

    def test_overwritten_slot_refuses_old_packet_and_keeps_returned_copy_frozen(self):
        self.assertTrue(
            self.writer.write_captured_frame(self.name, bytes([11] * 16), 100)
        )
        frame = self.reader.get_captured_frame(self.name, (4, 4), 100)
        self.assertTrue(
            self.writer.write_captured_frame(self.name, bytes([22] * 16), 101)
        )
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 100))
        np.testing.assert_array_equal(frame, np.full((4, 4), 11))
        np.testing.assert_array_equal(
            self.reader.get_captured_frame(self.name, (4, 4), 101), np.full((4, 4), 22)
        )
        self.assertEqual(self.reader.get(self.name, (4, 4)).shape, (4, 4))

    def test_duplicate_regressed_invalid_or_partial_write_does_not_retime_pixels(self):
        self.writer.write_captured_frame(self.name, bytes([11] * 16), 100)
        for clock in (100, 99, 0, True, float("nan"), float("inf")):
            self.assertFalse(
                self.writer.write_captured_frame(self.name, bytes([22] * 16), clock)
            )
        self.assertFalse(self.writer.write_captured_frame(self.name, bytes(15), 101))
        np.testing.assert_array_equal(
            self.reader.get_captured_frame(self.name, (4, 4), 100), np.full((4, 4), 11)
        )
        # Simulate a failure at the actual pixel-copy seam after invalidation.
        with self.assertRaises(TypeError):
            self.writer.write_captured_frame(self.name, [1] * 16, 101)
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 100))
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 101))

    def test_missing_legacy_and_wrong_size_slots_are_unknown(self):
        self.assertIsNone(self.reader.get_captured_frame(self.name, (2, 4), 100))
        self.writer.delete(self.name)
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 100))
        self.writer.create(self.name, 16)
        with self.assertRaises(ValueError):
            self.writer.create_captured_frame(self.name, 16)
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 100))

    def test_recreated_name_cannot_validate_a_cached_mapping_with_new_clock(self):
        self.writer.write_captured_frame(self.name, bytes([11] * 16), 100)
        cached = self.reader.get(self.name, (4, 4))
        self.assertEqual(int(cached[0, 0]), 11)
        self.writer.delete(self.name)
        self.writer.create_captured_frame(self.name, 16)
        self.writer.write_captured_frame(self.name, bytes([22] * 16), 101)
        self.assertIsNone(self.reader.get_captured_frame(self.name, (4, 4), 100))
        np.testing.assert_array_equal(
            self.reader.get_captured_frame(self.name, (4, 4), 101), np.full((4, 4), 22)
        )
        # The strict helper did not use the reader's still-old raw mapping.
        self.assertEqual(int(cached[0, 0]), 11)
        del cached

    def _contention(self, thread, exclusive):
        self.writer.write_captured_frame(self.name, bytes([11] * 16), 100)
        if thread:
            acquired, release = threading.Event(), threading.Event()
            worker = threading.Thread(
                target=hold_slot, args=(self.name, exclusive, acquired, release)
            )
        else:
            context = multiprocessing.get_context("spawn")
            acquired, release = context.Event(), context.Event()
            worker = context.Process(
                target=hold_slot, args=(self.name, exclusive, acquired, release)
            )
        worker.start()
        try:
            self.assertTrue(acquired.wait(5))
            self.assertFalse(
                self.writer.write_captured_frame(self.name, bytes([22] * 16), 101)
            )
            frame = self.reader.get_captured_frame(self.name, (4, 4), 100)
            if exclusive:
                self.assertIsNone(frame)
            else:
                np.testing.assert_array_equal(frame, np.full((4, 4), 11))
        finally:
            release.set()
            worker.join(5)
            self.assertFalse(worker.is_alive())
        if not thread:
            self.assertEqual(worker.exitcode, 0)
        self.assertTrue(
            self.writer.write_captured_frame(self.name, bytes([22] * 16), 101)
        )

    def test_independent_process_writer_lock_drops_reader_and_writer_without_wait(self):
        self._contention(False, True)

    def test_independent_process_reader_lock_drops_writer_without_wait(self):
        self._contention(False, False)

    def test_same_manager_threads_still_use_independent_open_descriptions(self):
        self._contention(True, True)


if __name__ == "__main__":
    unittest.main()
