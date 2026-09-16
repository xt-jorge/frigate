"""Occupancy snapshots retain the real Norfair lifetime and measurement clocks."""

import unittest

import numpy as np
from norfair import Detection, Tracker

from frigate.track.norfair_tracker import NorfairTracker


class TestOccupancyTrackerIdentity(unittest.TestCase):
    def setUp(self):
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=10,
            initialization_delay=2,
            hit_counter_max=5,
        )
        self.owner = NorfairTracker.__new__(NorfairTracker)
        self.owner.occupancy_generation = "test-generation"
        self.owner.trackers = {"car": {"static": self.tracker}}
        self.owner.default_tracker = {}

    def detect(self, at):
        self.tracker.update(
            [
                Detection(
                    points=np.array([[30, 40], [90, 100]]),
                    data={
                        "label": "car",
                        "box": [30, 40, 90, 100],
                        "frame_time": at,
                        "detector_observed_at": at,
                    },
                )
            ]
        )
        return self.owner.occupancy_tracks()[0]

    def test_initialization_and_coasting_preserve_identity_and_last_measurement(self):
        first = self.detect(100)
        self.assertFalse(first["initialized"])
        self.detect(100.25)
        confirmed = self.detect(100.5)
        self.assertTrue(confirmed["initialized"])
        self.assertEqual(confirmed["id"], first["id"])
        self.tracker.update([])
        coast = self.owner.occupancy_tracks()[0]
        self.assertEqual(coast, confirmed)
        self.assertEqual(coast["detector_observed_at"], 100.5)

    def test_deleted_track_id_is_not_reused_in_the_same_tracker(self):
        first = self.detect(100)
        for _ in range(10):
            self.tracker.update([])
        self.assertEqual(self.owner.occupancy_tracks(), [])
        second = self.detect(110)
        self.assertNotEqual(second["id"], first["id"])
        self.assertFalse(second["initialized"])
