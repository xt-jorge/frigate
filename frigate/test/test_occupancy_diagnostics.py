"""Local occupancy diagnostics explain holds without changing clearance evidence."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from frigate.video.occupancy import OccupancyContinuity


class TestOccupancyDiagnostics(unittest.TestCase):
    def setUp(self):
        self.clock = 1.0
        self.clock_patch = patch("time.monotonic", side_effect=lambda: self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.log_patch = patch("frigate.video.occupancy.logger.info")
        self.log = self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.owner = OccupancyContinuity(5)
        self.zones = {
            "zone": np.array([[0, 20], [120, 20], [120, 110], [0, 110]], dtype=np.int32)
        }
        self.frame = np.random.default_rng(0).integers(
            30, 210, (180, 120), dtype=np.uint8
        )
        self.track = {
            "id": "private-native-track-id",
            "initialized": True,
            "box": [30, 40, 90, 100],
            "frame_time": 100,
            "detector_observed_at": 100,
        }
        self.candidate = ("car", 0.12, (5, 45, 25, 95), 1000, 0.4, (0, 20, 120, 110))

    def observe(self, at, tracks=(), raw=(), complete=True):
        result = self.owner.observe(
            self.frame, at, self.zones, list(raw), list(tracks), complete
        )
        self.assertEqual(set(result[0]), {"zone", "box", "uncertain"})
        return result[0]["uncertain"]

    def diagnostic(self):
        return json.loads(self.log.call_args.args[1])

    def test_distinguishes_track_pixel_and_untracked_candidate_holds(self):
        self.assertTrue(self.observe(100, [self.track], [self.candidate]))
        row = self.diagnostic()
        self.assertEqual(
            row["reason_frames"], {"pending_candidates": 1, "current_track": 1}
        )
        self.assertEqual(row["pending_candidates"], 1)
        self.assertTrue(row["pending_untracked"])
        self.assertEqual(row["raw_overlaps"], 1)
        self.assertEqual(row["candidate_observation_frames"], 1)
        self.assertEqual(row["untracked_observation_frames"], 1)
        self.assertEqual(row["grace_rearm_frames"], 0)
        self.assertEqual(row["track_footprints"], 1)
        self.assertEqual(row["pixel_footprints"], 0)
        self.assertEqual(row["oldest_footprint_anchor_age_s"], 0)
        self.clock += 10
        self.assertTrue(self.observe(100.5))
        row = self.diagnostic()
        self.assertEqual(
            row["reason_frames"], {"pending_candidates": 1, "retained_pixels": 1}
        )
        self.assertEqual(row["track_overlaps"], 0)
        self.assertEqual(row["track_footprints"], 0)
        self.assertEqual(row["pixel_footprints"], 1)
        self.assertEqual(row["oldest_pending_quiet_age_s"], 0.5)
        self.assertEqual(row["oldest_footprint_anchor_age_s"], 0.5)
        self.assertEqual(row["candidate_observation_frames"], 0)
        self.assertEqual(row["untracked_observation_frames"], 0)
        self.assertEqual(row["grace_rearm_frames"], 0)
        self.assertNotIn(self.track["id"], str(self.log.call_args_list))
        self.assertNotIn("box", row)

    def test_rate_limit_uses_monotonic_clock_and_preserves_transient_reasons(self):
        self.observe(100)
        for tick in range(1, 41):
            self.clock = 1 + tick / 4
            at = 100 + tick / 4
            self.observe(at, raw=[self.candidate] if tick == 1 else [])
        self.assertEqual(self.log.call_count, 2)
        row = self.diagnostic()
        self.assertEqual(row["frames"], 40)
        self.assertEqual(row["window_s"], 10)
        self.assertEqual(
            row["reason_frames"], {"pending_candidates": 20, "no_continuity_hold": 20}
        )
        self.assertEqual(row["pending_candidates"], 0)
        self.assertEqual(row["candidate_observation_frames"], 1)
        self.assertEqual(row["untracked_observation_frames"], 1)
        self.assertEqual(row["oldest_pending_quiet_age_s"], None)
        # A jumped or repeated capture clock cannot bypass the logging limit.
        for at in [10000, 20000, 20000, 19000]:
            self.observe(at)
        self.assertEqual(self.log.call_count, 2)

    def test_coverage_gaps_rearm_grace_without_claiming_a_new_candidate(self):
        self.observe(100, raw=[self.candidate])
        self.clock += 10
        self.assertTrue(self.observe(102))
        row = self.diagnostic()
        self.assertEqual(row["grace_rearm_frames"], 1)
        self.assertEqual(row["candidate_observation_frames"], 0)
        self.assertEqual(row["oldest_pending_quiet_age_s"], 0)
        self.observe(102.25, complete=False)
        self.clock += 10
        self.observe(102.5)
        row = self.diagnostic()
        self.assertEqual(row["grace_rearm_frames"], 2)
        self.assertEqual(row["untracked_observation_frames"], 0)
        self.assertEqual(
            row["reason_frames"], {"incomplete_coverage": 1, "pending_candidates": 1}
        )

    def test_incomplete_and_non_monotonic_frames_are_distinct_without_state_changes(
        self,
    ):
        self.observe(100, [self.track])
        footprints = dict(self.owner.footprints)
        self.clock += 10
        self.assertTrue(self.observe(99))
        row = self.diagnostic()
        self.assertEqual(row["reason_frames"], {"non_monotonic_frame": 1})
        self.assertIsNone(row["track_overlaps"])
        self.assertEqual(self.owner.footprints, footprints)
        self.assertEqual(self.owner.last_frame, 100)
        self.clock += 10
        self.assertTrue(self.observe(101, complete=False))
        self.assertEqual(self.diagnostic()["reason_frames"], {"incomplete_coverage": 1})
        self.assertEqual(self.owner.footprints, footprints)

    def test_coasting_candidates_and_persistent_saturation_remain_observable(self):
        candidate = {**self.track, "initialized": False}
        raw = ("car", 0.6, tuple(candidate["box"]), 3600, 1, (0, 20, 120, 110))
        self.observe(100, [candidate], [raw])
        self.owner.saturated.add("zone")
        self.clock += 10
        self.observe(100.5, [candidate])
        row = self.diagnostic()
        self.assertEqual(
            row["reason_frames"], {"pending_candidates": 1, "saturated": 1}
        )
        self.assertEqual(row["uninitialized_overlaps"], 1)
        self.assertEqual(row["stale_track_overlaps"], 1)
        self.assertEqual(row["oldest_track_observation_age_s"], 0.5)
        self.assertTrue(row["saturated"])
        self.clock += 10
        self.assertTrue(self.observe(101))
        self.assertTrue(self.diagnostic()["saturated"])
