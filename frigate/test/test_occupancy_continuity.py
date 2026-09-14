"""Current image evidence resolves detector gaps without a full passage trajectory."""

import unittest

import numpy as np

from frigate.video.occupancy import OccupancyContinuity


class TestOccupancyContinuity(unittest.TestCase):
    def setUp(self):
        self.owner = OccupancyContinuity(5)
        self.region = (0, 20, 120, 110)
        self.zones = {
            "zone": np.array([[0, 20], [120, 20], [120, 110], [0, 110]], dtype=np.int32)
        }
        self.box = (30, 40, 90, 100)
        self.frame = np.random.default_rng(0).integers(
            30, 210, (180, 120), dtype=np.uint8
        )
        self.track = {
            "id": "native-generation:car:1",
            "label": "car",
            "initialized": True,
            "box": list(self.box),
            "frame_time": 100,
            "detector_observed_at": 100,
        }

    def observe(self, at, tracks=(), raw=(), frame=None, complete=True):
        return self.owner.observe(
            self.frame if frame is None else frame,
            at,
            self.zones,
            list(raw),
            list(tracks),
            complete,
        )[0]["uncertain"]

    def test_stationary_loss_keeps_footprint_through_long_negative_inference(self):
        self.assertTrue(self.observe(100, [self.track]))
        for tick in range(1, 2401):
            self.assertTrue(self.observe(100 + tick / 4))

    def test_background_motion_and_brightness_shift_do_not_discharge_unchanged_car(
        self,
    ):
        self.observe(100, [self.track])
        changed = self.frame.copy()
        changed[:35, :] = 255
        changed[:, :25] = 0
        changed[:, 100:] = 250
        changed[110:120, :] = 0
        for tick in range(1, 30):
            self.assertTrue(self.observe(100 + tick / 4, frame=changed))
        bright = np.minimum(self.frame.astype(np.uint16) + 15, 255).astype(np.uint8)
        for tick in range(30, 60):
            self.assertTrue(self.observe(100 + tick / 4, frame=bright))

    def test_departure_can_recover_without_native_id_outside_the_polygon(self):
        self.observe(100, [self.track])
        empty = self.frame.copy()
        empty[40:100, 30:90] = np.random.default_rng(8).integers(
            30, 210, (60, 60), dtype=np.uint8
        )
        values = [self.observe(100 + tick / 4, frame=empty) for tick in range(1, 12)]
        self.assertIn(False, values)
        self.assertFalse(values[-1])

    def test_weak_candidate_covered_by_confirmed_track_adds_no_departure_grace(self):
        weak = ("car", 0.2, (35, 45, 85, 95), 2500, 1, self.region)
        self.observe(100, [self.track], [weak])
        self.assertEqual(self.owner.raw_pending["zone"], {})
        empty = self.frame.copy()
        empty[40:100, 30:90] = np.random.default_rng(8).integers(
            30, 210, (60, 60), dtype=np.uint8
        )
        values = [self.observe(100 + tick / 4, frame=empty) for tick in range(1, 5)]
        self.assertIn(False, values)
        self.assertEqual(self.owner.raw_pending["zone"], {})

    def test_separate_weak_follower_keeps_its_grace_when_confirmed_car_departs(self):
        follower = ("car", 0.2, (5, 45, 25, 95), 1000, 0.4, self.region)
        self.observe(100, [self.track], [follower])
        empty = self.frame.copy()
        empty[40:100, 30:90] = np.random.default_rng(8).integers(
            30, 210, (60, 60), dtype=np.uint8
        )
        for tick in range(1, 20):
            self.assertTrue(self.observe(100 + tick / 4, frame=empty))
        self.assertFalse(self.observe(105, frame=empty))

    def test_new_generation_or_a_second_departing_car_cannot_remove_unchanged_footprint(
        self,
    ):
        self.observe(100, [self.track])
        other = {
            **self.track,
            "id": "other-generation:car:1",
            "box": [0, 0, 10, 10],
            "frame_time": 101,
            "detector_observed_at": 101,
        }
        for tick in range(1, 20):
            self.assertTrue(self.observe(100 + tick / 4, [other]))

    def test_stationary_seed_does_not_replace_occupied_anchor_with_current_empty_pixels(
        self,
    ):
        self.observe(100, [self.track])
        empty = np.random.default_rng(4).integers(30, 210, (180, 120), dtype=np.uint8)
        self.observe(100.25, [{**self.track, "frame_time": 100.25}], frame=empty)
        values = [self.observe(100 + tick / 4, frame=empty) for tick in range(2, 15)]
        self.assertFalse(values[-1])

    def test_rejected_candidate_needs_tracking_grace_of_independent_full_negative_frames(
        self,
    ):
        detection = ("car", 0.6, self.box, 3600, 1, self.region)
        self.assertTrue(self.observe(100, raw=[detection]))
        for tick in range(1, 20):
            self.assertTrue(self.observe(100 + tick / 4))
        self.assertFalse(self.observe(105))
        self.observe(106, raw=[detection])
        self.assertTrue(self.observe(120, complete=False))
        self.assertTrue(self.observe(121))
        for tick in range(1, 20):
            self.assertTrue(self.observe(121 + tick / 4))
        self.assertFalse(self.observe(126))

    def test_native_initialization_resolves_same_candidate_without_waiting_an_extra_grace(
        self,
    ):
        raw = ("car", 0.8, self.box, 3600, 1, self.region)
        self.observe(100, [{**self.track, "initialized": False}], [raw])
        self.observe(
            100.25,
            [{**self.track, "frame_time": 100.25, "detector_observed_at": 100.25}],
            [raw],
        )
        self.assertEqual(self.owner.raw_pending["zone"], {})

    def test_partial_occlusion_or_door_motion_does_not_erase_remaining_footprint(self):
        self.observe(100, [self.track])
        occluded = self.frame.copy()
        occluded[40:100, 30:60] = np.random.default_rng(12).integers(
            30, 210, (60, 30), dtype=np.uint8
        )
        for tick in range(1, 80):
            self.assertTrue(self.observe(100 + tick / 4, frame=occluded))

    def test_retired_candidates_are_pruned_during_other_continuous_occupancy(self):
        raw = ("car", 0.8, self.box, 3600, 1, self.region)
        for tick in range(100):
            at = 100 + tick / 4
            candidate = {
                **self.track,
                "initialized": False,
                "id": f"candidate:{tick}",
                "frame_time": at,
                "detector_observed_at": at,
            }
            self.assertTrue(self.observe(at, [candidate], [raw]))
            self.assertLessEqual(len(self.owner.raw_pending["zone"]), 20)

    def test_stale_pixels_cannot_discharge_footprints_or_shorten_candidate_grace(self):
        self.observe(100, [self.track])
        empty = np.zeros_like(self.frame)
        for at in [80, 90, 100]:
            self.assertTrue(self.observe(at, frame=empty))
        self.assertTrue(self.observe(100.25))
        candidate = OccupancyContinuity(5)
        raw = [("car", 0.6, self.box, 3600, 1, self.region)]
        candidate.observe(self.frame, 100, self.zones, raw, [], True)
        candidate.observe(empty, 90, self.zones, [], [], True)
        self.assertTrue(
            candidate.observe(empty, 100.25, self.zones, [], [], True)[0]["uncertain"]
        )

    def test_polygon_excluded_corner_does_not_create_an_occupied_footprint(self):
        self.zones = {"zone": np.array([[0, 20], [120, 20], [0, 110]], dtype=np.int32)}
        outside = {**self.track, "box": [90, 80, 110, 100]}
        self.assertFalse(self.observe(100, [outside]))
        self.assertEqual(self.owner.footprints, {})

    def test_two_polygons_with_same_bounds_keep_separate_named_states(self):
        self.zones = {
            "left": np.array([[0, 20], [120, 20], [0, 110]], dtype=np.int32),
            "right": np.array([[0, 20], [120, 20], [120, 110]], dtype=np.int32),
        }
        outside = {**self.track, "box": [90, 80, 110, 100]}
        result = self.owner.observe(self.frame, 100, self.zones, [], [outside], True)
        self.assertEqual(
            [(r["zone"], r["uncertain"]) for r in result],
            [("left", False), ("right", True)],
        )

    def test_recreated_confirmed_tracks_coalesce_the_same_spatial_footprint(self):
        for tick in range(100):
            at = 100 + tick / 4
            track = {
                **self.track,
                "id": f"generation:{tick}",
                "frame_time": at,
                "detector_observed_at": at,
            }
            self.assertTrue(self.observe(at, [track]))
            self.assertEqual(len(self.owner.footprints), 1)
            self.assertEqual(len(self.owner.classifier.anchor_crops), 5)

    def test_same_track_replaces_jittered_footprint_without_capacity_growth(self):
        for tick in range(100):
            at = 100 + tick / 4
            offset = tick % 2
            track = {
                **self.track,
                "box": [30 + offset, 40, 90 + offset, 100],
                "frame_time": at,
                "detector_observed_at": at,
            }
            self.assertTrue(self.observe(at, [track]))
            self.assertEqual(len(self.owner.footprints), 1)
            self.assertFalse(self.owner.saturated)

    def test_pathological_history_exhaustion_is_bounded_and_explicit_unknown(self):
        for tick in range(70):
            at = 100 + tick / 4
            x, y = (tick % 10) * 10, 25 + (tick // 10) * 10
            track = {
                **self.track,
                "id": f"candidate:{tick}",
                "box": [x, y, x + 4, y + 4],
                "frame_time": at,
                "detector_observed_at": at,
            }
            self.assertTrue(self.observe(at, [track]))
            self.assertLessEqual(len(self.owner.footprints), 64)
            self.assertLessEqual(len(self.owner.classifier.anchor_crops), 320)
        self.assertEqual(self.owner.saturated, {"zone"})
        empty = np.zeros_like(self.frame)
        for tick in range(100):
            self.assertTrue(self.observe(120 + tick / 4, frame=empty))

    def test_boundary_only_contact_does_not_anchor_unoccupied_background_pixels(self):
        self.zones = {
            "zone": np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32)
        }
        touching = {**self.track, "box": [20, 40, 40, 60]}
        self.observe(100, [touching])
        self.assertEqual(self.owner.footprints, {})
        outside = {
            **touching,
            "frame_time": 100.25,
            "detector_observed_at": 100.25,
            "box": [10, 40, 20, 60],
        }
        self.assertFalse(self.observe(100.25, [outside]))
