"""Which pixels the one OCR call gets, and where its region came from."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# frigate.embeddings must be imported before the LPR mixin: the mixin, the LPR
# model runner and the embeddings maintainer form an import cycle that only
# resolves when the embeddings package is the one that starts it.
import frigate.embeddings  # noqa: F401
from frigate.config.camera.camera import CameraTypeEnum
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)

FRAME_TIME = 1000.0
# A frame big enough that a doubled vehicle crop is a real image.
FRAME_SHAPE = (120, 160)

# Sentinel for "this track carries no detector clock at all", which is not the
# same as carrying an older one.
_ABSENT = object()


def camera_config(tracks_plates=True):
    return SimpleNamespace(
        enabled=True,
        type=CameraTypeEnum.generic,
        detect=SimpleNamespace(
            enabled=True, fps=15, stationary=SimpleNamespace(threshold=150)
        ),
        objects=SimpleNamespace(track=["license_plate"] if tracks_plates else ["car"]),
        lpr=SimpleNamespace(enabled=True, min_area=1, expire_time=10, enhancement=0),
        frame_shape=FRAME_SHAPE,
        frame_shape_yuv=(FRAME_SHAPE[0] * 3 // 2, FRAME_SHAPE[1]),
        motion=SimpleNamespace(rasterized_mask=np.ones(FRAME_SHAPE)),
    )


class PlateLocatorHarness(unittest.TestCase):
    """One OCR attempt at a time, with every locator invocation observable."""

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
        p.lp_objects = ["car", "motorcycle", "school_bus", "garbage_truck"]
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
        p._process_license_plate = MagicMock(
            return_value=(["ABC123"], [[0.9] * 6], [400])
        )
        # The secondary locator returns a box inside the doubled vehicle crop.
        p._detect_license_plate = MagicMock(return_value=(10, 60, 70, 80))

        # A gradient frame so a crop's contents identify where it came from.
        self.frame = np.arange(
            FRAME_SHAPE[0] * 3 // 2 * FRAME_SHAPE[1], dtype=np.uint8
        ).reshape(FRAME_SHAPE[0] * 3 // 2, FRAME_SHAPE[1])

    def vehicle(
        self,
        attributes=None,
        label="car",
        box=(20, 20, 100, 90),
        observed_at=FRAME_TIME,
        stationary=False,
    ):
        """A tracked vehicle believed present on FRAME_TIME.

        `observed_at` is when the model last actually measured this box. The
        tracker advances `frame_time` onto every frame the track is assumed to
        still be on, so the two are independent and a carried box has an older
        `observed_at` than the frame it arrives with.
        """
        obj = {
            "camera": "a",
            "id": "track",
            "label": label,
            "position_changes": 1,
            "stationary": stationary,
            "frame_time": FRAME_TIME,
            "box": list(box),
            "current_attributes": attributes or [],
        }
        if observed_at is not _ABSENT:
            obj["detector_observed_at"] = observed_at
        return obj

    def attribute(self, box=(40, 60, 80, 76), score=0.9, at=FRAME_TIME):
        return {
            "label": "license_plate",
            "score": score,
            "box": list(box),
            "detector_observed_at": at,
        }

    def process(self, obj, frame_time=FRAME_TIME):
        self.processor.lpr_process(obj, self.frame, source_frame_time=frame_time)

    @property
    def locator_calls(self):
        return self.processor._detect_license_plate.call_count

    @property
    def ocr_calls(self):
        return self.processor._process_license_plate.call_count


class TestPrimaryAttributePath(PlateLocatorHarness):
    def test_a_same_frame_region_reaches_ocr_with_no_locator_at_all(self):
        self.process(self.vehicle([self.attribute()]))

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 1)

    def test_the_highest_scoring_same_frame_region_wins(self):
        self.process(
            self.vehicle(
                [
                    self.attribute(box=(40, 60, 80, 76), score=0.6),
                    self.attribute(box=(40, 60, 100, 90), score=0.95),
                ]
            )
        )
        crop = self.processor._process_license_plate.call_args.args[2]

        self.assertEqual(self.locator_calls, 0)
        # The winner's box expanded by 10% and doubled for OCR: rows 57..93 and
        # columns 34..106. The loser's crop would be 38x96.
        self.assertEqual(crop.shape[:2], (72, 144))

    def test_a_region_from_another_frame_is_not_used_on_these_pixels(self):
        """current_attributes says the track has a plate, not that it is here."""
        self.process(self.vehicle([self.attribute(at=FRAME_TIME - 1)]))

        # It fell through to the locator rather than cropping the stale box.
        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)

    def test_a_fresh_region_is_read_even_when_the_vehicle_box_is_carried(self):
        """The region is its own evidence; it needs no help from the parent.

        A stationary vehicle whose box the model did not re-measure can still
        have had its plate measured on this frame, and that region describes
        these pixels on its own. The fallback's parent-freshness rule is about
        deciding where to look, not about reading a region already measured.
        """
        self.process(
            self.vehicle(
                [self.attribute()], observed_at=FRAME_TIME - 12, stationary=True
            )
        )

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 1)

    def test_a_region_with_no_clock_at_all_is_not_used(self):
        attribute = self.attribute()
        del attribute["detector_observed_at"]
        self.process(self.vehicle([attribute]))

        self.assertEqual(self.locator_calls, 1)

    def test_a_region_below_min_area_falls_through_rather_than_failing(self):
        self.processor.config.cameras["a"].lpr.min_area = 100000
        self.process(self.vehicle([self.attribute()]))

        self.assertEqual(self.locator_calls, 1)
        # The fallback's own min_area check then rejects it too, so nothing is
        # read - but the frame was still given its one honest attempt.
        self.assertEqual(self.ocr_calls, 0)

    def test_a_fresh_parent_with_an_undersized_region_falls_back_once(self):
        """Too small to read is not a region, so the vehicle is searched."""
        # 40px^2 region, below min_area; the locator's own find is 1200px^2,
        # which clears the fallback's quadrupled threshold.
        self.processor.config.cameras["a"].lpr.min_area = 100
        self.process(self.vehicle([self.attribute(box=(40, 60, 50, 64))]))

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)


class TestBoundedFallback(PlateLocatorHarness):
    def test_a_missing_region_falls_back_exactly_once_into_one_ocr_call(self):
        self.process(self.vehicle())

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)

    def test_the_fallback_crop_comes_from_this_track_and_this_frame(self):
        """No neighbour's box, no earlier frame's pixels."""
        self.process(self.vehicle(box=(20, 20, 100, 90)))
        car = self.processor._detect_license_plate.call_args.args[1]

        # The doubled crop of exactly this box out of exactly this frame.
        expected = cv2_resize_bgr(self.frame, (20, 20, 100, 90))
        self.assertEqual(car.shape, expected.shape)
        np.testing.assert_array_equal(car, expected)

    def test_an_adjacent_vehicle_gets_its_own_crop(self):
        self.process(self.vehicle(box=(20, 20, 60, 60)))
        near = self.processor._detect_license_plate.call_args.args[1]
        self.processor._detect_license_plate.reset_mock()
        self.processor._detect_license_plate.return_value = (10, 60, 70, 80)
        self.process(self.vehicle(box=(90, 20, 130, 60)))
        far = self.processor._detect_license_plate.call_args.args[1]

        self.assertEqual(near.shape, far.shape)
        self.assertFalse(np.array_equal(near, far))

    def test_a_locator_that_finds_nothing_reads_nothing(self):
        self.processor._detect_license_plate.return_value = None
        self.process(self.vehicle())

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 0)

    def test_a_vehicle_with_no_box_never_reaches_the_locator(self):
        obj = self.vehicle()
        obj["box"] = []
        self.process(obj)

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)

    def test_a_box_that_selects_no_pixels_is_refused_before_resizing(self):
        self.process(self.vehicle(box=(50, 50, 50, 90)))

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)

    def test_a_plate_that_is_itself_the_track_has_nowhere_to_fall_back_to(self):
        obj = self.vehicle(label="license_plate", box=(40, 60, 80, 76))
        obj["current_attributes"] = []
        self.process(obj)

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 1)

    def test_every_plus_vehicle_label_may_fall_back(self):
        for label in ("car", "motorcycle", "school_bus", "garbage_truck"):
            with self.subTest(label=label):
                self.processor._detect_license_plate.reset_mock()
                self.processor._process_license_plate.reset_mock()
                self.process(self.vehicle(label=label))
                self.assertEqual(self.locator_calls, 1)
                self.assertEqual(self.ocr_calls, 1)


class TestFallbackParentCustody(PlateLocatorHarness):
    """The fallback may only look inside a vehicle the model measured here.

    A tracker advances `frame_time` onto every frame a track is believed to
    still be on while keeping the older `detector_observed_at` of the box it
    actually measured. Looking inside a carried box and publishing whatever
    plate is found there attaches that plate to this track on geometry nothing
    confirmed for this image - in close headway, the next car's plate.
    """

    def test_a_carried_box_over_a_newer_frame_emits_no_read_and_no_identity(self):
        """Vehicle A measured at t0; at t1 car B's plate sits in A's old box."""
        carried = self.vehicle(observed_at=FRAME_TIME - 12, stationary=True)
        self.assertEqual(carried["frame_time"], FRAME_TIME)

        self.process(carried)

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)
        # Nothing was attributed to A: no sub label, no plate publication.
        self.processor.sub_label_publisher.publish.assert_not_called()
        self.processor.requestor.send_data.assert_not_called()
        self.assertEqual(self.processor.detected_license_plates, {})

    def test_the_same_vehicle_measured_on_this_frame_still_gets_one_of_each(self):
        self.process(self.vehicle(stationary=True))

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)

    def test_clocks_that_round_to_the_same_millisecond_are_still_different(self):
        """Rounded equality is not measurement identity."""
        near = FRAME_TIME + 0.0004
        self.assertEqual(round(near * 1000), round(FRAME_TIME * 1000))

        self.process(self.vehicle(observed_at=near))

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)

    def test_absent_and_unusable_clocks_are_refused_rather_than_assumed_current(self):
        for name, observed_at in (
            ("absent", _ABSENT),
            ("null", None),
            ("not a number", "1000.0"),
            ("boolean", True),
            ("future", FRAME_TIME + 1),
        ):
            with self.subTest(name=name):
                self.processor._detect_license_plate.reset_mock()
                self.processor._process_license_plate.reset_mock()
                self.process(self.vehicle(observed_at=observed_at))
                self.assertEqual(self.locator_calls, 0)
                self.assertEqual(self.ocr_calls, 0)

    def test_the_scheduler_predicate_and_the_processor_agree(self):
        """One rule, asked twice: at candidate selection and before acting."""
        fresh = self.vehicle()
        carried = self.vehicle(observed_at=FRAME_TIME - 12, stationary=True)
        with_region = self.vehicle(
            [self.attribute()], observed_at=FRAME_TIME - 12, stationary=True
        )

        self.assertTrue(self.processor.plate_region_available("a", fresh, FRAME_TIME))
        self.assertFalse(
            self.processor.plate_region_available("a", carried, FRAME_TIME)
        )
        # A region measured on this frame is enough on its own.
        self.assertTrue(
            self.processor.plate_region_available("a", with_region, FRAME_TIME)
        )

    def test_a_carried_parent_with_a_region_too_small_to_read_is_unavailable(self):
        """Availability must mean readable, or the slot buys nothing.

        The processor would discard this region on min_area and then refuse the
        fallback for the stale parent. If the scheduler called that available,
        the attempt clocks would be spent on a frame that can yield nothing -
        which is the phase starvation this rule exists to prevent.
        """
        self.processor.config.cameras["a"].lpr.min_area = 100
        undersized = self.vehicle(
            [self.attribute(box=(40, 60, 50, 64))],
            observed_at=FRAME_TIME - 12,
            stationary=True,
        )

        self.assertFalse(
            self.processor.plate_region_available("a", undersized, FRAME_TIME)
        )

        self.process(undersized)
        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)

    def test_the_predicate_leaves_the_pre_existing_paths_alone(self):
        carried = self.vehicle(observed_at=FRAME_TIME - 12, stationary=True)

        # A dedicated Frigate+ LPR camera tracks the plate itself.
        plate_track = self.vehicle(
            label="license_plate", observed_at=FRAME_TIME - 12, stationary=True
        )
        self.assertTrue(
            self.processor.plate_region_available("a", plate_track, FRAME_TIME)
        )

        # A camera with no plate-detecting model has only ever had the
        # secondary locator; its behaviour is deliberately unchanged here.
        self.processor.config.cameras["a"] = camera_config(tracks_plates=False)
        self.assertTrue(self.processor.plate_region_available("a", carried, FRAME_TIME))


class TestReprojectedRegions(PlateLocatorHarness):
    """The post processor re-reads a plate off a recording keyframe.

    It maps the stored regions onto that image itself, so no detector clock can
    vouch for them - and its vehicle box is in the detect stream's coordinates,
    so there is nothing honest to fall back to.
    """

    def reprocess(self, obj, frame_time=FRAME_TIME):
        self.processor.lpr_process(
            obj,
            self.frame,
            source_frame_time=frame_time,
            reprojected_regions=True,
        )

    def test_a_reprojected_region_is_read_without_a_matching_clock(self):
        self.reprocess(self.vehicle([self.attribute(at=FRAME_TIME - 90)]))

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 1)

    def test_a_reprojected_object_never_falls_back_to_the_vehicle_crop(self):
        self.reprocess(self.vehicle())

        self.assertEqual(self.locator_calls, 0)
        self.assertEqual(self.ocr_calls, 0)


class TestSecondaryOnlyPathUnchanged(PlateLocatorHarness):
    """A camera that does not track license_plate still behaves as it did."""

    def setUp(self):
        super().setUp()
        self.processor.config.cameras["a"] = camera_config(tracks_plates=False)

    def test_the_locator_runs_once_and_ignores_any_attribute(self):
        self.process(self.vehicle([self.attribute()]))

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)

    def test_a_carried_box_is_still_read_here_as_it_always_was(self):
        """Deliberately not tightened: this is the only locator on this path.

        Requiring a fresh parent measurement here would remove recognition
        from every frame the detector did not re-scan the vehicle on, which at
        the default stationary interval is nearly all of them. That is a
        pre-existing limitation of a path this work did not introduce, not a
        new association the Plus fallback mints.
        """
        self.process(self.vehicle(observed_at=FRAME_TIME - 12, stationary=True))

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 1)

    def test_nothing_found_means_nothing_read(self):
        self.processor._detect_license_plate.return_value = None
        self.process(self.vehicle())

        self.assertEqual(self.locator_calls, 1)
        self.assertEqual(self.ocr_calls, 0)


def cv2_resize_bgr(frame, box):
    """The doubled BGR crop the locator is expected to receive."""
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    left, top, right, bottom = box
    car = rgb[top:bottom, left:right]
    return cv2.resize(car, (int(2 * car.shape[1]), int(2 * car.shape[0])))


if __name__ == "__main__":
    unittest.main()
