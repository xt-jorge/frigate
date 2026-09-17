"""Frigate+ vehicle classes must reach every gate that names vehicles by hand.

Frigate+ emits `school_bus` and `garbage_truck` as their own classes and has no
generic `truck` or `bus`. A gate written against the COCO names therefore drops
those vehicles silently: they are tracked, but they never get a plate read, a
capture frame or a stationary allowance.
"""

import unittest
from types import SimpleNamespace

import numpy as np

import frigate.embeddings  # noqa: F401
from frigate.camera.state import FrozenVehicle
from frigate.const import (
    DEFAULT_ATTRIBUTE_LABEL_MAP,
    TRACK_FRAME_LABELS,
    VEHICLE_LABELS,
)
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.test import test_tracked_object_publication as tracked_fixture
from frigate.track.stationary_classifier import get_stationary_threshold
from frigate.util.image import is_better_thumbnail

PLUS_ONLY = ("school_bus", "garbage_truck")


class TestPlusVehicleLabelMap(unittest.TestCase):
    def test_the_plus_classes_are_vehicles_and_capture_subjects(self):
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                self.assertIn(label, VEHICLE_LABELS)
                self.assertIn(label, TRACK_FRAME_LABELS)

    def test_an_attribute_label_is_never_a_vehicle(self):
        for label in ("license_plate", "face", "person", "waste_bin"):
            with self.subTest(label=label):
                self.assertNotIn(label, VEHICLE_LABELS)

    def test_the_plus_classes_carry_plates(self):
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                self.assertIn(
                    "license_plate", DEFAULT_ATTRIBUTE_LABEL_MAP.get(label, [])
                )

    def test_lp_objects_is_derived_from_the_map_rather_than_a_second_list(self):
        processor = LicensePlateProcessingMixin.__new__(LicensePlateProcessingMixin)
        processor.config = SimpleNamespace(
            model=SimpleNamespace(attributes_map=DEFAULT_ATTRIBUTE_LABEL_MAP)
        )
        lp_objects = [
            label
            for label, attributes in processor.config.model.attributes_map.items()
            if "license_plate" in attributes
        ]

        for label in PLUS_ONLY:
            self.assertIn(label, lp_objects)
        self.assertNotIn("person", lp_objects)

    def test_the_plus_classes_get_the_vehicle_stationary_allowance(self):
        car = get_stationary_threshold("car")
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                self.assertEqual(get_stationary_threshold(label), car)

    def test_a_plate_on_a_plus_vehicle_still_wins_the_thumbnail(self):
        current = {
            "attributes": [],
            "box": (10, 10, 60, 60),
            "area": 2500,
            "score": 0.8,
            "region": (0, 0, 320, 240),
        }
        new = {
            **current,
            "attributes": [
                {"label": "license_plate", "score": 0.9, "box": (1, 1, 9, 5)}
            ],
        }
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                self.assertTrue(is_better_thumbnail(label, current, new, (240, 320)))


class TestPlusVehicleCapture(unittest.TestCase):
    """A Plus vehicle track must be able to supply a detector frame."""

    def setUp(self):
        self.fixture = tracked_fixture.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = self.fixture.state
        self.state.camera_config.enabled = True
        self.state.camera_config.detect.enabled = True
        objects = self.state.camera_config.objects
        for label in PLUS_ONLY:
            objects.track.append(label)
            objects.filters[label] = objects.filters["car"].model_copy(deep=True)

    def relabel(self, label):
        """Publish frames for a track of this Plus vehicle class."""
        self.fixture.frame_time += 1
        detection = {
            "id": tracked_fixture.EVENT_ID,
            "label": label,
            "frame_time": self.fixture.frame_time,
            "detector_observed_at": self.fixture.frame_time,
            "start_time": 100.0,
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
        self.state.update(
            f"frame-{self.fixture.frame_time}",
            self.fixture.frame_time,
            {tracked_fixture.EVENT_ID: detection},
            [],
            [],
            (),
        )

    def test_a_plus_vehicle_track_yields_pixels_and_frozen_geometry(self):
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                for _ in range(4):
                    self.relabel(label)
                snapshot = self.state.get_track_frame(tracked_fixture.EVENT_ID)
                frame, frame_time, track, vehicles, faces = snapshot

                self.assertIsInstance(frame, np.ndarray)
                self.assertEqual(track[1], label)
                self.assertEqual(
                    vehicles,
                    (
                        FrozenVehicle(
                            (100, 100, 200, 200),
                            tracked_fixture.EVENT_ID,
                            frame_time,
                        ),
                    ),
                )
                self.assertEqual(faces, ())
                self.state.finished(tracked_fixture.EVENT_ID)
                self.state.tracked_objects.pop(tracked_fixture.EVENT_ID, None)

    def test_an_attribute_track_is_never_a_frozen_vehicle(self):
        objects = self.state.camera_config.objects
        objects.track.append("license_plate")
        objects.filters["license_plate"] = objects.filters["car"].model_copy(deep=True)
        for _ in range(4):
            self.relabel("license_plate")

        self.assertEqual(self.state._current_frame_vehicles, ())
        self.assertIsNone(self.state.get_track_frame(tracked_fixture.EVENT_ID)[0])


class TestPlusVehicleTaxonomyIsNotWidened(unittest.TestCase):
    def test_a_garbage_truck_is_a_vehicle_and_not_a_container(self):
        """Adding the class must not drag unrelated labels in with it."""
        self.assertIn("garbage_truck", VEHICLE_LABELS)
        self.assertIn("license_plate", DEFAULT_ATTRIBUTE_LABEL_MAP["garbage_truck"])
        for unrelated in ("waste_bin", "package", "bbq_grill"):
            with self.subTest(label=unrelated):
                self.assertNotIn(unrelated, VEHICLE_LABELS)
                self.assertNotIn(unrelated, TRACK_FRAME_LABELS)

    def test_the_plus_classes_carry_no_delivery_logos(self):
        """Only the plate is claimed for them; nothing else was copied over."""
        for label in PLUS_ONLY:
            with self.subTest(label=label):
                self.assertEqual(DEFAULT_ATTRIBUTE_LABEL_MAP[label], ["license_plate"])


if __name__ == "__main__":
    unittest.main()
