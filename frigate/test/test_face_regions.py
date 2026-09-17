"""Raw face regions must ride the exact frame the model measured them on."""

import unittest
from queue import Queue
from unittest.mock import MagicMock, patch

from frigate.camera.state import FrozenFace
from frigate.test import test_tracked_object_publication as fixtures
from frigate.video.detect import process_frames

FACE = ("face", 0.86, (140, 118, 172, 156), 1216, 0.842, (0, 0, 320, 240))
PLATE = ("license_plate", 0.91, (130, 180, 180, 196), 800, 3.125, (0, 0, 320, 240))
CAR = ("car", 0.9, (100, 100, 200, 200), 10000, 1.0, (0, 0, 320, 240))


class FaceRegionHarness(unittest.TestCase):
    """Drive the real detection loop and read what it hands the processor."""

    def setUp(self):
        self.fixture = fixtures.TestRecognizedPlatePublication()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.processor.config
        self.camera = self.config.cameras["front"]
        self.camera.detect.enabled = True
        self.camera.objects.track = ["car", "license_plate", "face"]

    def run_frames(
        self, times, detections=(), regions=((0, 0, 320, 240),), motion_boxes=()
    ):
        """Return every queued payload from one pass of the detection loop.

        `regions` only seeds the startup scan, so a run over several frames
        needs motion to keep giving the detector something to look at.
        """
        frames, output = Queue(), Queue()
        for time in times:
            frames.put(("fixture", time))
        tracker = MagicMock()
        tracker.tracked_objects = {}
        tracker.occupancy_tracks.return_value = []
        tracker.untracked_object_boxes = []
        detector = MagicMock()
        detector.last_detection_successful = True
        motion = MagicMock()
        motion.detect.return_value = list(motion_boxes)
        motion.is_calibrating.return_value = not motion_boxes
        stop = MagicMock()
        stop.is_set.return_value = False

        def inference(*args):
            if args[-1] is not None:
                args[-1].extend(detections)
            return list(detections)

        with (
            patch("frigate.video.detect.CameraConfigUpdateSubscriber") as subscriber,
            patch(
                "frigate.video.detect.get_startup_regions", return_value=list(regions)
            ),
            patch("frigate.video.detect.ptz_moving_at_frame_time", return_value=False),
            patch("frigate.video.detect.detect", side_effect=inference),
        ):
            subscriber.return_value.check_for_updates.return_value = []
            process_frames(
                MagicMock(),
                frames,
                (240, 320),
                self.config.model,
                self.camera,
                self.fixture.processor.frame_manager,
                motion,
                detector,
                tracker,
                output,
                MagicMock(),
                stop,
                MagicMock(),
                [],
                exit_on_empty=True,
            )
        payloads = []
        while not output.empty():
            payloads.append(output.get_nowait())
        return payloads


class TestRawFaceRegions(FaceRegionHarness):
    def test_a_windshield_face_with_no_person_still_reaches_the_frame(self):
        """The whole point: face maps to person, and there is no person here."""
        self.assertEqual(
            self.config.model.attributes_map.get("car", []).count("face"), 0
        )
        payload = self.run_frames([100], detections=[CAR, FACE])[0]

        self.assertEqual(payload[-1], (FrozenFace((140, 118, 172, 156), 0.86, 100),))
        # ... and it really was assigned to no tracked object.
        self.assertEqual(
            [obj["attributes"] for obj in payload[3].values()],
            [[] for _ in payload[3]],
        )

    def test_a_frame_with_no_face_pass_is_not_a_frame_with_no_faces(self):
        """None and () are different claims and must stay different."""
        self.camera.objects.track = ["car"]
        untracked = self.run_frames([100], detections=[CAR])[0]
        self.assertIsNone(untracked[-1])

        self.camera.objects.track = ["car", "face"]
        empty_pass = self.run_frames([100], detections=[CAR])[0]
        self.assertEqual(empty_pass[-1], ())

    def test_a_frame_the_detector_never_ran_on_carries_no_hint(self):
        no_regions = self.run_frames([100], detections=[CAR], regions=())[0]
        self.assertIsNone(no_regions[-1])

    def test_a_disabled_detector_carries_no_hint(self):
        self.camera.detect.enabled = False
        self.assertIsNone(self.run_frames([100], detections=[CAR, FACE])[0][-1])

    def test_each_frame_gets_its_own_regions_and_clock(self):
        for clock in (100, 101.5):
            with self.subTest(clock=clock):
                payload = self.run_frames([clock], detections=[CAR, FACE])[0]
                # The region's clock is the frame's clock, unrounded. Nothing
                # here is allowed to be near enough.
                self.assertEqual(payload[2], clock)
                self.assertEqual(payload[-1][0].detector_observed_at, clock)

    def test_a_later_frame_without_a_pass_does_not_inherit_the_earlier_regions(self):
        """Frame two has nothing to look at, so it must claim nothing."""
        payloads = self.run_frames([100, 101], detections=[CAR, FACE])
        self.assertEqual(len(payloads), 2)
        self.assertEqual(
            payloads[0][-1], (FrozenFace((140, 118, 172, 156), 0.86, 100),)
        )
        self.assertIsNone(payloads[1][-1])


class TestAttributesAreNotOccupants(FaceRegionHarness):
    """A plate is a property of a car, not a second car standing next to it."""

    def setUp(self):
        super().setUp()
        self.camera.detect.occupancy_zones = ["approach"]

    def test_plate_and_face_boxes_are_neither_objects_nor_candidates(self):
        occupancy = self.run_frames([100], detections=[CAR, FACE, PLATE])[0][6]

        self.assertEqual(
            [obj["label"] for obj in occupancy["objects"]],
            ["car"],
        )
        self.assertNotIn(
            "face", [region.get("label") for region in occupancy["regions"]]
        )

    def test_an_attribute_alone_leaves_the_zone_empty(self):
        occupancy = self.run_frames([100], detections=[FACE, PLATE])[0][6]
        self.assertEqual(occupancy["objects"], [])


if __name__ == "__main__":
    unittest.main()
