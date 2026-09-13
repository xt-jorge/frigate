"""Padded detector pixels cannot become occupancy coverage outside the image."""

import unittest

from frigate.video.occupancy import occupancy_frame


class TestOccupancyCoverage(unittest.TestCase):
    def test_live_padded_crop_covers_only_actual_image_pixels(self):
        detection = ("car", 0.9, (10, 370, 80, 500), 9100, 0.54, (0, 0, 796, 796))
        for detections in ([], [detection]):
            with self.subTest(occupied=bool(detections)):
                frame = occupancy_frame(
                    "gate",
                    100,
                    (576, 704),
                    [(0, 367, 704, 525)],
                    [(0, 0, 796, 796)],
                    detections,
                    [],
                    [],
                )
                self.assertEqual(frame["coverage"], [[0, 0, 704, 576]])
                self.assertTrue(frame["complete"])
                self.assertEqual(len(frame["objects"]), len(detections))
                if detections:
                    self.assertEqual(frame["objects"][0]["box"], list(detection[2]))
                    self.assertEqual(frame["objects"][0]["detector_observed_at"], 100)

    def test_padding_and_partial_inference_cannot_claim_complete_coverage(self):
        frame = occupancy_frame(
            "gate",
            100,
            (576, 704),
            [(0, 367, 704, 525)],
            [(-40, -40, 796, 400), (704, 0, 796, 796), (0, 576, 796, 796)],
            [],
            [],
            [],
        )
        self.assertEqual(frame["coverage"], [[0, 0, 704, 400]])
        self.assertFalse(frame["complete"])
        self.assertEqual(frame["objects"], [])

    def test_inferred_padding_cannot_cover_an_out_of_image_requirement(self):
        frame = occupancy_frame(
            "gate",
            100,
            (576, 704),
            [(0, 367, 704, 700)],
            [(0, 0, 796, 796)],
            [],
            [],
            [],
        )
        self.assertFalse(frame["complete"])
