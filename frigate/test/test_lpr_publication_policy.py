"""Published OCR results describe one actual current sample, not an aggregate."""

import json
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

TRACK_ID = "track"


def camera_config():
    return SimpleNamespace(
        enabled=True,
        type=CameraTypeEnum.generic,
        detect=SimpleNamespace(
            enabled=True, fps=15, stationary=SimpleNamespace(threshold=150)
        ),
        objects=SimpleNamespace(track=["license_plate"]),
        lpr=SimpleNamespace(enabled=True, min_area=1, expire_time=10, enhancement=0),
        frame_shape=(32, 32),
        frame_shape_yuv=(48, 32),
        motion=SimpleNamespace(rasterized_mask=np.ones((32, 32))),
    )


class CurrentSampleFixture(unittest.TestCase):
    """Drive the mixin with scripted OCR results and read what it published."""

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
        p.lp_objects = ["car"]
        p.stationary_scan_duration = 5
        p.cluster_threshold = 0.85
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
        p._process_license_plate = MagicMock()
        self.obj = {
            "camera": "a",
            "id": TRACK_ID,
            "label": "license_plate",
            "position_changes": 1,
            "stationary": False,
            "box": [5, 5, 25, 15],
        }
        self.frame = np.zeros((48, 32), np.uint8)

    def sample(self, text, confidence, clock, area=100):
        """Feed one OCR result that already passed the recognizer."""
        self.processor._process_license_plate.return_value = (
            [text],
            [[confidence] * len(text)],
            [area],
        )
        self.processor.lpr_process(self.obj, self.frame, source_frame_time=clock)

    def published(self):
        """The (text, score, capture clock) triples published as attributes."""
        return [
            (call.args[0][2], round(call.args[0][3], 6), call.args[0][4])
            for call in self.processor.sub_label_publisher.publish.call_args_list
            if len(call.args[0]) == 5 and call.args[0][1] == "recognized_license_plate"
        ]

    def tracked_object_updates(self):
        """The same triples as seen on the tracked_object_update rail."""
        payloads = [
            json.loads(call.args[1])
            for call in self.processor.requestor.send_data.call_args_list
            if call.args[0] == "tracked_object_update"
        ]
        return [
            (
                payload["plate"],
                round(payload["score"], 6),
                payload["timestamp"],
                payload["recognized_license_plate_frame_time"],
            )
            for payload in payloads
        ]

    def aggregate(self):
        return self.processor.detected_license_plates[TRACK_ID]["plate"]

    def sample_clock(self):
        return self.processor.detected_license_plates[TRACK_ID].get(
            "recognized_license_plate_frame_time"
        )


class TestCurrentSamplePublication(CurrentSampleFixture):
    def test_a_lower_scoring_complete_read_is_published_over_the_representative(self):
        """The reproduced defect: clustering must not swallow a real sample.

        Mirrors current-sample-policy-probe: a fresh accepted 1ABC234 at 0.921
        was dropped because clustering still preferred the earlier ABC234.
        """
        self.sample("ABC234", 0.958, 100.0)
        self.sample("1ABC234", 0.921, 101.0)

        self.assertEqual(
            self.published(),
            [("ABC234", 0.958, 100.0), ("1ABC234", 0.921, 101.0)],
        )
        self.assertEqual(self.sample_clock(), 101.0)
        # The display aggregate is free to disagree - it just may not publish.
        self.assertEqual(self.aggregate(), "ABC234")

    def test_every_published_triple_is_internally_coherent(self):
        self.sample("ABC234", 0.958, 100.0)
        self.sample("1ABC234", 0.921, 101.0)

        self.assertEqual(
            self.tracked_object_updates(),
            [
                ("ABC234", 0.958, 100.0, 100.0),
                ("1ABC234", 0.921, 101.0, 101.0),
            ],
        )

    def test_repeated_identical_reads_advance_the_clock_without_regressing(self):
        self.sample("ABC123", 0.90, 100.0)
        self.sample("ABC123", 0.88, 101.0)
        self.sample("ABC123", 0.99, 100.5)
        self.sample("ABC123", 0.95, 101.0)
        self.sample("ABC123", 0.80, 102.0)

        self.assertEqual(
            self.published(),
            [
                ("ABC123", 0.90, 100.0),
                ("ABC123", 0.88, 101.0),
                ("ABC123", 0.80, 102.0),
            ],
        )
        self.assertEqual(self.sample_clock(), 102.0)

    def test_changed_text_publishes_while_the_aggregate_keeps_its_representative(self):
        self.sample("ABC123", 0.99, 100.0)
        self.sample("ABC123", 0.98, 101.0)
        aggregate_before = self.aggregate()

        self.sample("ABD123", 0.75, 102.0)

        self.assertEqual(self.published()[-1], ("ABD123", 0.75, 102.0))
        self.assertEqual(self.aggregate(), aggregate_before)
        self.assertEqual(aggregate_before, "ABC123")

    def test_a_shorter_later_read_publishes_so_length_is_never_preferred(self):
        self.sample("1ABC234", 0.95, 100.0)
        self.sample("ABC234", 0.80, 101.0)

        self.assertEqual(
            self.published(),
            [("1ABC234", 0.95, 100.0), ("ABC234", 0.80, 101.0)],
        )

    def test_a_sample_below_the_recognition_threshold_never_reaches_publication(self):
        self.sample("ABC123", 0.95, 100.0)
        self.sample("ABD123", 0.69, 101.0)

        self.assertEqual(self.published(), [("ABC123", 0.95, 100.0)])
        self.assertEqual(self.sample_clock(), 100.0)
        self.assertEqual(
            len(self.processor.detected_license_plates[TRACK_ID]["plates"]), 1
        )


class TestCurrentSampleFilters(CurrentSampleFixture):
    def test_a_short_current_sample_is_not_published_behind_a_passing_aggregate(self):
        self.processor.lpr_config.min_plate_length = 6
        self.sample("ABC123", 0.95, 100.0)
        self.sample("ABC12", 0.90, 101.0)

        self.assertEqual(self.published(), [("ABC123", 0.95, 100.0)])
        self.assertEqual(self.aggregate(), "ABC123")
        # A rejected sample must not consume the clock a later good one needs.
        self.assertEqual(self.sample_clock(), 100.0)
        self.sample("ABC124", 0.80, 101.0)
        self.assertEqual(self.published()[-1], ("ABC124", 0.80, 101.0))

    def test_a_misformatted_current_sample_is_not_published_behind_a_passing_aggregate(
        self,
    ):
        self.processor.lpr_config.format = "^[A-Z]{3}[0-9]{3}$"
        self.sample("ABC123", 0.95, 100.0)
        self.sample("AB1234", 0.90, 101.0)

        self.assertEqual(self.published(), [("ABC123", 0.95, 100.0)])
        self.assertEqual(self.sample_clock(), 100.0)

    def test_a_passing_current_sample_publishes_even_when_the_aggregate_fails(self):
        self.processor.lpr_config.min_plate_length = 6
        self.sample("AB12", 0.99, 100.0)
        self.assertEqual(self.published(), [])

        self.sample("ABC123", 0.80, 101.0)

        # Clustering still prefers the higher-confidence AB12, which the filters
        # reject, so nothing was ever stored as the aggregate.
        self.assertEqual(self.aggregate(), "")
        self.assertEqual(self.published(), [("ABC123", 0.80, 101.0)])
        self.assertEqual(self.sample_clock(), 101.0)


class TestCurrentSampleSubLabel(CurrentSampleFixture):
    def sub_labels(self):
        return [
            (call.args[0][0], call.args[0][1], round(call.args[0][2], 6))
            for call in self.processor.sub_label_publisher.publish.call_args_list
            if len(call.args[0]) == 3
        ]

    def names(self):
        return [
            json.loads(call.args[1])["name"]
            for call in self.processor.requestor.send_data.call_args_list
            if call.args[0] == "tracked_object_update"
        ]

    def test_the_sub_label_names_the_text_that_was_published(self):
        self.processor.lpr_config.known_plates = {"resident": ["ABC123"]}
        self.sample("ABC123", 0.95, 100.0)
        self.assertEqual(self.sub_labels(), [(TRACK_ID, "resident", 0.95)])

        # A later disagreeing sample publishes itself and claims no sub label,
        # even though the stored aggregate still matches the known plate.
        self.sample("ABC124", 0.90, 101.0)

        self.assertEqual(self.aggregate(), "ABC123")
        self.assertEqual(self.published()[-1], ("ABC124", 0.90, 101.0))
        self.assertEqual(self.sub_labels(), [(TRACK_ID, "resident", 0.95)])
        self.assertEqual(self.names(), ["resident", None])


class TestRecognitionPreprocessing(unittest.TestCase):
    """What each preprocessing branch actually hands the recognizer."""

    def processor(self, enhancement=0):
        processor = LicensePlateProcessingMixin.__new__(LicensePlateProcessingMixin)
        processor.config = SimpleNamespace(
            cameras={"a": SimpleNamespace(lpr=SimpleNamespace(enhancement=enhancement))}
        )
        processor.model_runner = SimpleNamespace(
            recognition_model=SimpleNamespace(
                runner=SimpleNamespace(
                    # v6 declares a dynamic input width, which is not an int.
                    get_input_width=lambda: "DynamicDimension.1"
                )
            )
        )
        return processor

    def coloured_crop(self):
        """A BGR crop whose three channels are distinguishable on sight."""
        crop = np.zeros((24, 48, 3), np.uint8)
        crop[:, :, 0] = 30  # blue
        crop[:, :, 1] = 140  # green
        crop[:, :, 2] = 220  # red
        return crop

    def test_the_unused_width_is_zero_padded_not_filled_with_the_crop_mean(self):
        crop = np.full((24, 48, 3), 200, dtype=np.uint8)

        padded = self.processor()._preprocess_recognition_image("a", crop, 320 / 48)

        self.assertEqual(padded.shape, (3, 48, 320))
        resized_w = 96
        self.assertTrue(np.all(padded[:, :, resized_w:] == 0.0))
        # The real content is normalized to (x/255 - 0.5) / 0.5, so a bright
        # crop must not be confused with the zero padding beside it.
        self.assertGreater(float(padded[:, :, :resized_w].min()), 0.0)

    def test_an_unenhanced_crop_reaches_the_recognizer_in_colour(self):
        """PP-OCR text recognition is exported for colour input.

        With no enhancement asked for, the crop `lpr_process` decoded must
        arrive channel for channel, not flattened to luminance.
        """
        crop = self.coloured_crop()

        padded = self.processor(enhancement=0)._preprocess_recognition_image(
            "a", crop, 320 / 48
        )

        content = padded[:, :, :96]
        self.assertEqual(
            [round(float(content[c].mean()), 6) for c in range(3)],
            [round((v / 255.0 - 0.5) / 0.5, 6) for v in (30, 140, 220)],
        )

    def test_enhancement_hands_the_recognizer_one_channel_on_purpose(self):
        """The denoise/CLAHE operators are single channel; say so in the data.

        An operator who turns enhancement on is choosing those filters, and
        they only exist for luminance - so all three channels come back equal.
        """
        rng = np.random.default_rng(7)
        crop = rng.integers(0, 255, (24, 48, 3), dtype=np.uint8)

        for enhancement in (1, 5, 8):
            with self.subTest(enhancement=enhancement):
                padded = self.processor(
                    enhancement=enhancement
                )._preprocess_recognition_image("a", crop, 320 / 48)

                content = padded[:, :, :96]
                self.assertTrue(np.array_equal(content[0], content[1]))
                self.assertTrue(np.array_equal(content[1], content[2]))
                # Whatever the filters did, the unused width is still zeros.
                self.assertTrue(np.all(padded[:, :, 96:] == 0.0))

    def test_only_the_enhanced_branch_discards_colour(self):
        """The two branches are genuinely different inputs, not one path."""
        crop = self.coloured_crop()

        plain = self.processor(enhancement=0)._preprocess_recognition_image(
            "a", crop, 320 / 48
        )
        enhanced = self.processor(enhancement=1)._preprocess_recognition_image(
            "a", crop, 320 / 48
        )

        self.assertFalse(np.array_equal(plain[0], plain[2]))
        self.assertTrue(np.array_equal(enhanced[0], enhanced[2]))


if __name__ == "__main__":
    unittest.main()
