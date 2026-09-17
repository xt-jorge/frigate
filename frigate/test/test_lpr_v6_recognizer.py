"""The pinned PP-OCRv6 medium artifact, its label map and a real decode.

These tests read the recognition artifact itself, so they need it on disk.
Point `FRIGATE_LPR_MODEL_DIR` at a directory holding the pinned recognizer and
its inference config - either under the cache names Frigate downloads them as,
or under the upstream names they carry in the model repository - or let them
fall back to the `paddleocr-onnx` model cache. Without it there is nothing to
assert against and they skip rather than pretend.
"""

import os
import unittest

import cv2
import numpy as np

# frigate.embeddings must be imported before the LPR mixin: the mixin, the LPR
# model runner and the embeddings maintainer form an import cycle that only
# resolves when the embeddings package is the one that starts it.
import frigate.embeddings  # noqa: F401
from frigate.const import MODEL_CACHE_DIR
from frigate.data_processing.common.license_plate.mixin import (
    CTCDecoder,
    LicensePlateProcessingMixin,
)
from frigate.embeddings.onnx.lpr_embedding import (
    PPOCRV6_MEDIUM_CLASS_COUNT,
    PPOCRV6_MEDIUM_CONFIG_FILE,
    PPOCRV6_MEDIUM_MODEL_FILE,
)

# The dictionary index PaddleOCR gives U+3000 IDEOGRAPHIC SPACE, and the
# dictionary indices of the glyphs plates are made of. Spelled as a code point
# because the character itself is invisible and would not survive an editor.
IDEOGRAPHIC_SPACE = chr(0x3000)
IDEOGRAPHIC_SPACE_INDEX = 1748
GLYPH_INDICES = {32: "0", 41: "9", 42: "A", 67: "Z"}


def artifact_dir():
    for candidate in (
        os.environ.get("FRIGATE_LPR_MODEL_DIR"),
        os.path.join(MODEL_CACHE_DIR, "paddleocr-onnx"),
    ):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def artifact(directory, cache_name, upstream_name):
    for name in (cache_name, upstream_name):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


DIRECTORY = artifact_dir()
MODEL_PATH = (
    artifact(DIRECTORY, PPOCRV6_MEDIUM_MODEL_FILE, "inference.onnx")
    if DIRECTORY
    else None
)
CONFIG_PATH = (
    artifact(DIRECTORY, PPOCRV6_MEDIUM_CONFIG_FILE, "inference.yml")
    if DIRECTORY
    else None
)


@unittest.skipUnless(CONFIG_PATH, "recognition inference config not available")
class TestRecognitionLabelMap(unittest.TestCase):
    """The label map the decoder loads must be the one the model was trained on."""

    def setUp(self):
        self.decoder = CTCDecoder(CONFIG_PATH, PPOCRV6_MEDIUM_CLASS_COUNT)

    @unittest.skipUnless(MODEL_PATH, "recognition model not available")
    def test_the_asserted_class_count_is_the_models_own_output_width(self):
        """The constant the loader fails closed on is measured, not guessed."""
        import onnxruntime as ort

        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        output_width = session.get_outputs()[0].shape[-1]

        self.assertEqual(output_width, PPOCRV6_MEDIUM_CLASS_COUNT)
        self.assertEqual(len(self.decoder.characters), output_width)

    def test_a_label_map_of_the_wrong_size_refuses_to_load(self):
        with self.assertRaises(ValueError):
            CTCDecoder(CONFIG_PATH, PPOCRV6_MEDIUM_CLASS_COUNT - 1)

    def test_a_missing_label_map_refuses_to_load(self):
        with self.assertRaises(FileNotFoundError):
            CTCDecoder(
                os.path.join(DIRECTORY, "no-such-inference.yml"),
                PPOCRV6_MEDIUM_CLASS_COUNT,
            )

    def test_the_ideographic_space_entry_survives_loading(self):
        """The trap: U+3000 is whitespace to Python but a class to the model."""
        # characters is ["blank"] + dictionary + [" "], so the dictionary sits
        # one place along.
        self.assertEqual(
            self.decoder.characters[1 + IDEOGRAPHIC_SPACE_INDEX], IDEOGRAPHIC_SPACE
        )

    def test_dropping_whitespace_entries_would_shift_the_map_past_the_plate_glyphs(
        self,
    ):
        """Why the loader may not strip: the corruption hides behind plates.

        A strip-and-drop loader loses exactly one entry, and it sits after every
        glyph a licence plate is made of - so Latin plates keep decoding while
        every CJK class above it is silently wrong.
        """
        dictionary = self.decoder.characters[1:-1]
        kept = [entry for entry in dictionary if entry.strip()]

        self.assertEqual(len(kept), len(dictionary) - 1)
        divergence = next(i for i, entry in enumerate(kept) if entry != dictionary[i])
        self.assertEqual(divergence, IDEOGRAPHIC_SPACE_INDEX)
        self.assertLess(max(GLYPH_INDICES), divergence)

    def test_the_plate_glyphs_sit_where_paddleocr_puts_them(self):
        for index, glyph in GLYPH_INDICES.items():
            with self.subTest(glyph=glyph):
                self.assertEqual(self.decoder.characters[1 + index], glyph)


@unittest.skipUnless(
    MODEL_PATH and CONFIG_PATH, "recognition model or config not available"
)
class TestRecognitionEndToEnd(unittest.TestCase):
    """A rendered plate through the real preprocessing and the real decoder."""

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort

        cls.session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        cls.decoder = CTCDecoder(CONFIG_PATH, PPOCRV6_MEDIUM_CLASS_COUNT)
        cls.processor = LicensePlateProcessingMixin.__new__(LicensePlateProcessingMixin)

    def preprocess(self, crop, enhancement=0):
        from types import SimpleNamespace

        self.processor.config = SimpleNamespace(
            cameras={"a": SimpleNamespace(lpr=SimpleNamespace(enhancement=enhancement))}
        )
        self.processor.model_runner = SimpleNamespace(
            recognition_model=SimpleNamespace(
                runner=SimpleNamespace(get_input_width=lambda: "DynamicDimension.1")
            )
        )
        height, width = crop.shape[:2]
        # What _recognize computes for a single-image batch.
        max_wh_ratio = max(320 / 48, width / height)
        return self.processor._preprocess_recognition_image("a", crop, max_wh_ratio)

    def render(self, text):
        """A dark-on-light BGR plate crop, the way lpr_process hands them over."""
        crop = np.full((48, 24 * len(text) + 32, 3), 235, np.uint8)
        cv2.putText(
            crop,
            text,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        return crop

    def read(self, crop, enhancement=0):
        normalized = self.preprocess(crop, enhancement)
        outputs = self.session.run(None, {"x": normalized[np.newaxis, :]})[0]
        texts, confidences = self.decoder([outputs[0]])
        return texts[0], confidences[0]

    def test_a_rendered_plate_decodes_through_the_shipped_pipeline(self):
        for text in ("1ABC234", "XT79QZ", "7MPK432"):
            with self.subTest(text=text):
                decoded, confidences = self.read(self.render(text))
                self.assertEqual(decoded, text)
                self.assertEqual(len(confidences), len(text))
                self.assertGreater(min(confidences), 0.9)

    def test_the_enhanced_branch_still_decodes(self):
        """Turning the existing knob on must not break recognition."""
        decoded, _ = self.read(self.render("1ABC234"), enhancement=5)
        self.assertEqual(decoded, "1ABC234")

    def test_a_decoder_that_disagrees_with_the_model_recognizes_nothing(self):
        """A mis-sized map must yield no text, never a confident wrong plate."""
        normalized = self.preprocess(self.render("1ABC234"))
        outputs = self.session.run(None, {"x": normalized[np.newaxis, :]})[0]

        mismatched = CTCDecoder(CONFIG_PATH, PPOCRV6_MEDIUM_CLASS_COUNT)
        mismatched.characters = mismatched.characters[:-1]

        self.assertEqual(mismatched([outputs[0]]), ([], []))


if __name__ == "__main__":
    unittest.main()
