"""Weak same-inference boxes veto clearance without changing ordinary detection."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.config import DetectConfig, ModelConfig
from frigate.util.model import post_process_yolo
from frigate.util.object import reduce_detections
from frigate.video.detect import detect
from frigate.video.occupancy import OccupancyContinuity, intersects_polygon


class TestOccupancyCandidates(unittest.TestCase):
    def setUp(self):
        self.model = ModelConfig(
            labelmap_path=str(Path(__file__).resolve().parents[2] / "labelmap.txt")
        )

    def test_retained_model_outputs_keep_initial_clear_and_partial_roof_unknown(self):
        fixture = json.loads(
            (
                Path(__file__).parent
                / "fixtures/occupancy-candidate-model-results.json"
            ).read_text()
        )
        labels = {int(k): v for k, v in fixture["labelmap"].items()}
        polygon = np.array(fixture["polygon"], dtype=np.int32)
        filters = {
            label: SimpleNamespace(
                min_score=0.5,
                min_area=0,
                max_area=24000000,
                min_ratio=0,
                max_ratio=24000000,
                rasterized_mask=None,
            )
            for label in ("car", "motorcycle", "person")
        }
        owners = {}
        for row in fixture["frames"]:
            with self.subTest(group=row["group"], capture=row["captureAt"]):
                predictions = np.zeros(
                    (1, fixture["classes"] + 4, max(85, len(row["predictions"]))),
                    dtype=np.float32,
                )
                for index, (x, y, w, h, label, score) in enumerate(row["predictions"]):
                    predictions[0, :4, index] = [x, y, w, h]
                    predictions[0, label + 4, index] = score
                raw = post_process_yolo([predictions], 320, 320)
                self.assertEqual(raw.shape, (20, 6))
                detector = MagicMock()
                detector.detect.side_effect = lambda _, threshold, snapshot=raw: [
                    (labels[int(d[0])], float(d[1]), tuple(d[2:]))
                    for d in snapshot
                    if d[1] >= threshold
                ]
                candidates = []
                with patch("frigate.video.detect.create_tensor_input"):
                    ordinary = reduce_detections(
                        (576, 704),
                        detect(
                            DetectConfig(width=704, height=576),
                            detector,
                            None,
                            self.model,
                            tuple(fixture["crop"]),
                            list(filters),
                            filters,
                            candidates,
                        ),
                    )
                self.assertEqual(
                    [
                        {
                            "label": d[0],
                            "score": d[1],
                            "box": list(d[2]),
                            "overlap": intersects_polygon(d[2], polygon),
                        }
                        for d in ordinary
                    ],
                    row["ordinary"],
                )
                self.assertEqual(
                    [
                        {"label": d[0], "score": d[1], "box": list(d[2])}
                        for d in candidates
                        if intersects_polygon(d[2], polygon)
                    ],
                    row["overlapCandidates"],
                )
                owner = owners.setdefault(row["group"], OccupancyContinuity(5))
                # No initialized tracks or anchors: pixels are not consulted here.
                uncertain = owner.observe(
                    None, row["captureAt"], {"zone": polygon}, candidates, [], True
                )[0]["uncertain"]
                self.assertEqual(
                    uncertain, row["group"] in ("parked_roof", "current_roof")
                )
                self.assertEqual(owner.footprints, {})

    def predictions(self, rows):
        # cx, cy, width, height, then person / bicycle / car scores.
        result = np.zeros((1, 7, max(8, len(rows))), dtype=np.float32)
        for index, (box, label, score) in enumerate(rows):
            result[0, :4, index] = box
            result[0, 4 + label, index] = score
        return result

    def test_partial_roof_candidate_survives_without_becoming_an_ordinary_read(self):
        # Score and mapped footprint from the retained native-size roof JPEG.
        raw = self.predictions([((105, 163, 209, 63), 2, 0.20054244995117188)])
        result = post_process_yolo([raw], 320, 320)
        self.assertEqual(result.shape, (20, 6))
        self.assertAlmostEqual(result[0, 1], 0.20054244995117188)
        detector = MagicMock()
        detector.detect.return_value = [
            ("car", float(result[0, 1]), tuple(result[0, 2:]))
        ]
        candidates = []
        with patch("frigate.video.detect.create_tensor_input", return_value=raw):
            ordinary = detect(
                DetectConfig(width=704, height=576),
                detector,
                None,
                self.model,
                (0, 0, 948, 948),
                ["car"],
                {"car": SimpleNamespace(min_score=0.5)},
                candidates,
            )
        detector.detect.assert_called_once_with(raw, threshold=0.1)
        self.assertEqual(ordinary, [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "car")
        self.assertGreater(candidates[0][2][3], 524)

    def test_strict_ordinary_boundary_and_twenty_row_capacity_are_preserved(self):
        rows = [((10, 10, 4, 4), 2, 0.4)]
        rows += [((8 * 2**i, 20, 4, 4), 2, 0.9 - i / 100) for i in range(21)]
        rows += [((20 + i * 9, 90, 4, 4), 2, 0.2) for i in range(10)]
        output = post_process_yolo([self.predictions(rows)], 320, 320)
        self.assertEqual(output.shape, (20, 6))
        self.assertTrue(np.all(output[:, 1] > 0.4))
        self.assertTrue(np.all(output[:-1, 1] >= output[1:, 1]))
        boundary = post_process_yolo(
            [self.predictions([((10, 10, 4, 4), 2, 0.4)])], 320, 320
        )
        self.assertEqual(np.count_nonzero(boundary), 0)

    def test_candidates_respect_selected_classes_and_never_change_camera_filters(self):
        detector = MagicMock()
        detector.detect.return_value = [
            ("car", 0.9, (0.1, 0.1, 0.5, 0.5)),
            ("car", 0.45, (0.5, 0.5, 0.9, 0.9)),
            ("person", 0.2, (0.2, 0.2, 0.7, 0.7)),
            ("car", 0.2, (0.4, 0.4, 0.3, 0.3)),
        ]
        filters = {
            "car": SimpleNamespace(
                min_score=0.5,
                min_area=0,
                max_area=999999,
                min_ratio=0,
                max_ratio=999,
                rasterized_mask=None,
            )
        }
        candidates = []
        with patch("frigate.video.detect.create_tensor_input"):
            ordinary = detect(
                DetectConfig(width=320, height=320),
                detector,
                None,
                self.model,
                (0, 0, 320, 320),
                ["car"],
                filters,
                candidates,
            )
        self.assertEqual([d[1] for d in ordinary], [0.9])
        self.assertEqual([d[1] for d in candidates], [0.9, 0.45])

    def test_camera_without_occupancy_keeps_ordinary_detector_threshold(self):
        detector = MagicMock()
        detector.detect.return_value = []
        with patch("frigate.video.detect.create_tensor_input", return_value="frame"):
            self.assertEqual(
                detect(
                    DetectConfig(width=320, height=320),
                    detector,
                    None,
                    self.model,
                    (0, 0, 320, 320),
                    ["car"],
                    {},
                ),
                [],
            )
        detector.detect.assert_called_once_with("frame", threshold=0.4)
