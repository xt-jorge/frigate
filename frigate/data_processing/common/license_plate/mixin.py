"""Handle processing images for face detection and recognition."""

import base64
import datetime
import json
import logging
import math
import os
import random
import re
import string
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from pyclipper import ET_CLOSEDPOLYGON, JT_ROUND, PyclipperOffset
from rapidfuzz.distance import JaroWinkler, Levenshtein
from ruamel.yaml import YAML, YAMLError
from shapely.geometry import Polygon

from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.config.classification import LicensePlateRecognitionConfig
from frigate.const import CLIPS_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.license_plate.model import LicensePlateModelRunner
from frigate.embeddings.onnx.lpr_embedding import (
    LPR_EMBEDDING_SIZE,
    PPOCRV6_MEDIUM_CLASS_COUNT,
    PPOCRV6_MEDIUM_CONFIG_FILE,
)
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.image import area

from ...types import DataProcessorMetrics

logger = logging.getLogger(__name__)

WRITE_DEBUG_IMAGES = False


class LicensePlateProcessingMixin:
    # Attributes expected from consuming classes (set before super().__init__)
    config: FrigateConfig
    metrics: DataProcessorMetrics
    model_runner: LicensePlateModelRunner
    lpr_config: LicensePlateRecognitionConfig
    requestor: InterProcessRequestor
    detected_license_plates: dict[str, dict[str, Any]]
    camera_current_cars: dict[str, list[str]]
    sub_label_publisher: EventMetadataPublisher

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plate_rec_speed = InferenceSpeed(self.metrics.alpr_speed)
        self.plates_rec_second = EventsPerSecond()
        self.plates_rec_second.start()
        self.plate_det_speed = InferenceSpeed(self.metrics.yolov9_lpr_speed)
        self.plates_det_second = EventsPerSecond()
        self.plates_det_second.start()
        self.event_metadata_publisher = EventMetadataPublisher()
        self.ctc_decoder = CTCDecoder(
            character_dict_path=os.path.join(
                MODEL_CACHE_DIR, "paddleocr-onnx", PPOCRV6_MEDIUM_CONFIG_FILE
            ),
            expected_class_count=PPOCRV6_MEDIUM_CLASS_COUNT,
        )
        self.batch_size = 6

        # Object config
        self.lp_objects: list[str] = []

        for obj, attributes in self.config.model.attributes_map.items():
            if "license_plate" in attributes:
                self.lp_objects.append(obj)

        # Detection specific parameters
        self.min_size = 8
        self.max_size = 960
        self.box_thresh = 0.6
        self.mask_thresh = 0.6

        # matching
        self.similarity_threshold = 0.8
        self.cluster_threshold = 0.85

    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect possible areas of text in the input image by first resizing and normalizing it,
        running a detection model, and filtering out low-probability regions.

        Args:
            image (np.ndarray): The input image in which license plates will be detected.

        Returns:
            List[np.ndarray]: A list of bounding box coordinates representing detected license plates.
        """
        h, w = image.shape[:2]

        if sum([h, w]) < 64:
            image = self._zero_pad(image)

        resized_image = self._resize_image(image)
        normalized_image = self._normalize_image(resized_image)

        if WRITE_DEBUG_IMAGES:
            current_time = int(datetime.datetime.now().timestamp())
            cv2.imwrite(
                f"debug/frames/license_plate_resized_{current_time}.jpg",
                resized_image,
            )

        try:
            outputs = self.model_runner.detection_model([normalized_image])[0]  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Error running LPR box detection model: {e}")
            return []

        outputs = outputs[0, :, :]

        if False:
            current_time = int(datetime.datetime.now().timestamp())  # type: ignore[unreachable]
            cv2.imwrite(
                f"debug/frames/probability_map_{current_time}.jpg",
                (outputs * 255).astype(np.uint8),
            )

        boxes, _ = self._boxes_from_bitmap(outputs, outputs > self.mask_thresh, w, h)
        return self._filter_polygon(boxes, (h, w))  # type: ignore[return-value,arg-type]

    def _classify(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]] | None:
        """
        Classify the orientation or category of each detected license plate.

        Args:
            images (List[np.ndarray]): A list of images of detected license plates.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[str, float]]]: A tuple of rotated/normalized plate images
                                                            and classification results with confidence scores.
        """
        num_images = len(images)
        indices = np.argsort([x.shape[1] / x.shape[0] for x in images])

        for i in range(0, num_images, self.batch_size):
            norm_images = []
            for j in range(i, min(num_images, i + self.batch_size)):
                norm_img = self._preprocess_classification_image(images[indices[j]])
                norm_img = norm_img[np.newaxis, :]
                norm_images.append(norm_img)

        try:
            outputs = self.model_runner.classification_model(norm_images)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Error running LPR classification model: {e}")
            return None

        return self._process_classification_output(images, outputs)

    def _recognize(
        self, camera: str, images: List[np.ndarray]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Recognize the characters on the detected license plates using the recognition model.

        Args:
            images (List[np.ndarray]): A list of images of license plates to recognize.

        Returns:
            Tuple[List[str], List[List[float]]]: A tuple of recognized license plate texts and confidence scores.
        """
        input_shape = [3, 48, 320]
        num_images = len(images)

        for index in range(0, num_images, self.batch_size):
            input_h, input_w = input_shape[1], input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []

            # calculate the maximum aspect ratio in the current batch
            for i in range(index, min(num_images, index + self.batch_size)):
                h, w = images[i].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)

            # preprocess the images based on the max aspect ratio
            for i in range(index, min(num_images, index + self.batch_size)):
                norm_image = self._preprocess_recognition_image(
                    camera, images[i], max_wh_ratio
                )
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)

        try:
            outputs = self.model_runner.recognition_model(norm_images)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Error running LPR recognition model: {e}")
            return [], []

        return self.ctc_decoder(outputs)

    def _process_license_plate(
        self, camera: str, id: str, image: np.ndarray
    ) -> Tuple[List[str], List[List[float]], List[int]]:
        """
        Complete pipeline for detecting, classifying, and recognizing license plates in the input image.
        Combines multi-line plates into a single plate string, grouping boxes by vertical alignment and ordering top to bottom,
        but only combines boxes if their average confidence scores meet the threshold and their heights are similar.

        Args:
            camera (str): Camera identifier.
            id (str): Event identifier.
            image (np.ndarray): The input image in which to detect, classify, and recognize license plates.

        Returns:
            Tuple[List[str], List[List[float]], List[int]]: Detected license plate texts, character-level confidence scores for each plate (flattened into a single list per plate), and areas of the plates.
        """
        if (
            self.model_runner.detection_model.runner is None
            or self.model_runner.classification_model.runner is None
            or self.model_runner.recognition_model.runner is None
        ):
            # we might still be downloading the models
            logger.debug("Model runners not loaded")
            return [], [], []

        boxes = self._detect(image)
        if len(boxes) == 0:
            logger.debug(f"{camera}: No boxes found by OCR detector model")
            return [], [], []

        if len(boxes) > 0:
            plate_left = np.min([np.min(box[:, 0]) for box in boxes])
            plate_right = np.max([np.max(box[:, 0]) for box in boxes])
            plate_width = plate_right - plate_left
        else:
            plate_width = 0

        boxes = self._merge_nearby_boxes(
            boxes, plate_width=plate_width, gap_fraction=0.1
        )

        current_time = int(datetime.datetime.now().timestamp())
        if WRITE_DEBUG_IMAGES:
            debug_image = image.copy()
            for box in boxes:
                box = box.astype(int)
                x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
                x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
                cv2.rectangle(
                    debug_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(0, 255, 0),
                    thickness=2,
                )

            cv2.imwrite(
                f"debug/frames/license_plate_boxes_{current_time}.jpg", debug_image
            )

        boxes = self._sort_boxes(list(boxes))

        # Step 1: Compute box heights and group boxes by vertical alignment and height similarity
        box_info = []
        for i, box in enumerate(boxes):
            y_coords = box[:, 1]
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            height = y_max - y_min
            box_info.append((y_min, y_max, height, i))

        # Initial grouping based on y-coordinate overlap and height similarity
        initial_groups = []
        current_group = [box_info[0]]
        height_tolerance = 0.25  # Allow 25% difference in height for grouping

        for i in range(1, len(box_info)):
            prev_y_min, prev_y_max, prev_height, _ = current_group[-1]
            curr_y_min, _, curr_height, _ = box_info[i]

            # Check y-coordinate overlap
            overlap_threshold = 0.1 * (prev_y_max - prev_y_min)
            overlaps = curr_y_min <= prev_y_max + overlap_threshold

            # Check height similarity
            height_ratio = min(prev_height, curr_height) / max(prev_height, curr_height)
            height_similar = height_ratio >= (1 - height_tolerance)

            if overlaps and height_similar:
                current_group.append(box_info[i])
            else:
                initial_groups.append(current_group)
                current_group = [box_info[i]]
        initial_groups.append(current_group)

        # Step 2: Process each initial group, filter by confidence
        all_license_plates = []
        all_confidences = []
        all_areas = []
        processed_indices = set()

        recognition_threshold = self.lpr_config.recognition_threshold

        for group in initial_groups:
            # Sort group by y-coordinate (top to bottom)
            group.sort(key=lambda x: x[0])
            group_indices = [item[3] for item in group]

            # Skip if all indices in this group have already been processed
            if all(idx in processed_indices for idx in group_indices):
                continue

            # Crop images for the group
            group_boxes = [boxes[i] for i in group_indices]
            group_plate_images = [
                self._crop_license_plate(image, box) for box in group_boxes
            ]

            if WRITE_DEBUG_IMAGES:
                for i, img in enumerate(group_plate_images):
                    cv2.imwrite(
                        f"debug/frames/license_plate_cropped_{current_time}_{group_indices[i] + 1}.jpg",
                        img,
                    )

            if self.config.lpr.debug_save_plates:
                logger.debug(f"{camera}: Saving plates for event {id}")
                Path(os.path.join(CLIPS_DIR, f"lpr/{camera}/{id}")).mkdir(
                    parents=True, exist_ok=True
                )
                for i, img in enumerate(group_plate_images):
                    cv2.imwrite(
                        os.path.join(
                            CLIPS_DIR,
                            f"lpr/{camera}/{id}/{current_time}_{group_indices[i] + 1}.jpg",
                        ),
                        img,
                    )

            # Recognize text in each cropped image
            results, confidences = self._recognize(camera, group_plate_images)

            if not results:
                continue

            if not confidences:
                confidences = [[0.0] for _ in results]

            # Compute average confidence for each box's recognized text
            avg_confidences = []
            for conf_list in confidences:
                avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
                avg_confidences.append(avg_conf)

            # Filter boxes based on the recognition threshold
            qualifying_indices = []
            qualifying_results = []
            qualifying_confidences = []
            for i, (avg_conf, result, conf_list) in enumerate(
                zip(avg_confidences, results, confidences)
            ):
                if avg_conf >= recognition_threshold:
                    qualifying_indices.append(group_indices[i])
                    qualifying_results.append(result)
                    qualifying_confidences.append(conf_list)

            if not qualifying_results:
                continue

            processed_indices.update(qualifying_indices)

            # Combine the qualifying results into a single plate string
            combined_plate = " ".join(qualifying_results)

            flat_confidences = [
                conf for conf_list in qualifying_confidences for conf in conf_list
            ]

            # Apply replace rules to combined_plate if configured
            original_combined = combined_plate
            if self.lpr_config.replace_rules:
                for rule in self.lpr_config.replace_rules:
                    try:
                        pattern = getattr(rule, "pattern", "")
                        replacement = getattr(rule, "replacement", "")
                        if pattern:
                            combined_plate = re.sub(
                                pattern, replacement, combined_plate
                            )
                            logger.debug(
                                f"{camera}: Processing replace rule: '{pattern}' -> '{replacement}', result: '{combined_plate}'"
                            )
                    except re.error as e:
                        logger.warning(
                            f"{camera}: Invalid regex in replace_rules '{pattern}': {e}"
                        )

            if combined_plate != original_combined:
                logger.debug(
                    f"{camera}: All rules applied: '{original_combined}' -> '{combined_plate}'"
                )

            # Compute the combined area for qualifying boxes
            qualifying_boxes = [boxes[i] for i in qualifying_indices]
            qualifying_plate_images = [
                self._crop_license_plate(image, box) for box in qualifying_boxes
            ]
            group_areas = [
                img.shape[0] * img.shape[1] for img in qualifying_plate_images
            ]
            combined_area = sum(group_areas)

            all_license_plates.append(combined_plate)
            all_confidences.append(flat_confidences)
            all_areas.append(combined_area)

        # Step 3: Sort the combined plates
        if all_license_plates:
            sorted_data = sorted(
                zip(all_license_plates, all_confidences, all_areas),
                key=lambda x: (x[2], len(x[0]), sum(x[1]) / len(x[1]) if x[1] else 0),
                reverse=True,
            )

            if sorted_data:
                plates, confs, areas_list = zip(*sorted_data)
                return list(plates), list(confs), list(areas_list)

        return [], [], []

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the input image while maintaining the aspect ratio, ensuring dimensions are multiples of 32.

        Args:
            image (np.ndarray): The input image to resize.

        Returns:
            np.ndarray: The resized image.
        """
        h, w = image.shape[:2]
        ratio = min(self.max_size / max(h, w), 1.0)
        resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)
        return cv2.resize(image, (resize_w, resize_h))

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the input image by subtracting the mean and multiplying by the standard deviation.

        Args:
            image (np.ndarray): The input image to normalize.

        Returns:
            np.ndarray: The normalized image, transposed to match the model's expected input format.
        """
        mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1).astype("float64")
        std = 1 / np.array([58.395, 57.12, 57.375]).reshape(1, -1).astype("float64")

        image = image.astype("float32")
        cv2.subtract(image, mean, image)
        cv2.multiply(image, std, image)
        return image.transpose((2, 0, 1))[np.newaxis, ...]

    def _merge_nearby_boxes(
        self,
        boxes: List[np.ndarray],
        plate_width: float,
        gap_fraction: float = 0.1,
        min_overlap_fraction: float = -0.2,
    ) -> List[np.ndarray]:
        """
        Merge bounding boxes that are likely part of the same license plate based on proximity,
        with a dynamic max_gap based on the provided width of the entire license plate.

        Args:
            boxes (List[np.ndarray]): List of bounding boxes with shape (n, 4, 2), where n is the number of boxes,
                                    each box has 4 corners, and each corner has (x, y) coordinates.
            plate_width (float): The width of the entire license plate in pixels, used to calculate max_gap.
            gap_fraction (float): Fraction of the plate width to use as the maximum gap.
                                Default is 0.1 (10% of the plate width).

        Returns:
            List[np.ndarray]: List of merged bounding boxes.
        """
        if len(boxes) == 0:
            return []

        max_gap = plate_width * gap_fraction
        min_overlap = plate_width * min_overlap_fraction

        # Sort boxes by top left x
        sorted_boxes = sorted(boxes, key=lambda x: x[0][0])

        merged_boxes = []
        current_box = sorted_boxes[0]

        for i in range(1, len(sorted_boxes)):
            next_box = sorted_boxes[i]

            # Calculate the horizontal gap between the current box and the next box
            current_right = np.max(
                current_box[:, 0]
            )  # Rightmost x-coordinate of current box
            next_left = np.min(next_box[:, 0])  # Leftmost x-coordinate of next box
            horizontal_gap = next_left - current_right

            # Check if the boxes are vertically aligned (similar y-coordinates)
            current_top = np.min(current_box[:, 1])
            current_bottom = np.max(current_box[:, 1])
            next_top = np.min(next_box[:, 1])
            next_bottom = np.max(next_box[:, 1])

            # Consider boxes part of the same plate if they are close horizontally or overlap
            # within the allowed limit and their vertical positions overlap significantly
            if min_overlap <= horizontal_gap <= max_gap and max(
                current_top, next_top
            ) <= min(current_bottom, next_bottom):
                merged_points = np.vstack((current_box, next_box))
                new_box = np.array(
                    [
                        [
                            np.min(merged_points[:, 0]),
                            np.min(merged_points[:, 1]),
                        ],
                        [
                            np.max(merged_points[:, 0]),
                            np.min(merged_points[:, 1]),
                        ],
                        [
                            np.max(merged_points[:, 0]),
                            np.max(merged_points[:, 1]),
                        ],
                        [
                            np.min(merged_points[:, 0]),
                            np.max(merged_points[:, 1]),
                        ],
                    ]
                )
                current_box = new_box
            else:
                # If the boxes are not close enough or overlap too much, add the current box to the result
                merged_boxes.append(current_box)
                current_box = next_box

        # Add the last box
        merged_boxes.append(current_box)

        return np.array(merged_boxes, dtype=np.int32)  # type: ignore[return-value]

    def _boxes_from_bitmap(
        self, output: np.ndarray, mask: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Process the binary mask to extract bounding boxes and associated confidence scores.

        Args:
            output (np.ndarray): Output confidence map from the model.
            mask (np.ndarray): Binary mask of detected regions.
            dest_width (int): Target width for scaling the box coordinates.
            dest_height (int): Target height for scaling the box coordinates.

        Returns:
            Tuple[np.ndarray, List[float]]: Array of bounding boxes and list of corresponding scores.
        """

        mask = (mask * 255).astype(np.uint8)
        height, width = mask.shape
        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # handle different return values of findContours between OpenCV versions
        contours = outs[0] if len(outs) == 2 else outs[1]

        boxes = []
        scores = []

        for index in range(len(contours)):  # type: ignore[arg-type]
            contour = contours[index]  # type: ignore[index]

            # get minimum bounding box (rotated rectangle) around the contour and the smallest side length.
            points, sside = self._get_min_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points, dtype=np.float32)  # type: ignore[assignment]

            score = self._box_score(output, contour)
            if self.box_thresh > score:
                continue

            points = self._expand_box(points)  # type: ignore[assignment]

            # Get the minimum area rectangle again after expansion
            points, sside = self._get_min_boxes(points.reshape(-1, 1, 2))  # type: ignore[attr-defined]
            if sside < self.min_size + 2:
                continue

            points = np.array(points, dtype=np.float32)  # type: ignore[assignment]

            # normalize and clip box coordinates to fit within the destination image size.
            points[:, 0] = np.clip(  # type: ignore[call-overload]
                np.round(points[:, 0] / width * dest_width),  # type: ignore[call-overload]
                0,
                dest_width,
            )
            points[:, 1] = np.clip(  # type: ignore[call-overload]
                np.round(points[:, 1] / height * dest_height),  # type: ignore[call-overload]
                0,
                dest_height,
            )

            boxes.append(points.astype("int32"))  # type: ignore[attr-defined]
            scores.append(score)

        return np.array(boxes, dtype="int32"), scores

    @staticmethod
    def _get_min_boxes(contour: np.ndarray) -> Tuple[List[Tuple[float, float]], float]:
        """
        Calculate the minimum bounding box (rotated rectangle) for a given contour.

        Args:
            contour (np.ndarray): The contour points of the detected shape.

        Returns:
            Tuple[List[Tuple[float, float]], float]: A list of four points representing the
            corners of the bounding box, and the length of the shortest side.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])
        index_1, index_4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
        index_2, index_3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score(bitmap: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate the average score within the bounding box of a contour.

        Args:
            bitmap (np.ndarray): The output confidence map from the model.
            contour (np.ndarray): The contour of the detected shape.

        Returns:
            float: The average score of the pixels inside the contour region.
        """
        h, w = bitmap.shape[:2]
        contour = contour.reshape(-1, 2)
        x1, y1 = np.clip(contour.min(axis=0), 0, [w - 1, h - 1])
        x2, y2 = np.clip(contour.max(axis=0), 0, [w - 1, h - 1])
        mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [contour - [x1, y1]], 1)  # type: ignore[call-overload]
        return cv2.mean(bitmap[y1 : y2 + 1, x1 : x2 + 1], mask)[0]

    @staticmethod
    def _expand_box(points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Expand a polygonal shape slightly by a factor determined by the area-to-perimeter ratio.

        Args:
            points (List[Tuple[float, float]]): Points of the polygon to expand.

        Returns:
            np.ndarray: Expanded polygon points.
        """
        polygon = Polygon(points)
        distance = polygon.area / polygon.length
        offset = PyclipperOffset()
        offset.AddPath(points, JT_ROUND, ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance * 1.5)).reshape((-1, 2))
        return expanded

    def _filter_polygon(
        self, points: List[np.ndarray], shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Filter a set of polygons to include only valid ones that fit within an image shape
        and meet size constraints.

        Args:
            points (List[np.ndarray]): List of polygons to filter.
            shape (Tuple[int, int]): Shape of the image (height, width).

        Returns:
            np.ndarray: List of filtered polygons.
        """
        height, width = shape
        return np.array(
            [
                self._clockwise_order(point)
                for point in points
                if self._is_valid_polygon(point, width, height)
            ]
        )

    @staticmethod
    def _is_valid_polygon(point: np.ndarray, width: int, height: int) -> bool:
        """
        Check if a polygon is valid, meaning it fits within the image bounds
        and has sides of a minimum length.

        Args:
            point (np.ndarray): The polygon to validate.
            width (int): Image width.
            height (int): Image height.

        Returns:
            bool: Whether the polygon is valid or not.
        """
        return bool(
            point[:, 0].min() >= 0
            and point[:, 0].max() < width
            and point[:, 1].min() >= 0
            and point[:, 1].max() < height
            and np.linalg.norm(point[0] - point[1]) > 3
            and np.linalg.norm(point[0] - point[3]) > 3
        )

    @staticmethod
    def _clockwise_order(pts: np.ndarray) -> np.ndarray:
        """
        Arrange the points of a polygon in order: top-left, top-right, bottom-right, bottom-left.
        taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py

        Args:
            pts (np.ndarray): Array of points of the polygon.

        Returns:
            np.ndarray: Points ordered clockwise starting from top-left.
        """
        # Sort the points based on their x-coordinates
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        # Separate the left-most and right-most points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        # Sort the left-most coordinates by y-coordinates
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most  # Top-left and bottom-left

        # Use the top-left as an anchor to calculate distances to right points
        # The further point will be the bottom-right
        distances = np.sqrt(
            ((tl[0] - right_most[:, 0]) ** 2) + ((tl[1] - right_most[:, 1]) ** 2)
        )

        # Sort right points by distance (descending)
        right_idx = np.argsort(distances)[::-1]
        (br, tr) = right_most[right_idx, :]  # Bottom-right and top-right

        return np.array([tl, tr, br, bl])

    @staticmethod
    def _sort_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
        """
        Sort polygons based on their position in the image. If boxes are close in vertical
        position (within 5 pixels), sort them by horizontal position.

        Args:
            points: detected text boxes with shape [4, 2]

        Returns:
            List: sorted boxes(array) with shape [4, 2]
        """
        boxes.sort(key=lambda x: (x[0][1], x[0][0]))
        for i in range(len(boxes) - 1):
            for j in range(i, -1, -1):
                if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 5 and (
                    boxes[j + 1][0][0] < boxes[j][0][0]
                ):
                    temp = boxes[j]
                    boxes[j] = boxes[j + 1]
                    boxes[j + 1] = temp
                else:
                    break
        return boxes

    @staticmethod
    def _zero_pad(image: np.ndarray) -> np.ndarray:
        """
        Apply zero-padding to an image, ensuring its dimensions are at least 32x32.
        The padding is added only if needed.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Zero-padded image.
        """
        h, w, c = image.shape
        pad = np.zeros((max(32, h), max(32, w), c), np.uint8)
        pad[:h, :w, :] = image
        return pad

    @staticmethod
    def _preprocess_classification_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for classification by resizing, normalizing, and padding.

        This method resizes the input image to a fixed height of 48 pixels while adjusting
        the width dynamically up to a maximum of 192 pixels. The image is then normalized and
        padded to fit the required input dimensions for classification.

        Args:
            image (np.ndarray): Input image to preprocess.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        # fixed height of 48, dynamic width up to 192
        input_shape = (3, 48, 192)
        input_c, input_h, input_w = input_shape

        h, w = image.shape[:2]
        ratio = w / h
        resized_w = min(input_w, math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))

        # handle single-channel images (grayscale) if needed
        if input_c == 1 and resized_image.ndim == 2:
            resized_image = resized_image[np.newaxis, :, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1))

        # normalize
        resized_image = (resized_image.astype("float32") / 255.0 - 0.5) / 0.5

        padded_image = np.zeros((input_c, input_h, input_w), dtype=np.float32)
        padded_image[:, :, :resized_w] = resized_image

        return padded_image

    def _process_classification_output(
        self, images: List[np.ndarray], outputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """
        Process the classification model output by matching labels with confidence scores.

        This method processes the outputs from the classification model and rotates images
        with high confidence of being labeled "180". It ensures that results are mapped to
        the original image order.

        Args:
            images (List[np.ndarray]): List of input images.
            outputs (List[np.ndarray]): Corresponding model outputs.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[str, float]]]: A tuple of processed images and
            classification results (label and confidence score).
        """
        labels = ["0", "180"]
        results = [["", 0.0]] * len(images)
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        stacked_outputs = np.stack(outputs)

        stacked_outputs = [
            (labels[idx], stacked_outputs[i, idx])
            for i, idx in enumerate(stacked_outputs.argmax(axis=1))
        ]

        for i in range(0, len(images), self.batch_size):
            for j in range(len(stacked_outputs)):
                label, score = stacked_outputs[j]
                results[indices[i + j]] = [label, score]
                # make sure we have high confidence if we need to flip a box
                if "180" in label and score >= 0.7:
                    images[indices[i + j]] = cv2.rotate(
                        images[indices[i + j]], cv2.ROTATE_180
                    )

        return images, results  # type: ignore[return-value]

    def _preprocess_recognition_image(
        self, camera: str, image: np.ndarray, max_wh_ratio: float
    ) -> np.ndarray:
        """
        Preprocess an image for recognition by dynamically adjusting its width.

        This method adjusts the width of the image based on the maximum width-to-height ratio
        while keeping the height fixed at 48 pixels. The image is then normalized and padded
        to fit the required input dimensions for recognition.

        Args:
            image (np.ndarray): Input BGR plate crop to preprocess.
            max_wh_ratio (float): Maximum width-to-height ratio for resizing.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        # fixed height of 48, dynamic width based on ratio
        input_shape = [3, 48, 320]
        input_h, input_w = input_shape[1], input_shape[2]

        assert image.shape[2] == input_shape[0], "Unexpected number of image channels."

        enhancement = self.config.cameras[camera].lpr.enhancement

        if enhancement > 0:
            # The denoise/CLAHE operators below are single-channel, so this
            # branch pays a colour round trip to get them. It applies only when
            # an operator has asked for enhancement.
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if enhancement > 3:
                # denoise using a configurable pixel neighborhood value
                logger.debug(
                    f"{camera}: Denoising recognition image (level: {enhancement})"
                )
                smoothed = cv2.bilateralFilter(
                    processed,
                    d=5 + enhancement,
                    sigmaColor=10 * enhancement,
                    sigmaSpace=10 * enhancement,
                )
                sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                processed = cv2.filter2D(smoothed, -1, sharpening_kernel)

            logger.debug(
                f"{camera}: Enhancing contrast for recognition image (level: {enhancement})"
            )
            grid_size = (
                max(4, input_w // 40),
                max(4, input_h // 40),
            )
            clahe = cv2.createCLAHE(
                clipLimit=2 if enhancement > 5 else 1.5,
                tileGridSize=grid_size,
            )
            image = cv2.cvtColor(clahe.apply(processed), cv2.COLOR_GRAY2BGR)

        # With no enhancement requested the crop reaches the recognizer as the
        # BGR data `lpr_process` decoded, which is the colour input PaddleOCR
        # text recognition is trained and exported for.

        # dynamically adjust input width based on max_wh_ratio
        input_w = int(input_h * max_wh_ratio)

        # check for model-specific input width
        model_input_w = self.model_runner.recognition_model.runner.get_input_width()  # type: ignore[union-attr]
        if isinstance(model_input_w, int) and model_input_w > 0:
            input_w = model_input_w

        h, w = image.shape[:2]
        aspect_ratio = w / h
        resized_w = min(input_w, math.ceil(input_h * aspect_ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = (resized_image.astype("float32") / 255.0 - 0.5) / 0.5

        # Official PaddleOCR RecResizeImg pads with zeros, which after the
        # (x/255 - 0.5)/0.5 normalization above is mid-gray in pixel space.
        # Padding with the crop's mean instead feeds the recognizer a trailing
        # block the model never saw in training.
        padded_image = np.zeros((input_shape[0], input_h, input_w), dtype=np.float32)
        padded_image[:, :, :resized_w] = resized_image

        if False:
            current_time = int(datetime.datetime.now().timestamp() * 1000)  # type: ignore[unreachable]
            cv2.imwrite(
                f"debug/frames/preprocessed_recognition_{current_time}.jpg",
                image,
            )

        return padded_image

    @staticmethod
    def _crop_license_plate(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Crop the license plate from the image using four corner points.

        This method crops the region containing the license plate by using the perspective
        transformation based on four corner points. If the resulting image is significantly
        taller than wide, the image is rotated to the correct orientation.

        Args:
            image (np.ndarray): Input image containing the license plate.
            points (np.ndarray): Four corner points defining the plate's position.

        Returns:
            np.ndarray: Cropped and potentially rotated license plate image.
        """
        assert len(points) == 4, "shape of points must be 4*2"
        points = points.astype(np.float32)
        crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.array(
            [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, pts_std)
        image = cv2.warpPerspective(
            image,
            matrix,
            (crop_width, crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        height, width = image.shape[0:2]
        if height * 1.0 / width >= 1.5:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    def _detect_license_plate(
        self, camera: str, input: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """
        Use a lightweight YOLOv9 model to detect license plates for users without Frigate+

        Return the dimensions of the detected plate as [x1, y1, x2, y2].
        """
        try:
            predictions = self.model_runner.yolov9_detection_model(input)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Error running YOLOv9 license plate detection model: {e}")
            return None

        confidence_threshold = self.lpr_config.detection_threshold

        top_score = -1
        top_box = None

        img_h, img_w = input.shape[0], input.shape[1]

        # Calculate resized dimensions and padding based on _preprocess_inputs
        if img_w > img_h:
            resized_h = int(((img_h / img_w) * LPR_EMBEDDING_SIZE) // 4 * 4)
            resized_w = LPR_EMBEDDING_SIZE
            x_offset = (LPR_EMBEDDING_SIZE - resized_w) // 2
            y_offset = (LPR_EMBEDDING_SIZE - resized_h) // 2
            scale_x = img_w / resized_w
            scale_y = img_h / resized_h
        else:
            resized_w = int(((img_w / img_h) * LPR_EMBEDDING_SIZE) // 4 * 4)
            resized_h = LPR_EMBEDDING_SIZE
            x_offset = (LPR_EMBEDDING_SIZE - resized_w) // 2
            y_offset = (LPR_EMBEDDING_SIZE - resized_h) // 2
            scale_x = img_w / resized_w
            scale_y = img_h / resized_h

        # Loop over predictions
        for prediction in predictions:
            score = prediction[6]
            if score >= confidence_threshold:
                bbox = prediction[1:5]
                # Adjust for padding and scale to original image
                bbox[0] = (bbox[0] - x_offset) * scale_x
                bbox[1] = (bbox[1] - y_offset) * scale_y
                bbox[2] = (bbox[2] - x_offset) * scale_x
                bbox[3] = (bbox[3] - y_offset) * scale_y

                if score > top_score:
                    top_score = score
                    top_box = bbox

        # Return the top scoring bounding box if found
        if top_box is not None:
            # expand box by 5% to help with OCR
            expansion = (top_box[2:] - top_box[:2]) * 0.05

            # Expand box
            expanded_box = np.array(
                [
                    top_box[0] - expansion[0],  # x1
                    top_box[1] - expansion[1],  # y1
                    top_box[2] + expansion[0],  # x2
                    top_box[3] + expansion[1],  # y2
                ]
            ).clip(0, [input.shape[1], input.shape[0]] * 2)

            return tuple(int(x) for x in expanded_box)  # type: ignore[return-value]
        else:
            return None  # No detection above the threshold

    def _get_cluster_rep(
        self, plates: List[dict]
    ) -> Tuple[str, float, List[float], int]:
        """
        Cluster plate variants and select the representative from the best cluster.
        """
        if len(plates) == 0:
            return "", 0.0, [], 0

        if len(plates) == 1:
            p = plates[0]
            return p["plate"], p["conf"], p["char_confidences"], p["area"]

        # Log initial variants
        logger.debug(f"Clustering {len(plates)} plate variants:")
        for i, p in enumerate(plates):
            logger.debug(
                f"  Variant {i + 1}: '{p['plate']}' (conf: {p['conf']:.3f}, area: {p['area']})"
            )

        clusters: list[list[dict[str, Any]]] = []
        for i, plate in enumerate(plates):
            merged = False
            for j, cluster in enumerate(clusters):
                sims = [
                    JaroWinkler.similarity(plate["plate"], v["plate"]) for v in cluster
                ]
                if len(sims) > 0:
                    avg_sim = sum(sims) / len(sims)
                    if avg_sim >= self.cluster_threshold:
                        cluster.append(plate)
                        logger.debug(
                            f"  Merged variant {i + 1} '{plate['plate']}' (conf: {plate['conf']:.3f}) into cluster {j + 1} (avg_sim: {avg_sim:.3f})"
                        )
                        merged = True
                        break
            if not merged:
                clusters.append([plate])
                logger.debug(
                    f"  Started new cluster {len(clusters)} with variant {i + 1} '{plate['plate']}' (conf: {plate['conf']:.3f})"
                )

        if not clusters:
            return "", 0.0, [], 0

        # Log cluster summaries
        for j, cluster in enumerate(clusters):
            cluster_size = len(cluster)
            max_conf = max(v["conf"] for v in cluster)
            sample_variants = [v["plate"] for v in cluster[:3]]  # First 3 for brevity
            logger.debug(
                f"  Cluster {j + 1}: size {cluster_size}, max conf {max_conf:.3f}, variants: {sample_variants}{'...' if cluster_size > 3 else ''}"
            )

        # Best cluster: largest size, tiebroken by max conf
        def cluster_score(c: list[dict[str, Any]]) -> tuple[int, float]:
            return (len(c), max(v["conf"] for v in c))

        best_cluster_idx = max(
            range(len(clusters)), key=lambda j: cluster_score(clusters[j])
        )
        best_cluster = clusters[best_cluster_idx]
        best_size, best_max_conf = cluster_score(best_cluster)
        logger.debug(
            f"  Selected best cluster {best_cluster_idx + 1}: size {best_size}, max conf {best_max_conf:.3f}"
        )

        # Rep: highest conf in best cluster
        rep = max(best_cluster, key=lambda v: v["conf"])
        logger.debug(
            f"  Selected rep from best cluster: '{rep['plate']}' (conf: {rep['conf']:.3f})"
        )
        logger.debug(
            f"  Final clustered plate: '{rep['plate']}' (conf: {rep['conf']:.3f})"
        )

        return rep["plate"], rep["conf"], rep["char_confidences"], rep["area"]

    def _passes_plate_filters(self, camera: str, plate: str, kind: str) -> bool:
        """Check a plate string against the configured length and format filters.

        Args:
            camera: Camera the reading came from, for logging.
            plate: The text to check.
            kind: What the text is ("clustered" aggregate or "recognized"
                current sample), for logging.

        Returns:
            Whether the text may be used for its purpose. An invalid `format`
            regex is a configuration error, not a rejection, so it is logged and
            the text passes - matching the previous behaviour.
        """
        if len(plate) < self.lpr_config.min_plate_length:
            logger.debug(
                f"{camera}: Filtered out {kind} plate '{plate}' due to length ({len(plate)} < {self.lpr_config.min_plate_length})"
            )
            return False

        if self.lpr_config.format:
            try:
                if not re.fullmatch(self.lpr_config.format, plate):
                    logger.debug(
                        f"{camera}: Filtered out {kind} plate '{plate}' due to format mismatch"
                    )
                    return False
            except re.error:
                logger.error(
                    f"{camera}: Invalid regex in LPR format configuration: {self.lpr_config.format}"
                )

        return True

    def _current_plate_attribute(
        self, camera: str, obj_data: dict[str, Any], source_frame_time: float | None
    ) -> dict[str, Any] | None:
        """Return the best license plate region the model measured on this frame.

        Args:
            camera: Camera the track belongs to, for logging.
            obj_data: The tracked object, carrying its `current_attributes`.
            source_frame_time: Clock of the frame the crop will come from, or
                None when the caller has re-projected the regions onto an image
                no detector ran on and vouches for the correspondence itself.

        Returns:
            The highest scoring `license_plate` attribute whose own detector
            observation clock is exactly this frame's, or None.

        A track carries its attributes across frames, so `current_attributes`
        on its own says only that this track has a plate somewhere - not that
        the plate is in these pixels. Pairing a region measured on an earlier
        frame with a newer crop reads whatever has since moved into that box.
        """
        best: dict[str, Any] | None = None

        for attr in obj_data.get("current_attributes") or []:
            if attr.get("label") != "license_plate":
                continue

            if (
                source_frame_time is not None
                and attr.get("detector_observed_at") != source_frame_time
            ):
                logger.debug(
                    f"{camera}: Ignoring license plate region observed at {attr.get('detector_observed_at')}, not {source_frame_time}"
                )
                continue

            if best is None or attr.get("score", 0.0) > best.get("score", 0.0):
                best = attr

        return best

    @staticmethod
    def _vehicle_measured_on_frame(
        obj_data: dict[str, Any], source_frame_time: float
    ) -> bool:
        """Whether the model measured this track's own box on exactly this frame.

        The tracker advances a track's `frame_time` onto every frame it is
        assumed to still be present on, while deliberately preserving the older
        `detector_observed_at` of the measurement the box actually came from
        (`NorfairTracker.update_frame_times`). So `frame_time` says "believed
        present", and only `detector_observed_at` says "measured here".

        Compared raw, before any rounding: two different measurements can round
        to the same millisecond, and a missing or non-numeric clock is refused
        rather than treated as current.
        """
        observed = obj_data.get("detector_observed_at")

        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and observed == source_frame_time
        )

    def _usable_plate_box(
        self, camera: str, region: dict[str, Any] | None
    ) -> Any | None:
        """Return this region's box when it is large enough to be worth reading.

        The one place `min_area` is applied to a reported region, so the
        scheduler's view of what a frame can yield and the processor's decision
        about what to read cannot drift apart.
        """
        box = region.get("box") if region is not None else None

        if not box:
            return None

        if area(box) < self.config.cameras[camera].lpr.min_area:
            logger.debug(
                f"{camera}: Area for license plate box {area(box)} is less than min_area {self.config.cameras[camera].lpr.min_area}"
            )
            return None

        return box

    def plate_region_available(
        self, camera: str, obj_data: dict[str, Any], source_frame_time: float
    ) -> bool:
        """Whether this track can yield a plate region from this exact frame.

        Args:
            camera: Camera the track belongs to.
            obj_data: The tracked object being considered.
            source_frame_time: Clock of the frame that would be sampled.

        Returns:
            Whether an attempt on this frame could produce a region whose
            geometry the model measured on this frame.

        The scheduler asks this before it selects a candidate and spends the
        one OCR slot, and `lpr_process` asks it again before it acts. Keeping
        both on this single rule is what stops a track from repeatedly
        consuming the budget on frames it can never be read from - the periodic
        fresh-parent frame would otherwise be missed on phase alignment alone.
        """
        # A dedicated Frigate+ LPR camera tracks the plate itself, and a camera
        # with no plate-detecting model has only ever had the secondary locator.
        # Both are pre-existing paths whose geometry handling is unchanged here,
        # and on both the processor acts on exactly what it is given, so there
        # is nothing here for the scheduler's view to disagree with.
        if obj_data.get("label") == "license_plate":
            return True

        if "license_plate" not in self.config.cameras[camera].objects.track:
            return True

        # Plus vehicle: either a plate region the processor would actually read
        # from this frame, or a vehicle box measured on this frame to look
        # inside. A region too small to read is not one the processor would use,
        # so counting it here would spend the slot and then discard it - and a
        # carried box would then attach whatever plate is now in those pixels to
        # this track on geometry the model never confirmed for this image.
        return self._usable_plate_box(
            camera, self._current_plate_attribute(camera, obj_data, source_frame_time)
        ) is not None or self._vehicle_measured_on_frame(obj_data, source_frame_time)

    def _locate_plate_in_vehicle(
        self,
        camera: str,
        obj_data: dict[str, Any],
        frame: np.ndarray,
        current_time: float,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Look for a plate inside this frame's crop of this vehicle.

        Args:
            camera: Camera the track belongs to.
            obj_data: The tracked vehicle whose box selects the crop.
            frame: The YUV frame the OCR attempt is about.
            current_time: Wall clock, used only to name debug images.

        Returns:
            The plate crop and its box in frame coordinates, or None when the
            crop holds no usable plate.

        This reads the frame in hand through the box this track currently
        carries. Finding a plate in those pixels establishes a plate region on
        this frame; it does **not** establish that the box still frames this
        track's vehicle. A tracker advances a carried box onto new frames, so a
        caller that turns the result into a new plate-to-track association owes
        that check itself - see `_vehicle_measured_on_frame`.
        """
        car_box = obj_data.get("box")

        if not car_box:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

        # apply motion mask
        rgb[self.config.cameras[camera].motion.rasterized_mask == 0] = [0, 0, 0]  # type: ignore[attr-defined]

        left, top, right, bottom = (int(value) for value in car_box)
        car = rgb[top:bottom, left:right]

        if car.size == 0:
            logger.debug(f"{camera}: Vehicle box {car_box} selects no pixels")
            return None

        # double the size of the car for better box detection
        car = cv2.resize(car, (int(2 * car.shape[1]), int(2 * car.shape[0])))

        if WRITE_DEBUG_IMAGES:
            cv2.imwrite(
                f"debug/frames/car_frame_{current_time}.jpg",
                car,
            )

        yolov9_start = datetime.datetime.now().timestamp()
        license_plate = self._detect_license_plate(camera, car)
        logger.debug(
            f"{camera}: YOLOv9 LPD inference time: {(datetime.datetime.now().timestamp() - yolov9_start) * 1000:.2f} ms"
        )
        self.plates_det_second.update()
        self.plate_det_speed.update(datetime.datetime.now().timestamp() - yolov9_start)

        if not license_plate:
            logger.debug(f"{camera}: Detected no license plates for vehicle object.")
            return None

        license_plate_area = max(
            0,
            (license_plate[2] - license_plate[0])
            * (license_plate[3] - license_plate[1]),
        )

        # check that license plate is valid
        # quadruple the value because we've doubled both dimensions of the car
        if license_plate_area < self.config.cameras[camera].lpr.min_area * 4:
            logger.debug(f"{camera}: License plate is less than min_area")
            return None

        # Scale back to original car coordinates and then to frame
        plate_box = (
            left + license_plate[0] // 2,
            top + license_plate[1] // 2,
            left + license_plate[2] // 2,
            top + license_plate[3] // 2,
        )

        return (
            car[
                license_plate[1] : license_plate[3],
                license_plate[0] : license_plate[2],
            ],
            plate_box,
        )

    def _generate_plate_event(
        self, camera: str, plate: str, plate_score: float, source_frame_time: float
    ) -> str:
        """Generate a unique ID for a plate event based on camera and text."""
        now = source_frame_time
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        event_id = f"{now}-{rand_id}"

        self.event_metadata_publisher.publish(
            (
                now,
                camera,
                "license_plate",
                event_id,
                True,
                plate_score,
                None,
                plate,
            ),
            EventMetadataTypeEnum.lpr_event_create.value,
        )
        return event_id

    def lpr_process(
        self,
        obj_data: dict[str, Any],
        frame: np.ndarray,
        dedicated_lpr: bool = False,
        *,
        source_frame_time: float,
        reprojected_regions: bool = False,
    ) -> None:
        """Look for license plates in image.

        Args:
            reprojected_regions: Whether the caller has already mapped this
                object's regions onto these exact pixels. The post processor
                does that when it re-reads a plate from a recording keyframe,
                so its regions belong to the image even though no detector ran
                on it. Nothing else may claim this: it turns off both the
                same-frame check and the secondary fallback, whose vehicle box
                would be in the wrong coordinate space on a re-projected image.
        """
        self.metrics.alpr_pps.value = self.plates_rec_second.eps()
        self.metrics.yolov9_lpr_pps.value = self.plates_det_second.eps()
        camera = obj_data["camera"]
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        if (
            isinstance(source_frame_time, bool)
            or not isinstance(source_frame_time, (int, float))
            or not math.isfinite(source_frame_time)
            or source_frame_time <= 0
            or source_frame_time > current_time
        ):
            return

        if not self.config.cameras[camera].lpr.enabled:
            return

        # dedicated LPR cam without frigate+
        if dedicated_lpr:
            id = "dedicated-lpr"

            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

            # apply motion mask
            rgb[self.config.cameras[camera].motion.rasterized_mask == 0] = [0, 0, 0]  # type: ignore[attr-defined]

            if WRITE_DEBUG_IMAGES:
                cv2.imwrite(
                    f"debug/frames/dedicated_lpr_masked_{current_time}.jpg",
                    rgb,
                )

            yolov9_start = datetime.datetime.now().timestamp()
            license_plate = self._detect_license_plate(camera, rgb)

            logger.debug(
                f"{camera}: YOLOv9 LPD inference time: {(datetime.datetime.now().timestamp() - yolov9_start) * 1000:.2f} ms"
            )
            self.plates_det_second.update()
            self.plate_det_speed.update(
                datetime.datetime.now().timestamp() - yolov9_start
            )

            if not license_plate:
                logger.debug(f"{camera}: Detected no license plates in full frame.")
                return

            license_plate_area = (license_plate[2] - license_plate[0]) * (
                license_plate[3] - license_plate[1]
            )
            if license_plate_area < self.config.cameras[camera].lpr.min_area:
                logger.debug(f"{camera}: License plate area below minimum threshold.")
                return

            plate_box = license_plate

            license_plate_frame = rgb[
                license_plate[1] : license_plate[3],
                license_plate[0] : license_plate[2],
            ]

            # Double the size for better OCR
            license_plate_frame = cv2.resize(
                license_plate_frame,
                (
                    int(2 * license_plate_frame.shape[1]),
                    int(2 * license_plate_frame.shape[0]),
                ),
            )

        else:
            id = obj_data["id"]

            # don't run for non car/motorcycle or non license plate (dedicated lpr with frigate+) objects
            if (
                obj_data.get("label") not in self.lp_objects
                and obj_data.get("label") != "license_plate"
            ):
                logger.debug(
                    f"{camera}: Not a processing license plate for non car/motorcycle object."
                )
                return

            # don't run for non-stationary objects with no position changes to avoid processing uncertain moving objects
            # zero position_changes is the initial state after registering a new tracked object
            # LPR will run 2 frames after detect.min_initialized is reached
            if obj_data.get("position_changes", 0) == 0 and not obj_data.get(
                "stationary", False
            ):
                logger.debug(
                    f"{camera}: Skipping LPR for non-stationary {obj_data['label']} object {id} with no position changes.  (Detected in {self.config.cameras[camera].detect.min_initialized + 1} concurrent frames, threshold to run is {self.config.cameras[camera].detect.min_initialized + 2} frames)"  # type: ignore[operator]
                )
                return

            plate_region: dict[str, Any] | None = None
            located: tuple[np.ndarray, tuple[int, int, int, int]] | None = None

            if "license_plate" not in self.config.cameras[camera].objects.track:
                logger.debug(f"{camera}: Running manual license_plate detection.")
                located = self._locate_plate_in_vehicle(
                    camera, obj_data, frame, current_time
                )
                if located is None:
                    return
            elif obj_data.get("label") == "license_plate":
                # dedicated lpr with frigate+: the plate is the tracked object
                plate_region = obj_data
            else:
                plate_region = self._current_plate_attribute(
                    camera,
                    obj_data,
                    None if reprojected_regions else source_frame_time,
                )

            # The same check the scheduler used to decide this frame was worth
            # a slot, so the two can never disagree about what is readable.
            license_plate_box = self._usable_plate_box(camera, plate_region)

            if license_plate_box:
                license_plate_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                # Expand the license_plate_box by 10%
                box_array = np.array(license_plate_box)
                expansion = (box_array[2:] - box_array[:2]) * 0.10
                expanded_box = np.array(
                    [
                        license_plate_box[0] - expansion[0],
                        license_plate_box[1] - expansion[1],
                        license_plate_box[2] + expansion[0],
                        license_plate_box[3] + expansion[1],
                    ]
                ).clip(
                    0, [license_plate_frame.shape[1], license_plate_frame.shape[0]] * 2
                )

                plate_box = (
                    int(expanded_box[0]),
                    int(expanded_box[1]),
                    int(expanded_box[2]),
                    int(expanded_box[3]),
                )

                # Crop using the expanded box
                license_plate_frame = license_plate_frame[
                    int(expanded_box[1]) : int(expanded_box[3]),
                    int(expanded_box[0]) : int(expanded_box[2]),
                ]
            else:
                if located is None:
                    # The model published no plate region it measured on this
                    # frame, so look once inside the vehicle instead - but only
                    # when the model measured this vehicle here too. Reading a
                    # carried box would publish whatever plate has since moved
                    # into it under this track's identity, which in close
                    # headway is the next car's. A plate that is itself the
                    # tracked object has no vehicle to look inside, and a
                    # re-projected image's vehicle box is in the coordinate
                    # space of some other frame.
                    if reprojected_regions or obj_data.get("label") not in (
                        self.lp_objects
                    ):
                        return

                    if not self._vehicle_measured_on_frame(obj_data, source_frame_time):
                        logger.debug(
                            f"{camera}: Skipping license plate fallback for {id}, vehicle last measured at {obj_data.get('detector_observed_at')}, not {source_frame_time}"
                        )
                        return

                    logger.debug(
                        f"{camera}: No current license_plate attribute, falling back to manual detection for {id}."
                    )
                    located = self._locate_plate_in_vehicle(
                        camera, obj_data, frame, current_time
                    )
                    if located is None:
                        return

                license_plate_frame, plate_box = located

            # double the size of the license plate frame for better OCR
            license_plate_frame = cv2.resize(
                license_plate_frame,
                (
                    int(2 * license_plate_frame.shape[1]),
                    int(2 * license_plate_frame.shape[0]),
                ),
            )

            if WRITE_DEBUG_IMAGES:
                cv2.imwrite(
                    f"debug/frames/license_plate_frame_{current_time}.jpg",
                    license_plate_frame,
                )

        logger.debug(f"{camera}: Found license plate. Bounding box: {list(plate_box)}")
        logger.debug(f"{camera}: Running plate recognition for id: {id}.")

        # run detection, returns results sorted by confidence, best first
        start = datetime.datetime.now().timestamp()
        license_plates, confidences, areas = self._process_license_plate(
            camera, id, license_plate_frame
        )
        self.plates_rec_second.update()
        self.plate_rec_speed.update(datetime.datetime.now().timestamp() - start)

        if license_plates:
            for plate, confidence, text_area in zip(license_plates, confidences, areas):
                avg_confidence = (
                    (sum(confidence) / len(confidence)) if confidence else 0
                )

                logger.debug(
                    f"{camera}: Detected text: {plate} (average confidence: {avg_confidence:.2f}, area: {text_area} pixels)"
                )
        else:
            logger.debug(f"{camera}: No text detected")
            return

        top_plate, top_char_confidences, top_area = (
            license_plates[0],
            confidences[0],
            areas[0],
        )
        avg_confidence = (
            (sum(top_char_confidences) / len(top_char_confidences))
            if top_char_confidences
            else 0
        )

        # Check against minimum confidence threshold
        if (
            not math.isfinite(avg_confidence)
            or avg_confidence > 1
            or avg_confidence < self.lpr_config.recognition_threshold
        ):
            logger.debug(
                f"{camera}: Average character confidence {avg_confidence} is less than recognition_threshold ({self.lpr_config.recognition_threshold})"
            )
            return

        # For dedicated LPR cameras, match or assign plate ID using Jaro-Winkler distance
        if (
            dedicated_lpr
            and "license_plate" not in self.config.cameras[camera].objects.track
        ):
            plate_id = None

            for existing_id, data in self.detected_license_plates.items():
                if (
                    data["camera"] == camera
                    and data["last_seen"] is not None
                    and current_time - data["last_seen"]
                    <= self.config.cameras[camera].lpr.expire_time
                ):
                    similarity = JaroWinkler.similarity(data["plate"], top_plate)
                    if similarity >= self.similarity_threshold:
                        plate_id = existing_id
                        logger.debug(
                            f"{camera}: Matched plate {top_plate} to {data['plate']} (similarity: {similarity:.3f})"
                        )
                        break
            if plate_id is None:
                plate_id = self._generate_plate_event(
                    camera, top_plate, avg_confidence, source_frame_time
                )
                logger.debug(
                    f"{camera}: New plate event for dedicated LPR camera {plate_id}: {top_plate}"
                )
            else:
                logger.debug(
                    f"{camera}: Matched existing plate event for dedicated LPR camera {plate_id}: {top_plate}"
                )
                self.detected_license_plates[plate_id]["last_seen"] = current_time

            id = plate_id

        is_new = id not in self.detected_license_plates

        # Collect variant
        variant = {
            "plate": top_plate,
            "conf": avg_confidence,
            "char_confidences": top_char_confidences,
            "area": top_area,
            "timestamp": source_frame_time,
        }

        # Initialize or append to plates
        self.detected_license_plates.setdefault(id, {"plates": [], "camera": camera})
        self.detected_license_plates[id]["plates"].append(variant)

        # Prune old variants - this is probably higher than it needs to be
        # since we don't detect a plate every frame
        num_variants = self.config.cameras[camera].detect.fps * 5
        if len(self.detected_license_plates[id]["plates"]) > num_variants:
            self.detected_license_plates[id]["plates"] = self.detected_license_plates[
                id
            ]["plates"][-num_variants:]

        # Cluster and select rep. Clustering produces the historical/display
        # aggregate only - it never decides what gets published.
        plates = self.detected_license_plates[id]["plates"]
        rep_plate, rep_conf, rep_char_confs, rep_area = self._get_cluster_rep(plates)

        if rep_plate != top_plate:
            logger.debug(
                f"{camera}: Clustering changed top plate '{top_plate}' (conf: {avg_confidence:.3f}) to rep '{rep_plate}' (conf: {rep_conf:.3f})"
            )

        # Apply length and format filters to the clustered representative
        # rather than individual OCR readings, so noisy variants still
        # contribute to clustering even when they don't pass on their own.
        if self._passes_plate_filters(camera, rep_plate, "clustered"):
            self.detected_license_plates[id].update(
                {
                    "plate": rep_plate,
                    "char_confidences": rep_char_confs,
                    "area": rep_area,
                }
            )
        else:
            # Keep the previous aggregate rather than storing a rejected one,
            # while leaving every key present for later reads.
            self.detected_license_plates[id].setdefault("plate", "")
            self.detected_license_plates[id].setdefault("char_confidences", [])
            self.detected_license_plates[id].setdefault("area", 0)

        self.detected_license_plates[id]["last_seen"] = (
            current_time if dedicated_lpr else None
        )

        if not dedicated_lpr:
            self.detected_license_plates[id]["obj_data"] = obj_data

        if is_new:
            if camera not in self.camera_current_cars:
                self.camera_current_cars[camera] = []
            self.camera_current_cars[camera].append(id)

        # Publication describes this sample, not the aggregate. The OCR sample
        # clock only ever moves forward, and only on an actual accepted sample.
        previous_sample_time = self.detected_license_plates[id].get(
            "recognized_license_plate_frame_time", 0
        )
        if source_frame_time <= previous_sample_time:
            return

        # The published text must pass the filters on its own merit; a passing
        # aggregate never vouches for a sample that does not.
        if not self._passes_plate_filters(camera, top_plate, "recognized"):
            return

        self.detected_license_plates[id]["recognized_license_plate_frame_time"] = (
            source_frame_time
        )

        # Determine subLabel based on known plates, use regex matching.
        # Matched against the published text, so the sub label describes what
        # was published rather than some other reading of this track.
        sub_label = None
        try:
            sub_label = next(
                (
                    label
                    for label, plates_list in self.lpr_config.known_plates.items()  # type: ignore[union-attr]
                    if any(
                        re.match(f"^{plate}$", top_plate)
                        or Levenshtein.distance(plate, top_plate)
                        <= self.lpr_config.match_distance
                        for plate in plates_list
                    )
                ),
                None,
            )
        except re.error:
            logger.error(
                f"{camera}: Invalid regex in known plates configuration: {self.lpr_config.known_plates}"
            )

        # If it's a known plate, publish to sub_label
        if sub_label is not None:
            self.sub_label_publisher.publish(
                (id, sub_label, avg_confidence), EventMetadataTypeEnum.sub_label.value
            )

        # Publish the text, confidence and capture time of this exact OCR
        # sample - one sample, one coherent triple.
        self.requestor.send_data(
            "tracked_object_update",
            json.dumps(
                {
                    "type": TrackedObjectUpdateTypesEnum.lpr,
                    "name": sub_label,
                    "plate": top_plate,
                    "score": avg_confidence,
                    "id": id,
                    "camera": camera,
                    "timestamp": source_frame_time,
                    "recognized_license_plate_frame_time": source_frame_time,
                    "plate_box": plate_box,
                }
            ),
        )
        self.sub_label_publisher.publish(
            (
                id,
                "recognized_license_plate",
                top_plate,
                avg_confidence,
                source_frame_time,
            ),
            EventMetadataTypeEnum.attribute.value,
        )

        # save the best snapshot for dedicated lpr cams not using frigate+
        if (
            dedicated_lpr
            and "license_plate" not in self.config.cameras[camera].objects.track
        ):
            logger.debug(
                f"{camera}: Writing snapshot for {id}, {top_plate}, {current_time}"
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            _, encoded_img = cv2.imencode(".jpg", frame_bgr)
            self.sub_label_publisher.publish(
                (base64.b64encode(encoded_img.tobytes()).decode("ASCII"), id, camera),
                EventMetadataTypeEnum.save_lpr_snapshot.value,
            )

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    def lpr_expire(self, object_id: str, camera: str) -> None:
        if object_id in self.detected_license_plates:
            self.detected_license_plates.pop(object_id)

            if object_id in self.camera_current_cars.get(camera, []):
                self.camera_current_cars[camera].remove(object_id)


class CTCDecoder:
    """
    A decoder for interpreting the output of a CTC (Connectionist Temporal Classification) model.

    This decoder converts the model's output probabilities into readable sequences of characters
    while removing duplicates and handling blank tokens. It also calculates the confidence scores
    for each decoded character sequence.
    """

    def __init__(self, character_dict_path: str, expected_class_count: int) -> None:
        """Initialize the decoder from the recognizer's own inference config.

        Args:
            character_dict_path: Path to the recognition model's official
                `inference.yml`. Its `PostProcess.character_dict` is the ordered
                label map the network was trained against.
            expected_class_count: Number of classes the recognition model emits.

        Raises:
            FileNotFoundError: The inference config is missing.
            ValueError: The config is unreadable, does not describe a CTC label
                map, or yields a class count other than `expected_class_count`.

        There is deliberately no built-in fallback list. A wrong label map does
        not fail loudly at decode time - it silently emits the wrong glyph for
        every index - so an unreadable dictionary must stop recognition instead.
        """
        self.characters = self._load_characters(character_dict_path)

        if len(self.characters) != expected_class_count:
            raise ValueError(
                f"Recognition character dictionary at {character_dict_path} "
                f"defines {len(self.characters)} classes but the model emits "
                f"{expected_class_count}"
            )

        self.char_map = {i: char for i, char in enumerate(self.characters)}

    @staticmethod
    def _load_characters(character_dict_path: str) -> List[str]:
        """Read the ordered PaddleOCR label map out of an inference config.

        Entries are taken verbatim and in order. `str.strip()` must never be
        applied: the dictionary contains U+3000 IDEOGRAPHIC SPACE, which Python
        treats as whitespace, and dropping it shifts every later index by one -
        a corruption that still decodes Latin plates and so goes unnoticed.
        """
        if not os.path.exists(character_dict_path):
            raise FileNotFoundError(
                f"Recognition character dictionary not found at {character_dict_path}"
            )

        try:
            with open(character_dict_path, "r", encoding="utf-8") as config_file:
                config = YAML(typ="safe", pure=True).load(config_file)
        except YAMLError as err:
            raise ValueError(
                f"Unable to parse recognition inference config at {character_dict_path}"
            ) from err

        post_process = (config or {}).get("PostProcess") or {}
        character_dict = post_process.get("character_dict")

        if post_process.get("name") != "CTCLabelDecode":
            raise ValueError(
                f"Recognition inference config at {character_dict_path} declares "
                f"post process '{post_process.get('name')}', not CTCLabelDecode"
            )

        if not isinstance(character_dict, list) or not character_dict:
            raise ValueError(
                f"Recognition inference config at {character_dict_path} has no "
                "PostProcess.character_dict label map"
            )

        # A line-based dictionary file cannot hold an empty entry or a line
        # terminator, so either means the label map is not the one the model
        # was trained against.
        for index, entry in enumerate(character_dict):
            if (
                not isinstance(entry, str)
                or not entry
                or "\n" in entry
                or "\r" in entry
            ):
                raise ValueError(
                    f"Recognition inference config at {character_dict_path} has an "
                    f"invalid label map entry at index {index}"
                )

        # PaddleOCR's CTCLabelDecode label order: the CTC blank, the dictionary
        # verbatim, then the trailing space class.
        return ["blank"] + list(character_dict) + [" "]

    def __call__(
        self, outputs: List[np.ndarray]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Decode a batch of model outputs into character sequences and their confidence scores.

        The method takes the output probability distributions for each time step and uses
        the best path decoding strategy. It then merges repeating characters and ignores
        blank tokens. Confidence scores for each decoded character are also calculated.

        Args:
            outputs (List[np.ndarray]): A list of model outputs, where each element is
                                        a probability distribution for each time step.

        Returns:
            Tuple[List[str], List[List[float]]]: A tuple of decoded character sequences
                                                and confidence scores for each sequence.
        """
        results = []
        confidences = []
        for output in outputs:
            if output.shape[-1] != len(self.characters):
                # The loaded label map does not describe this model's output,
                # so every index would decode to the wrong glyph. Recognize
                # nothing rather than publish a confident wrong plate.
                logger.error(
                    "Recognition model emitted %d classes but the character dictionary defines %d",
                    output.shape[-1],
                    len(self.characters),
                )
                return [], []

            seq_log_probs = np.log(output + 1e-8)
            best_path = np.argmax(seq_log_probs, axis=1)

            merged_path = []
            merged_probs = []
            for t, char_index in enumerate(best_path):
                if char_index != 0 and (t == 0 or char_index != best_path[t - 1]):
                    merged_path.append(char_index)
                    merged_probs.append(seq_log_probs[t, char_index])

            result = "".join(self.char_map.get(idx, "") for idx in merged_path)
            results.append(result)

            confidence = np.exp(merged_probs).tolist()
            confidences.append(confidence)

        return results, confidences
