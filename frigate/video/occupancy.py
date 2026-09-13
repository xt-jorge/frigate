"""Detector coverage of existing commissioned occupancy zones."""

import logging
from math import isfinite
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from frigate.track.stationary_classifier import StationaryMotionClassifier

if TYPE_CHECKING:
    from frigate.config import CameraConfig

logger = logging.getLogger(__name__)
MAX_FOOTPRINTS = 64

Box = tuple[int, int, int, int]
Detection = tuple[str, float, Box, int, float, Box]


def contains(outer: Box, inner: Box) -> bool:
    """Return whether one detector crop covers the complete zone bounds."""
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def zone_bounds(camera_config: "CameraConfig") -> list[Box] | None:
    """Return configured bounds, or unknown if any selected zone is unavailable."""
    bounds = []
    for name in camera_config.detect.occupancy_zones:
        zone = camera_config.zones.get(name)
        if zone is None or not zone.enabled or len(zone.contour) < 3:
            return None
        points = zone.contour
        bounds.append(
            (
                int(min(p[0] for p in points)),
                int(min(p[1] for p in points)),
                int(max(p[0] for p in points)),
                int(max(p[1] for p in points)),
            )
        )
    return bounds


def occupancy_frame(
    camera: str,
    frame_time: float,
    shape: tuple[int, int],
    bounds: list[Box] | None,
    coverage: list[Box],
    detections: list[Detection],
    tracks: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze current detections and existing track continuity for the same frame."""
    # Detector crops can include padding beyond the frame; only image pixels
    # within a successfully inferred crop establish occupancy coverage.
    image_coverage = []
    for x1, y1, x2, y2 in coverage:
        clipped = (max(0, x1), max(0, y1), min(shape[1], x2), min(shape[0], y2))
        if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
            image_coverage.append(clipped)
    return {
        "camera": camera,
        "frame_time": frame_time,
        "width": shape[1],
        "height": shape[0],
        "complete": bool(bounds)
        and isfinite(frame_time)
        and frame_time > 0
        and all(
            any(contains(region, box) for region in image_coverage) for box in bounds
        ),
        "coverage": [list(region) for region in image_coverage],
        "tracks": tracks,
        "regions": regions,
        "objects": [
            {
                "label": detection[0],
                "box": list(detection[2]),
                "detector_observed_at": frame_time,
            }
            for detection in detections
        ],
    }


def overlaps(first: Box, second: Box) -> bool:
    return (
        first[0] < second[2]
        and first[2] > second[0]
        and first[1] < second[3]
        and first[3] > second[1]
    )


def clip_box(box: Box, region: Box) -> Box:
    return (
        max(region[0], box[0]),
        max(region[1], box[1]),
        min(region[2], box[2]),
        min(region[3], box[3]),
    )


def intersects_polygon(box: Box, contour: np.ndarray) -> bool:
    """Whole-box overlap against the existing contour, including its boundary."""
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(cv2.pointPolygonTest(contour, point, False) >= 0 for point in corners):
        return True
    rect = (int(x1), int(y1), int(x2 - x1) + 1, int(y2 - y1) + 1)
    return any(
        cv2.clipLine(rect, tuple(map(int, first)), tuple(map(int, second)))[0]
        for first, second in zip(contour, np.roll(contour, -1, axis=0))
    )


class OccupancyContinuity:
    """Reuse native stationary footprints; rejected candidates get the tracker grace."""

    def __init__(self, rejection_grace: float) -> None:
        self.classifier = StationaryMotionClassifier()
        self.rejection_grace = rejection_grace
        self.footprints: dict[str, tuple[Box, str]] = {}
        self.raw_pending: dict[str, dict[str, float]] = {}
        self.zone_configuration: dict[str, bytes] = {}
        self.saturated: set[str] = set()
        self.last_frame = 0.0
        self.last_complete = False

    def observe(
        self,
        frame: np.ndarray,
        frame_time: float,
        zones: dict[str, np.ndarray],
        detections: list[Detection],
        tracks: list[dict[str, Any]],
        complete: bool,
    ) -> list[dict[str, Any]]:
        bounds = {
            name: (
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()),
                int(points[:, 1].max()),
            )
            for name, points in zones.items()
        }
        if frame_time <= self.last_frame:
            return [
                {"zone": name, "box": list(box), "uncertain": True}
                for name, box in bounds.items()
            ]
        if not complete or not self.last_complete or frame_time - self.last_frame > 1:
            for pending in self.raw_pending.values():
                for identity in pending:
                    pending[identity] = frame_time
        self.last_frame = frame_time
        self.last_complete = complete
        configuration = {name: points.tobytes() for name, points in zones.items()}
        for key, (_, zone) in list(self.footprints.items()):
            if configuration.get(zone) != self.zone_configuration.get(zone):
                self.footprints.pop(key)
                self.forget_footprint(key)
        self.raw_pending = {
            name: ids
            for name, ids in self.raw_pending.items()
            if name in configuration
            and configuration[name] == self.zone_configuration.get(name)
        }
        self.saturated = {
            zone
            for zone in self.saturated
            if configuration.get(zone) == self.zone_configuration.get(zone)
        }
        self.zone_configuration = configuration
        result = []
        for zone, region in bounds.items():
            if not complete:
                result.append({"zone": zone, "box": list(region), "uncertain": True})
                continue
            contour = zones[zone]
            current = [
                track for track in tracks if intersects_polygon(track["box"], contour)
            ]
            raw = [
                detection
                for detection in detections
                if intersects_polygon(detection[2], contour)
            ]
            confirmed = [track for track in current if track["initialized"]]
            pending = self.raw_pending.setdefault(zone, {})
            live_ids = {track["id"] for track in current}
            for identity, seen in list(pending.items()):
                if (
                    identity not in live_ids
                    and frame_time - seen >= self.rejection_grace
                ):
                    pending.pop(identity)
            for track in confirmed:
                pending.pop(track["id"], None)
            for track in confirmed:
                if (
                    track["frame_time"] != frame_time
                    or track.get("detector_observed_at") != frame_time
                ):
                    continue
                box = clip_box(track["box"], region)
                # Boundary contact still vetoes through the current track inventory,
                # but has no occupied image area from which to retain a footprint.
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                key = f"{zone}:{track['id']}"
                self.footprints.pop(key, None)
                self.forget_footprint(key)
                # Aggregate geometry, not vehicle identity: a fresh footprint
                # covering an older one carries all of its occupied pixels.
                for old_key, (old_box, old_zone) in list(self.footprints.items()):
                    if old_zone == zone and contains(box, old_box):
                        self.footprints.pop(old_key)
                        self.forget_footprint(old_key)
                if len(self.footprints) >= MAX_FOOTPRINTS:
                    if zone not in self.saturated:
                        logger.warning(
                            "Occupancy footprint capacity reached for zone %s; state unknown",
                            zone,
                        )
                    self.saturated.add(zone)
                    continue
                self.forget_footprint(key)
                for name, part in self.footprint_parts(key, box):
                    self.classifier.ensure_anchor(name, frame, part)
                self.footprints[key] = (box, zone)
            # Raw boxes not accounted for by a confirmed track remain candidates,
            # not permanent claims that a particular vehicle must later discharge.
            for detection in raw:
                if any(contains(track["box"], detection[2]) for track in confirmed):
                    continue
                candidates = [
                    track for track in current if contains(track["box"], detection[2])
                ]
                for track in candidates:
                    pending[track["id"]] = frame_time
                if not candidates:
                    pending["untracked"] = frame_time
            uncertain = bool(pending) or zone in self.saturated
            current_keys = {f"{zone}:{track['id']}" for track in current}
            for key, (box, area) in list(self.footprints.items()):
                if area != zone:
                    continue
                if key in current_keys or self.keep_footprint(key, frame, box):
                    uncertain = True
                else:
                    self.footprints.pop(key)
                    self.forget_footprint(key)
            result.append({"zone": zone, "box": list(region), "uncertain": uncertain})
        return result

    @staticmethod
    def footprint_parts(key: str, box: Box) -> list[tuple[str, Box]]:
        """Fixed image patches within one object footprint, not camera calibration."""
        x1, y1, x2, y2 = box
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if mx <= x1 or my <= y1:
            return [(key, box)]
        return [(key, box)] + [
            (f"{key}:{index}", part)
            for index, part in enumerate(
                [(x1, y1, mx, my), (mx, y1, x2, my), (x1, my, mx, y2), (mx, my, x2, y2)]
            )
        ]

    def keep_footprint(self, key: str, frame: np.ndarray, box: Box) -> bool:
        unchanged = [
            self.classifier.evaluate(name, frame, part)
            for name, part in self.footprint_parts(key, box)
        ]
        # A door, occupant, or partial occlusion may change the whole-crop score.
        # Keep the obstruction while at least half of its sub-patches persist.
        return unchanged[0] or sum(unchanged[1:]) >= 2

    def forget_footprint(self, key: str) -> None:
        for name in [key, *(f"{key}:{index}" for index in range(4))]:
            self.classifier.anchor_crops.pop(name, None)
            self.classifier.anchor_boxes.pop(name, None)
            self.classifier.changed_counts.pop(name, None)
            self.classifier.shift_histories.pop(name, None)
