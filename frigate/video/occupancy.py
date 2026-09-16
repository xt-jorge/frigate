"""Detector coverage of existing commissioned occupancy zones."""

import json
import logging
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from frigate.track.stationary_classifier import StationaryMotionClassifier

if TYPE_CHECKING:
    from frigate.config import CameraConfig

logger = logging.getLogger(__name__)
MAX_FOOTPRINTS = 64
DIAGNOSTIC_INTERVAL_SECONDS = 10

Box = tuple[int, int, int, int]
Detection = tuple[str, float, Box, int, float, Box]


@dataclass(frozen=True)
class _Footprint:
    box: Box
    zone: str
    observed_at: float


@dataclass
class _DiagnosticWindow:
    last_report_at: float | None = None
    frames: int = 0
    reason_frames: Counter[str] = field(default_factory=Counter)
    candidate_observation_frames: int = 0
    untracked_observation_frames: int = 0
    grace_rearm_frames: int = 0


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
        self.footprints: dict[str, _Footprint] = {}
        self.raw_pending: dict[str, dict[str, float]] = {}
        self.zone_configuration: dict[str, bytes] = {}
        self.saturated: set[str] = set()
        self.last_frame = 0.0
        self.last_complete = False
        self.diagnostics: dict[str, _DiagnosticWindow] = {}

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
        self.diagnostics = {
            zone: window for zone, window in self.diagnostics.items() if zone in zones
        }
        if frame_time <= self.last_frame:
            for zone in bounds:
                self.record_diagnostics(zone, frame_time, ["non_monotonic_frame"])
            return [
                {"zone": name, "box": list(box), "uncertain": True}
                for name, box in bounds.items()
            ]
        grace_rearmed = set()
        if not complete or not self.last_complete or frame_time - self.last_frame > 1:
            for zone, pending in self.raw_pending.items():
                if pending:
                    grace_rearmed.add(zone)
                for identity in pending:
                    pending[identity] = frame_time
        self.last_frame = frame_time
        self.last_complete = complete
        configuration = {name: points.tobytes() for name, points in zones.items()}
        for key, footprint in list(self.footprints.items()):
            if configuration.get(footprint.zone) != self.zone_configuration.get(
                footprint.zone
            ):
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
                self.record_diagnostics(
                    zone,
                    frame_time,
                    ["incomplete_coverage"],
                    grace_rearmed=zone in grace_rearmed,
                )
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
                for old_key, old in list(self.footprints.items()):
                    if old.zone == zone and contains(box, old.box):
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
                self.footprints[key] = _Footprint(box, zone, frame_time)
            # Raw boxes not accounted for by a confirmed track remain candidates,
            # not permanent claims that a particular vehicle must later discharge.
            candidate_observed = untracked_observed = False
            for detection in raw:
                if any(contains(track["box"], detection[2]) for track in confirmed):
                    continue
                candidate_observed = True
                candidates = [
                    track for track in current if contains(track["box"], detection[2])
                ]
                for track in candidates:
                    pending[track["id"]] = frame_time
                if not candidates:
                    untracked_observed = True
                    pending["untracked"] = frame_time
            uncertain = bool(pending) or zone in self.saturated
            reasons = []
            if pending:
                reasons.append("pending_candidates")
            if zone in self.saturated:
                reasons.append("saturated")
            current_keys = {f"{zone}:{track['id']}" for track in current}
            track_footprints = pixel_footprints = 0
            for key, footprint in list(self.footprints.items()):
                if footprint.zone != zone:
                    continue
                if key in current_keys:
                    track_footprints += 1
                    uncertain = True
                elif self.keep_footprint(key, frame, footprint.box):
                    pixel_footprints += 1
                    uncertain = True
                else:
                    self.footprints.pop(key)
                    self.forget_footprint(key)
            if track_footprints:
                reasons.append("current_track")
            if pixel_footprints:
                reasons.append("retained_pixels")
            self.record_diagnostics(
                zone,
                frame_time,
                reasons or ["no_continuity_hold"],
                current=current,
                raw_count=len(raw),
                track_footprints=track_footprints,
                pixel_footprints=pixel_footprints,
                candidate_observed=candidate_observed,
                untracked_observed=untracked_observed,
                grace_rearmed=zone in grace_rearmed,
            )
            result.append({"zone": zone, "box": list(region), "uncertain": uncertain})
        return result

    def record_diagnostics(
        self,
        zone: str,
        frame_time: float,
        reasons: list[str],
        *,
        current: list[dict[str, Any]] | None = None,
        raw_count: int | None = None,
        track_footprints: int | None = None,
        pixel_footprints: int | None = None,
        candidate_observed: bool = False,
        untracked_observed: bool = False,
        grace_rearmed: bool = False,
    ) -> None:
        """Count every hold but log only bounded, local summaries without object IDs.

        Reason counts cover the window; inventory and ages describe its last frame.
        An absent continuity hold does not establish clearance: the consumer also
        evaluates current object and track overlap from the unchanged frame payload.
        Pending quiet age restarts on a sighting or coverage gap. Footprint age is
        since its current image anchor was captured, not since the first obstacle.
        Neither clock is a deadline for clearing occupancy.
        """
        if zone not in self.diagnostics:
            self.diagnostics[zone] = _DiagnosticWindow()
        window = self.diagnostics[zone]
        window.frames += 1
        window.reason_frames.update(reasons)
        window.candidate_observation_frames += candidate_observed
        window.untracked_observation_frames += untracked_observed
        window.grace_rearm_frames += grace_rearmed
        now = time.monotonic()
        if (
            window.last_report_at is not None
            and now - window.last_report_at < DIAGNOSTIC_INTERVAL_SECONDS
        ):
            return
        pending = self.raw_pending.get(zone, {})
        footprints = [value for value in self.footprints.values() if value.zone == zone]
        clocks = [
            track["detector_observed_at"]
            for track in current or []
            if track.get("detector_observed_at") is not None
        ]

        def oldest_age(clocks: list[float]) -> float | None:
            return round(max(0.0, frame_time - min(clocks)), 3) if clocks else None

        summary = {
            "process": mp.current_process().name,
            "zone": zone,
            "frame_time": frame_time,
            "window_s": round(now - window.last_report_at, 3)
            if window.last_report_at is not None
            else 0.0,
            "frames": window.frames,
            "reason_frames": dict(window.reason_frames),
            "candidate_observation_frames": window.candidate_observation_frames,
            "untracked_observation_frames": window.untracked_observation_frames,
            "grace_rearm_frames": window.grace_rearm_frames,
            "pending_candidates": len(pending),
            "pending_untracked": "untracked" in pending,
            "oldest_pending_quiet_age_s": oldest_age(list(pending.values())),
            "footprints": len(footprints),
            "oldest_footprint_anchor_age_s": oldest_age(
                [value.observed_at for value in footprints]
            ),
            "saturated": zone in self.saturated,
            "raw_overlaps": raw_count,
            "track_overlaps": len(current) if current is not None else None,
            "uninitialized_overlaps": sum(not track["initialized"] for track in current)
            if current is not None
            else None,
            "stale_track_overlaps": sum(
                track["frame_time"] != frame_time
                or track.get("detector_observed_at") != frame_time
                for track in current
            )
            if current is not None
            else None,
            "oldest_track_observation_age_s": oldest_age(clocks),
            "track_footprints": track_footprints,
            "pixel_footprints": pixel_footprints,
        }
        logger.info(
            "Occupancy continuity %s", json.dumps(summary, separators=(",", ":"))
        )
        window.last_report_at = now
        window.frames = 0
        window.reason_frames.clear()
        window.candidate_observation_frames = 0
        window.untracked_observation_frames = 0
        window.grace_rearm_frames = 0

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
