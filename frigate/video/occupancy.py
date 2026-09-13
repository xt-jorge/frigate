"""Detector coverage of existing commissioned occupancy zones."""

from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frigate.config import CameraConfig

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
) -> dict[str, Any]:
    """Freeze raw current detector results, including positively observed emptiness."""
    return {
        "camera": camera,
        "frame_time": frame_time,
        "width": shape[1],
        "height": shape[0],
        "complete": bool(bounds)
        and isfinite(frame_time)
        and frame_time > 0
        and all(any(contains(region, box) for region in coverage) for box in bounds),
        "coverage": [list(region) for region in coverage],
        "objects": [
            {
                "label": detection[0],
                "box": list(detection[2]),
                "detector_observed_at": frame_time,
            }
            for detection in detections
        ],
    }
