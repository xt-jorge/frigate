from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from frigate.config import DetectConfig


class ObjectTracker(ABC):
    @abstractmethod
    def __init__(self, config: DetectConfig) -> None:
        pass

    @abstractmethod
    def update_frame_times(self, frame: np.ndarray, frame_time: float) -> None:
        """Advance existing lives against the frame the caller already owns."""
        pass

    @abstractmethod
    def match_and_update(
        self,
        frame: np.ndarray,
        frame_time: float,
        detections: list[tuple[Any, Any, Any, Any, Any, Any]],
        *,
        detector_observed_at: list[float | None] | None = None,
    ) -> None:
        """Match detections measured on ``frame`` at ``frame_time``.

        The caller owns the pixels and passes them in. A tracker must never
        reopen a shared-memory slot to recover them: the camera may already
        have reused it, and a missing frame must not stall tracking.
        """
        pass

    @abstractmethod
    def occupancy_tracks(self) -> list[dict[str, Any]]:
        """Return measured geometry for the tracker's existing object lives."""
        pass
