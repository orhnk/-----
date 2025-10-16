from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

MaskColor = Tuple[int, int, int]


@dataclass
class DetectionConfig:
    camera_index: int = 0
    target_fps: int = 30
    mask_color: MaskColor = (0, 255, 0)
    mask_alpha: float = 1.0
    border_thickness: int = 2
    max_faces: int = 5
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    enable_face_mask: bool = True
    enable_body_mask: bool = True
    segmentation_threshold: float = 0.5
    segmentation_model_selection: int = 1
    segmentation_smooth_factor: float = 0.6
    segmentation_kernel_size: int = 7


__all__ = ["DetectionConfig", "MaskColor"]
