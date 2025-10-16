from __future__ import annotations

import argparse
from typing import Sequence

from .config import DetectionConfig, MaskColor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live face detector with mask overlay.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target refresh rate in frames per second (default: 30).",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=1.0,
        help="Opacity of the mask overlay between 0 and 1 (default: 1 for opaque).",
    )
    parser.add_argument(
        "--mask-color",
        type=str,
        default="0,255,0",
        help="Mask color as comma-separated B,G,R values (default: 0,255,0).",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="Maximum number of faces to detect simultaneously (default: 5).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for the face mesh between 0 and 1 (default: 0.5).",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for the face mesh between 0 and 1 (default: 0.5).",
    )
    parser.add_argument(
        "--disable-face-mask",
        action="store_true",
        help="Disable the face polygon mask overlay.",
    )
    parser.add_argument(
        "--disable-body-mask",
        action="store_true",
        help="Disable the body segmentation mask overlay.",
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1) used to filter the body segmentation mask (default: 0.5).",
    )
    parser.add_argument(
        "--segmentation-model",
        type=int,
        choices=(0, 1),
        default=1,
        help="MediaPipe Selfie Segmentation model selection (0: general, 1: landscape).",
    )
    parser.add_argument(
        "--segmentation-smooth-factor",
        type=float,
        default=0.6,
        help="Temporal smoothing factor (0-0.99) applied to the body segmentation mask (default: 0.6).",
    )
    parser.add_argument(
        "--segmentation-kernel-size",
        type=int,
        default=7,
        help="Odd kernel size used for spatial smoothing and morphological cleanup of the body mask (default: 7).",
    )
    return parser


def parse_args(argv: Sequence[str]) -> DetectionConfig:
    parser = build_parser()
    namespace = parser.parse_args(list(argv))

    mask_color = _parse_color(namespace.mask_color)
    return DetectionConfig(
        camera_index=namespace.camera,
        target_fps=max(namespace.fps, 1),
        mask_color=mask_color,
        mask_alpha=_clamp(namespace.mask_alpha, 0.0, 1.0),
        max_faces=max(namespace.max_faces, 1),
        detection_confidence=_clamp(namespace.min_detection_confidence, 0.0, 1.0),
        tracking_confidence=_clamp(namespace.min_tracking_confidence, 0.0, 1.0),
        enable_face_mask=not namespace.disable_face_mask,
        enable_body_mask=not namespace.disable_body_mask,
        segmentation_threshold=_clamp(namespace.segmentation_threshold, 0.0, 1.0),
        segmentation_model_selection=int(namespace.segmentation_model),
        segmentation_smooth_factor=_clamp(namespace.segmentation_smooth_factor, 0.0, 0.99),
        segmentation_kernel_size=max(namespace.segmentation_kernel_size, 1),
    )


def _parse_color(spec: str) -> MaskColor:
    components = spec.split(",")
    if len(components) != 3:
        raise argparse.ArgumentTypeError("Mask color must be three comma-separated integers between 0 and 255.")

    try:
        values = tuple(int(part) for part in components)
    except ValueError as exc:  # pragma: no cover - validation error path
        raise argparse.ArgumentTypeError("Mask color components must be integers.") from exc

    for value in values:
        if not 0 <= value <= 255:
            raise argparse.ArgumentTypeError("Mask color components must be between 0 and 255.")
    return values  # type: ignore[return-value]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["build_parser", "parse_args"]
