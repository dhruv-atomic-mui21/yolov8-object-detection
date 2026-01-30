"""
YOLOv8 Object Detection API Package
FastAPI endpoints for real-time object detection
"""

from .app import app, get_detector
from .schemas import (
    Detection,
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    DetectionRequest,
    BatchDetectionRequest
)

__all__ = [
    "app",
    "get_detector",
    "Detection",
    "DetectionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "DetectionRequest",
    "BatchDetectionRequest"
]
