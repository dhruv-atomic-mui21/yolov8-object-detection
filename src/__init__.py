"""
YOLOv8 Object Detection Package
Real-time object detection with Ultralytics YOLOv8
"""

from .detector import YOLOv8Detector
from .utils import (
    xywh2xyxy,
    xyxy2xywh,
    scale_coords,
    non_max_suppression,
    box_iou,
    load_classes
)
from .visualize import (
    draw_detections,
    create_detection_video,
    plot_results
)
from .model import (
    load_model,
    export_model,
    get_model_info,
    ModelConfig
)

__version__ = "2.0.0"
__author__ = "Dhruv"

__all__ = [
    # Detector
    "YOLOv8Detector",
    
    # Bounding box utilities
    "xywh2xyxy",
    "xyxy2xywh", 
    "scale_coords",
    "non_max_suppression",
    "box_iou",
    "load_classes",
    
    # Visualization
    "draw_detections",
    "create_detection_video",
    "plot_results",
    
    # Model utilities
    "load_model",
    "export_model",
    "get_model_info",
    "ModelConfig"
]
