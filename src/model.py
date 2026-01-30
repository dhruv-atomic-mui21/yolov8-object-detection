"""
YOLOv8 Model Utilities
Model loading, export, and configuration helpers for Ultralytics YOLOv8
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from ultralytics import YOLO


# Available YOLOv8 model sizes
MODEL_VARIANTS = {
    'n': 'yolov8n.pt',   # Nano - fastest, smallest
    's': 'yolov8s.pt',   # Small
    'm': 'yolov8m.pt',   # Medium
    'l': 'yolov8l.pt',   # Large
    'x': 'yolov8x.pt',   # Extra large - most accurate
}


def load_model(
    weights: str = 'yolov8n.pt',
    device: Optional[str] = None,
    task: str = 'detect'
) -> YOLO:
    """
    Load YOLOv8 model from weights file.
    
    Args:
        weights: Model weights ('yolov8n.pt', 'yolov8s.pt', etc.) or path to custom weights
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        task: Task type ('detect', 'segment', 'classify', 'pose')
        
    Returns:
        Loaded YOLOv8 model
    """
    # Load model using Ultralytics
    model = YOLO(weights, task=task)
    
    # Move to device if specified
    if device:
        model.to(device)
    
    return model


def export_model(
    model: YOLO,
    format: str = 'onnx',
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True
) -> str:
    """
    Export YOLOv8 model to various formats.
    
    Args:
        model: YOLOv8 model to export
        format: Export format ('onnx', 'torchscript', 'tensorrt', 'coreml', etc.)
        imgsz: Input image size
        half: Use FP16 half precision
        dynamic: Use dynamic axes
        simplify: Simplify ONNX model
        
    Returns:
        Path to exported model
    """
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify
    )
    print(f"Model exported to: {export_path}")
    return str(export_path)


def get_model_info(model: YOLO) -> Dict[str, Any]:
    """
    Get information about the YOLOv8 model.
    
    Args:
        model: YOLOv8 model
        
    Returns:
        Dictionary containing model information
    """
    info = {
        'model_type': 'YOLOv8',
        'task': model.task,
        'device': str(model.device),
        'num_classes': len(model.names),
        'class_names': model.names,
    }
    
    # Get model parameters if available
    if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        info['total_parameters_millions'] = total_params / 1e6
    
    return info


def check_environment() -> Dict[str, Any]:
    """
    Check the current environment for YOLOv8 compatibility.
    
    Returns:
        Environment information dictionary
    """
    import ultralytics
    
    env_info = {
        'ultralytics_version': ultralytics.__version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        env_info['gpu_count'] = torch.cuda.device_count()
    
    return env_info


def list_available_models() -> Dict[str, str]:
    """
    List available pre-trained YOLOv8 models.
    
    Returns:
        Dictionary of model variants and their weights files
    """
    return MODEL_VARIANTS.copy()


class ModelConfig:
    """Configuration class for YOLOv8 model settings."""
    
    def __init__(
        self,
        weights: str = 'yolov8n.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        half: bool = False
    ):
        """
        Initialize model configuration.
        
        Args:
            weights: Model weights file ('yolov8n.pt', 'yolov8s.pt', etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            half: Whether to use FP16 half precision
        """
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = half and (self.device != 'cpu')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'weights': self.weights,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'img_size': self.img_size,
            'device': self.device,
            'half': self.half
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"ModelConfig({self.to_dict()})"
    
    @classmethod
    def nano(cls) -> 'ModelConfig':
        """Get config for nano model (fastest)."""
        return cls(weights='yolov8n.pt')
    
    @classmethod
    def small(cls) -> 'ModelConfig':
        """Get config for small model."""
        return cls(weights='yolov8s.pt')
    
    @classmethod
    def medium(cls) -> 'ModelConfig':
        """Get config for medium model."""
        return cls(weights='yolov8m.pt')
    
    @classmethod
    def large(cls) -> 'ModelConfig':
        """Get config for large model."""
        return cls(weights='yolov8l.pt')
    
    @classmethod
    def xlarge(cls) -> 'ModelConfig':
        """Get config for extra large model (most accurate)."""
        return cls(weights='yolov8x.pt')
