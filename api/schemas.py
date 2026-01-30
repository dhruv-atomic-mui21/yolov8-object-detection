"""
YOLOv8 API Schemas
Pydantic models for request/response validation
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single detection result."""
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., ge=0, description="Class ID")
    class_name: str = Field(..., description="Class name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [100, 150, 300, 400],
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "person"
            }
        }


class DetectionResponse(BaseModel):
    """Response for detection endpoint."""
    success: bool = Field(..., description="Whether detection was successful")
    num_detections: int = Field(..., ge=0, description="Number of detections")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    image_width: int = Field(..., ge=0, description="Original image width")
    image_height: int = Field(..., ge=0, description="Original image height")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "num_detections": 3,
                "detections": [
                    {
                        "bbox": [100, 150, 300, 400],
                        "confidence": 0.95,
                        "class_id": 0,
                        "class_name": "person"
                    }
                ],
                "image_width": 1920,
                "image_height": 1080,
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "YOLOv8 Object Detection API",
                "version": "2.0.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response for model information endpoint."""
    model_name: str = Field(..., description="Model name")
    device: str = Field(..., description="Computing device")
    conf_threshold: float = Field(..., description="Confidence threshold")
    iou_threshold: float = Field(..., description="IoU threshold")
    num_classes: int = Field(..., description="Number of classes")
    classes: List[str] = Field(default_factory=list, description="Class names")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "YOLOv8",
                "device": "cuda",
                "conf_threshold": 0.5,
                "iou_threshold": 0.45,
                "num_classes": 80,
                "classes": ["person", "bicycle", "car"]
            }
        }


class DetectionRequest(BaseModel):
    """Request body for detection (when using base64 encoded images)."""
    image_base64: str = Field(..., description="Base64 encoded image")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
                "conf_threshold": 0.5,
                "iou_threshold": 0.45
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request body for batch detection."""
    images_base64: List[str] = Field(..., max_length=10, description="List of base64 encoded images")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")


class VideoDetectionRequest(BaseModel):
    """Request for video detection."""
    video_url: str = Field(..., description="URL to video file")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    max_frames: int = Field(100, ge=1, le=1000, description="Maximum frames to process")
