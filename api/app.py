"""
YOLOv8 Object Detection API
FastAPI application for real-time object detection using Ultralytics YOLOv8
"""

import io
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLOv8Detector
from api.schemas import (
    DetectionResponse,
    Detection,
    HealthResponse,
    ModelInfoResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection API using Ultralytics YOLOv8",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[YOLOv8Detector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the detector on startup."""
    global detector
    try:
        detector = YOLOv8Detector(
            weights="yolov8n.pt",
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        print("YOLOv8 detector initialized successfully")
    except Exception as e:
        print(f"Warning: Could not load detector on startup: {e}")
        print("Detector will be initialized on first request")


def get_detector() -> YOLOv8Detector:
    """Get or initialize the detector."""
    global detector
    if detector is None:
        detector = YOLOv8Detector(
            weights="yolov8n.pt",
            conf_threshold=0.5,
            iou_threshold=0.45
        )
    return detector


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service info."""
    return HealthResponse(
        status="healthy",
        service="YOLOv8 Object Detection API",
        version="2.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="YOLOv8 Object Detection API",
        version="2.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    try:
        det = get_detector()
        return ModelInfoResponse(
            model_name="YOLOv8",
            device=str(det.model.device),
            conf_threshold=det.conf_threshold,
            iou_threshold=det.iou_threshold,
            num_classes=len(det.classes) if det.classes else 80,
            classes=list(det.classes.values()) if det.classes else []
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    """
    Detect objects in an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **conf_threshold**: Minimum confidence score (0-1)
    - **iou_threshold**: IoU threshold for Non-Maximum Suppression
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get detector and update thresholds
        det = get_detector()
        det.conf_threshold = conf_threshold
        det.iou_threshold = iou_threshold
        
        # Run detection (YOLOv8 handles BGR images)
        results = det.detect(image)
        
        # Convert to response format
        detections = [
            Detection(
                bbox=r['bbox'],
                confidence=r['confidence'],
                class_id=r['class_id'],
                class_name=r['class_name']
            )
            for r in results
        ]
        
        return DetectionResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
            image_width=image.shape[1],
            image_height=image.shape[0]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/visualize")
async def detect_and_visualize(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0)
):
    """
    Detect objects and return annotated image.
    
    Returns the image with bounding boxes and labels drawn.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get detector and update thresholds
        det = get_detector()
        det.conf_threshold = conf_threshold
        det.iou_threshold = iou_threshold
        
        # Get annotated frame using YOLOv8's built-in plotting
        annotated = det.get_annotated_frame(image)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', annotated)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch", response_model=List[DetectionResponse])
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.0, le=1.0)
):
    """
    Detect objects in multiple images.
    
    Maximum 10 images per request.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per request")
    
    responses = []
    det = get_detector()
    det.conf_threshold = conf_threshold
    det.iou_threshold = iou_threshold
    
    for file in files:
        if not file.content_type.startswith("image/"):
            responses.append(DetectionResponse(
                success=False,
                num_detections=0,
                detections=[],
                image_width=0,
                image_height=0,
                error=f"Invalid file type: {file.filename}"
            ))
            continue
        
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                responses.append(DetectionResponse(
                    success=False,
                    num_detections=0,
                    detections=[],
                    image_width=0,
                    image_height=0,
                    error=f"Could not decode: {file.filename}"
                ))
                continue
            
            results = det.detect(image)
            
            detections = [
                Detection(
                    bbox=r['bbox'],
                    confidence=r['confidence'],
                    class_id=r['class_id'],
                    class_name=r['class_name']
                )
                for r in results
            ]
            
            responses.append(DetectionResponse(
                success=True,
                num_detections=len(detections),
                detections=detections,
                image_width=image.shape[1],
                image_height=image.shape[0]
            ))
            
        except Exception as e:
            responses.append(DetectionResponse(
                success=False,
                num_detections=0,
                detections=[],
                image_width=0,
                image_height=0,
                error=str(e)
            ))
    
    return responses


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
