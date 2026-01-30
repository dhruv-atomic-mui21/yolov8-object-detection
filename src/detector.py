"""
YOLOv8 Object Detector
Main detector class for real-time object detection using Ultralytics YOLOv8
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from ultralytics import YOLO


class YOLOv8Detector:
    """
    YOLOv8 Object Detection wrapper class using Ultralytics.
    
    Attributes:
        model: Loaded YOLOv8 model
        device: Computing device (cuda/cpu)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        classes: Dictionary of class names
    """
    
    def __init__(
        self,
        weights: str = 'yolov8n.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            weights: Model weights ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        print(f"Loading YOLOv8 model: {weights}")
        self.model = YOLO(weights)
        
        # Set device if specified
        if device:
            self.model.to(device)
        
        # Get class names
        self.classes = self.model.names
        
        print(f"Model loaded successfully!")
        print(f"  Device: {self.model.device}")
        print(f"  Classes: {len(self.classes)}")
            
    def detect(
        self,
        source: Union[str, np.ndarray],
        img_size: int = 640
    ) -> List[Dict]:
        """
        Perform object detection on image.
        
        Args:
            source: Image path or numpy array
            img_size: Inference image size
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
            - class_name: str
        """
        # Run inference
        results = self.model(
            source, 
            imgsz=img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    'bbox': [int(x) for x in xyxy],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.classes[cls_id]
                })
            
        return detections
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        img_size: int = 640
    ) -> List[List[Dict]]:
        """Perform detection on multiple images."""
        return [self.detect(img, img_size) for img in images]
    
    def visualize(
        self,
        image: Union[str, np.ndarray],
        detections: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Image path or numpy array
            detections: List of detection dictionaries (if None, runs detection first)
            save_path: Optional path to save visualization
            show: Whether to display the image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Run detection if no detections provided
        if detections is None:
            detections = self.detect(img)
            
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            color = self._get_color(det['class_id'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Saved to {save_path}")
            
        # Show image
        if show:
            cv2.imshow('YOLOv8 Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return img
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID."""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def detect_video(
        self,
        video_path: Union[str, int],
        output_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Perform detection on video file or webcam.
        
        Args:
            video_path: Path to input video or webcam index (0, 1, etc.)
            output_path: Optional path to save output video
            show: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
                
            # Detect objects
            detections = self.detect(frame)
            
            # Visualize
            annotated = self.visualize(frame, detections, show=False)
            
            if writer:
                writer.write(annotated)
                
            if show:
                cv2.imshow('YOLOv8 Video Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        print(f"Processed {frame_count} frames")
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    def get_annotated_frame(self, source: Union[str, np.ndarray]) -> np.ndarray:
        """
        Get annotated frame using YOLOv8's built-in plotting.
        
        Args:
            source: Image path or numpy array
            
        Returns:
            Annotated image as numpy array (BGR format)
        """
        results = self.model(
            source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Use YOLOv8's built-in plot method
        annotated = results[0].plot()
        return annotated


# Backward compatibility alias
YOLOv7Detector = YOLOv8Detector


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8Detector(weights='yolov8n.pt', conf_threshold=0.5)
    
    # Detect on sample image
    results = detector.detect('sample.jpg')
    print(f"Found {len(results)} objects")
    
    for det in results:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
