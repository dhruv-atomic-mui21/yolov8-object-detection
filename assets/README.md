# Assets Directory

This directory contains demo assets and visualizations.

## Structure

```
assets/
├── README.md          # This file
├── demo.gif           # Demo animation (add your own)
└── samples/           # Sample images for testing
```

## Adding Demo Assets

1. Create a demo GIF showing object detection in action
2. Add sample images for testing
3. Store any visualizations here

## Generating Demo GIF

You can generate a demo GIF using the following script:

```python
import cv2
import imageio
from src.detector import YOLOv7Detector

# Initialize detector
detector = YOLOv7Detector(weights='yolov7.pt')

# Process video frames
cap = cv2.VideoCapture('input_video.mp4')
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and visualize
    detections = detector.detect(frame)
    annotated = detector.visualize(frame, detections, show=False)
    
    # Convert BGR to RGB
    frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

cap.release()

# Save as GIF
imageio.mimsave('assets/demo.gif', frames, fps=10)
```
