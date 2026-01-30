# YOLOv7 Object Detection - Exploration Notebook

This notebook provides interactive exploration and demonstration of the YOLOv7 object detection system.

## Table of Contents

1. [Setup and Installation](#setup)
2. [Loading the Model](#loading-model)
3. [Image Detection](#image-detection)
4. [Video Detection](#video-detection)
5. [Performance Benchmarking](#benchmarking)
6. [Visualization Examples](#visualization)

---

## 1. Setup and Installation <a name="setup"></a>

```python
# Install dependencies (run in first cell)
# !pip install torch torchvision opencv-python matplotlib

import sys
sys.path.append('..')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Set matplotlib inline
%matplotlib inline

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 2. Loading the Model <a name="loading-model"></a>

```python
from src.detector import YOLOv7Detector
from src.model import get_model_info, ModelConfig

# Initialize detector
detector = YOLOv7Detector(
    weights='yolov7.pt',
    conf_threshold=0.5,
    iou_threshold=0.45
)

# Print model info
info = get_model_info(detector.model)
print("Model Information:")
for key, value in info.items():
    if key != 'class_names':
        print(f"  {key}: {value}")
```

---

## 3. Image Detection <a name="image-detection"></a>

```python
# Load sample image
image_path = 'path/to/your/image.jpg'

# Or use a sample image URL
import requests
from io import BytesIO
from PIL import Image

# Download sample image
url = "https://ultralytics.com/images/zidane.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img_array = np.array(img)

# Run detection
detections = detector.detect(img_array)

# Print results
print(f"Found {len(detections)} objects:")
for det in detections:
    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    print(f"    Bounding box: {det['bbox']}")
```

```python
# Visualize detections
from src.visualize import draw_detections, plot_results

# Draw bounding boxes
annotated = draw_detections(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), detections)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(annotated_rgb)
plt.axis('off')
plt.title(f'YOLOv7 Detection - {len(detections)} objects found')
plt.show()
```

---

## 4. Video Detection <a name="video-detection"></a>

```python
# Process a video file
video_path = 'path/to/video.mp4'
output_path = 'output_detected.mp4'

# Note: This will process the entire video
# For demo purposes, you might want to limit frames

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {video_path}")
print(f"  FPS: {fps}")
print(f"  Total frames: {frame_count}")
print(f"  Duration: {frame_count/fps:.1f} seconds")

cap.release()

# Process video (uncomment to run)
# from src.visualize import create_detection_video
# create_detection_video(video_path, output_path, detector)
```

---

## 5. Performance Benchmarking <a name="benchmarking"></a>

```python
# Benchmark inference speed
num_runs = 100

# Create random test image
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warm-up
for _ in range(10):
    _ = detector.detect(test_image)

# Benchmark
times = []
for _ in range(num_runs):
    start = time.time()
    _ = detector.detect(test_image)
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000  # Convert to ms
std_time = np.std(times) * 1000
fps = 1000 / avg_time

print(f"Benchmark Results ({num_runs} runs):")
print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"  FPS: {fps:.1f}")
print(f"  Device: {detector.device}")
```

```python
# Plot inference time distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(np.array(times) * 1000, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Inference Time (ms)')
plt.ylabel('Count')
plt.title('Inference Time Distribution')
plt.axvline(avg_time, color='red', linestyle='--', label=f'Mean: {avg_time:.1f}ms')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.array(times) * 1000)
plt.xlabel('Run')
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time Over Runs')

plt.tight_layout()
plt.show()
```

---

## 6. Visualization Examples <a name="visualization"></a>

```python
# Multi-image detection grid
from src.visualize import plot_detection_grid

# Load multiple images
images = []
all_detections = []

sample_urls = [
    "https://ultralytics.com/images/zidane.jpg",
    "https://ultralytics.com/images/bus.jpg"
]

for url in sample_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    images.append(img_array)
    
    detections = detector.detect(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    all_detections.append(detections)

# Create grid
fig = plot_detection_grid(images, all_detections, figsize_per_image=(6, 6))
plt.show()
```

```python
# Detection statistics
from src.visualize import plot_results

# Combine all detections
combined_detections = [det for dets in all_detections for det in dets]

fig = plot_results(
    combined_detections,
    image_shape=(640, 640),
    figsize=(14, 5)
)
plt.show()
```

---

## Notes

- For best performance, use a CUDA-enabled GPU
- Adjust `conf_threshold` based on your use case:
  - Higher (0.7-0.9): Fewer false positives, may miss some objects
  - Lower (0.3-0.5): More detections, but more false positives
- The model supports 80 COCO classes by default
