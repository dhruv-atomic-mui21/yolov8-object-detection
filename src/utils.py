"""
YOLOv8 Utility Functions
Bounding box operations, NMS, and preprocessing utilities
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (x_center, y_center, width, height) to 
    (x1, y1, x2, y2) format.
    
    Args:
        boxes: Array of shape (N, 4) in xywh format
        
    Returns:
        Array of shape (N, 4) in xyxy format
    """
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to 
    (x_center, y_center, width, height) format.
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        
    Returns:
        Array of shape (N, 4) in xywh format
    """
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return xywh


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        box1: Array of shape (N, 4) in xyxy format
        box2: Array of shape (M, 4) in xyxy format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Intersection area
    inter_x1 = np.maximum(x1_1, x1_2)
    inter_y1 = np.maximum(y1_1, y1_2)
    inter_x2 = np.minimum(x2_1, x2_2)
    inter_y2 = np.minimum(y2_1, y2_2)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    return iou


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Perform Non-Maximum Suppression on detection boxes.
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices of kept boxes
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Keep the highest scoring box
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calculate IoU with remaining boxes
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = iou < iou_threshold
        order = order[1:][mask]
    
    return keep


def scale_coords(
    img1_shape: Tuple[int, int],
    coords: np.ndarray,
    img0_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale coordinates from img1_shape to img0_shape.
    
    Args:
        img1_shape: Shape of resized image (height, width)
        coords: Coordinates to rescale (N, 4) in xyxy format
        img0_shape: Shape of original image (height, width)
        
    Returns:
        Rescaled coordinates
    """
    # Calculate gain and pad
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    # Remove padding
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    
    # Rescale
    coords[:, :4] /= gain
    
    # Clip to image bounds
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
    
    return coords


def load_classes(path: str) -> List[str]:
    """
    Load class names from file.
    
    Args:
        path: Path to file with class names (one per line)
        
    Returns:
        List of class names
    """
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image to new_shape while maintaining aspect ratio.
    
    Args:
        img: Input image (HWC format)
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill
        scale_up: Allow scale up
        stride: Stride for padding alignment
        
    Returns:
        Tuple of (resized image, ratio, padding)
    """
    shape = img.shape[:2]  # Current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0, 0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


def preprocess_image(
    image: Union[str, np.ndarray],
    img_size: int = 640,
    stride: int = 32
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Preprocess image for YOLOv7 inference.
    
    Args:
        image: Image path or numpy array (BGR format)
        img_size: Target image size
        stride: Model stride
        
    Returns:
        Tuple of (preprocessed tensor, original image, original shape)
    """
    # Load image if path
    if isinstance(image, str):
        img0 = cv2.imread(image)
    else:
        img0 = image
    
    original_shape = img0.shape[:2]
    
    # Letterbox resize
    img, _, _ = letterbox(img0, new_shape=img_size, stride=stride)
    
    # Convert BGR to RGB and transpose to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    
    # Convert to tensor and normalize
    img = torch.from_numpy(img).float() / 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img, img0, original_shape


def clip_coords(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding box coordinates to image boundaries.
    
    Args:
        boxes: Boxes in xyxy format (N, 4)
        img_shape: Image shape (height, width)
        
    Returns:
        Clipped boxes
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])
    return boxes


def get_color_palette(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """
    Generate a color palette for visualization.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        List of RGB color tuples
    """
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    return colors


# COCO class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
