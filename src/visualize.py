"""
YOLOv8 Visualization Functions
Drawing and plotting utilities for object detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


def draw_detections(
    image: Union[str, np.ndarray],
    detections: List[Dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw detection boxes and labels on image.
    
    Args:
        image: Image path or numpy array (BGR format)
        detections: List of detection dicts with 'bbox', 'class_name', 'confidence'
        class_colors: Optional dict mapping class names to BGR colors
        thickness: Box line thickness
        font_scale: Font scale for labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        Annotated image as numpy array
    """
    # Load image if path
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        class_id = det.get('class_id', 0)
        
        # Get color
        if class_colors and class_name in class_colors:
            color = class_colors[class_name]
        else:
            # Generate consistent color from class_id
            np.random.seed(class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Create label
        if show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    return img


def create_detection_video(
    input_path: str,
    output_path: str,
    detector,
    show_progress: bool = True,
    fps: Optional[int] = None
) -> str:
    """
    Create annotated video with detections.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        detector: YOLOv7Detector instance
        show_progress: Whether to show processing progress
        fps: Override FPS (None to keep original)
        
    Returns:
        Path to output video
    """
    from tqdm import tqdm
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = fps if fps else original_fps
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Process frames
    iterator = range(total_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing video")
    
    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb)
        
        # Draw detections
        annotated = draw_detections(frame, detections)
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    print(f"Video saved to {output_path}")
    return output_path


def plot_results(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create summary plot of detection results.
    
    Args:
        detections: List of detection dictionaries
        image_shape: Original image shape (height, width)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Class distribution
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[0].barh(classes, counts, color='steelblue')
        axes[0].set_xlabel('Count')
        axes[0].set_title('Detected Objects by Class')
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14)
        axes[0].set_title('Detected Objects by Class')
    
    # Confidence distribution
    if detections:
        confidences = [det['confidence'] for det in detections]
        axes[1].hist(confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Confidence Distribution')
        axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.2f}')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14)
        axes[1].set_title('Confidence Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_detection_grid(
    images: List[np.ndarray],
    detections_list: List[List[Dict]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    max_cols: int = 3,
    figsize_per_image: Tuple[int, int] = (5, 5)
) -> plt.Figure:
    """
    Create a grid of detection results.
    
    Args:
        images: List of images (BGR format numpy arrays)
        detections_list: List of detection lists for each image
        titles: Optional titles for each image
        save_path: Optional path to save figure
        max_cols: Maximum columns in grid
        figsize_per_image: Figure size per image
        
    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    figsize = (figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_images == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        if idx < n_images:
            # Draw detections on image
            annotated = draw_detections(images[idx], detections_list[idx])
            # Convert BGR to RGB for matplotlib
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            ax.imshow(annotated_rgb)
            
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            else:
                n_det = len(detections_list[idx])
                ax.set_title(f"Image {idx + 1} ({n_det} detections)")
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grid saved to {save_path}")
    
    return fig


def create_comparison_image(
    original: np.ndarray,
    detected: np.ndarray,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of original and detected images.
    
    Args:
        original: Original image (BGR)
        detected: Image with detections (BGR)
        save_path: Optional path to save image
        
    Returns:
        Combined comparison image
    """
    # Ensure same height
    h1, w1 = original.shape[:2]
    h2, w2 = detected.shape[:2]
    
    if h1 != h2:
        # Resize to match height
        detected = cv2.resize(detected, (int(w2 * h1 / h2), h1))
    
    # Add labels
    original_labeled = original.copy()
    detected_labeled = detected.copy()
    
    cv2.putText(original_labeled, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(detected_labeled, "Detected", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Concatenate horizontally
    comparison = np.hstack([original_labeled, detected_labeled])
    
    if save_path:
        cv2.imwrite(save_path, comparison)
        print(f"Comparison saved to {save_path}")
    
    return comparison


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on image.
    
    Args:
        image: Image to draw on
        fps: Frames per second value
        
    Returns:
        Image with FPS counter
    """
    img = image.copy()
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return img


def generate_heatmap(
    image_shape: Tuple[int, int],
    detections: List[Dict],
    kernel_size: int = 50
) -> np.ndarray:
    """
    Generate heatmap from detection centers.
    
    Args:
        image_shape: Shape of image (height, width)
        detections: List of detection dictionaries
        kernel_size: Gaussian kernel size
        
    Returns:
        Heatmap as numpy array
    """
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Ensure within bounds
        cy = min(max(0, cy), image_shape[0] - 1)
        cx = min(max(0, cx), image_shape[1] - 1)
        
        heatmap[cy, cx] += 1
    
    # Apply Gaussian blur
    if kernel_size > 0:
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap
