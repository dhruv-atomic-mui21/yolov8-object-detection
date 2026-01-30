"""
YOLOv8 Training Script
Train YOLOv8 on custom datasets using Ultralytics
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def train_yolov8(
    data_yaml: str,
    model: str = 'yolov8n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = 'runs/train',
    name: str = None,
    device: str = None,
    pretrained: bool = True,
    resume: bool = False
):
    """
    Train YOLOv8 on custom dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model: Model to use (yolov8n/s/m/l/x.pt or path to custom weights)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        project: Project directory for saving results
        name: Experiment name (auto-generated if None)
        device: Training device (cuda/cpu/0,1,2,3)
        pretrained: Use pretrained weights
        resume: Resume training from last checkpoint
        
    Returns:
        Training results
    """
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics package not installed")
        print("Run: pip install ultralytics")
        return None
    
    # Generate experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'exp_{timestamp}'
    
    print("\n" + "=" * 50)
    print("YOLOv8 Training")
    print("=" * 50)
    print(f"  Model: {model}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Output: {project}/{name}")
    print("=" * 50 + "\n")
    
    # Load model
    yolo = YOLO(model)
    
    # Train
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        device=device,
        pretrained=pretrained,
        resume=resume,
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Results saved to: {project}/{name}")
    print("=" * 50)
    
    return results


def validate_model(
    model: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = None
):
    """
    Validate YOLOv8 model on dataset.
    
    Args:
        model: Path to model weights
        data_yaml: Path to data.yaml configuration file
        img_size: Input image size
        batch_size: Batch size
        device: Device (cuda/cpu)
        
    Returns:
        Validation metrics
    """
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics package not installed")
        return None
    
    yolo = YOLO(model)
    
    metrics = yolo.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device
    )
    
    return metrics


def export_model(
    model: str,
    format: str = 'onnx',
    img_size: int = 640,
    half: bool = False,
    dynamic: bool = False
):
    """
    Export YOLOv8 model to various formats.
    
    Args:
        model: Path to model weights
        format: Export format (onnx, torchscript, tensorrt, etc.)
        img_size: Input image size
        half: Use FP16 half precision
        dynamic: Use dynamic axes
        
    Returns:
        Path to exported model
    """
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics package not installed")
        return None
    
    yolo = YOLO(model)
    
    export_path = yolo.export(
        format=format,
        imgsz=img_size,
        half=half,
        dynamic=dynamic
    )
    
    print(f"Model exported to: {export_path}")
    return export_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 on custom dataset')
    
    # Data and model
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml configuration file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model (yolov8n/s/m/l/x.pt or path to weights)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/0,1,2,3)')
    
    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
    # Mode
    parser.add_argument('--val', action='store_true',
                        help='Run validation only')
    parser.add_argument('--export', type=str, default=None,
                        help='Export model to format (onnx, torchscript, etc.)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.val:
        # Validation mode
        print("Running validation...")
        validate_model(
            model=args.model,
            data_yaml=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )
    elif args.export:
        # Export mode
        print(f"Exporting model to {args.export}...")
        export_model(
            model=args.model,
            format=args.export,
            img_size=args.img_size
        )
    else:
        # Training mode
        train_yolov8(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            project=args.project,
            name=args.name,
            device=args.device,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
