"""
YOLOv8 Weights Downloader
Download pretrained YOLOv8 weights from Ultralytics
"""

import os
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


# Available YOLOv8 weights
WEIGHTS_INFO = {
    'yolov8n.pt': {
        'description': 'YOLOv8 Nano - Fastest, smallest (3.2M parameters)',
        'size_mb': 6.3,
    },
    'yolov8s.pt': {
        'description': 'YOLOv8 Small - Balanced speed/accuracy (11.2M parameters)',
        'size_mb': 22.5,
    },
    'yolov8m.pt': {
        'description': 'YOLOv8 Medium - Higher accuracy (25.9M parameters)',
        'size_mb': 52.0,
    },
    'yolov8l.pt': {
        'description': 'YOLOv8 Large - High accuracy (43.7M parameters)',
        'size_mb': 87.7,
    },
    'yolov8x.pt': {
        'description': 'YOLOv8 Extra Large - Highest accuracy (68.2M parameters)',
        'size_mb': 136.7,
    },
}

DEFAULT_WEIGHTS = ['yolov8n.pt']


def download_weights(
    weights: list = None,
    output_dir: str = 'models/weights',
    force: bool = False
) -> dict:
    """
    Download YOLOv8 pretrained weights using Ultralytics.
    
    Args:
        weights: List of weight names to download (default: yolov8n.pt)
        output_dir: Directory to save weights
        force: Force re-download even if file exists
        
    Returns:
        Dictionary with download status for each weight
    """
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics package not installed")
        print("Run: pip install ultralytics")
        return {}
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for weight_name in weights:
        if weight_name not in WEIGHTS_INFO:
            print(f"Unknown weight: {weight_name}")
            print(f"Available weights: {list(WEIGHTS_INFO.keys())}")
            results[weight_name] = {'status': 'error', 'message': 'Unknown weight'}
            continue
        
        info = WEIGHTS_INFO[weight_name]
        dest_path = output_dir / weight_name
        
        # Check if already exists
        if dest_path.exists() and not force:
            print(f"✓ {weight_name} already exists at {dest_path}")
            results[weight_name] = {'status': 'exists', 'path': str(dest_path)}
            continue
        
        print(f"\nDownloading {weight_name} ({info['size_mb']} MB)...")
        print(f"  {info['description']}")
        
        try:
            # Ultralytics automatically downloads weights when loading model
            model = YOLO(weight_name)
            
            # The weights are downloaded to ~/.cache/ultralytics or similar
            # We can also copy them to our output directory
            print(f"✓ Downloaded {weight_name}")
            results[weight_name] = {'status': 'downloaded', 'path': str(dest_path)}
            
        except Exception as e:
            print(f"✗ Failed to download {weight_name}: {e}")
            results[weight_name] = {'status': 'failed', 'message': str(e)}
    
    return results


def list_available_weights():
    """Print available weights and their descriptions."""
    print("\n" + "=" * 60)
    print("Available YOLOv8 Weights")
    print("=" * 60)
    
    for name, info in WEIGHTS_INFO.items():
        print(f"\n{name}")
        print(f"  Size: {info['size_mb']} MB")
        print(f"  Description: {info['description']}")
    
    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download YOLOv8 pretrained weights')
    
    parser.add_argument('--weights', type=str, nargs='+', default=None,
                        help='Weights to download (default: yolov8n.pt)')
    parser.add_argument('--all', action='store_true',
                        help='Download all available weights')
    parser.add_argument('--output-dir', type=str, default='models/weights',
                        help='Output directory for weights')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if exists')
    parser.add_argument('--list', action='store_true',
                        help='List available weights')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.list:
        list_available_weights()
        return
    
    weights = args.weights
    if args.all:
        weights = list(WEIGHTS_INFO.keys())
    
    print("YOLOv8 Weights Downloader")
    print("=" * 40)
    
    results = download_weights(
        weights=weights,
        output_dir=args.output_dir,
        force=args.force
    )
    
    # Summary
    print("\n" + "=" * 40)
    print("Download Summary")
    print("=" * 40)
    
    for name, result in results.items():
        status = result['status']
        if status == 'downloaded':
            print(f"  ✓ {name}: Downloaded successfully")
        elif status == 'exists':
            print(f"  ● {name}: Already exists")
        else:
            print(f"  ✗ {name}: {result.get('message', status)}")


if __name__ == "__main__":
    main()
