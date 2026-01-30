"""
YOLOv8 Object Detection - Main Entry Point
Run object detection on images, videos, or webcam using Ultralytics YOLOv8
"""

import argparse
from pathlib import Path
from src.detector import YOLOv8Detector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Image/video path or webcam index (0)')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Model weights (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Loading YOLOv8 model...")
    detector = YOLOv8Detector(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    source = args.source
    
    # Check if webcam
    if source.isdigit():
        print("Starting webcam detection...")
        detector.detect_video(
            int(source),
            output_path=str(output_dir / 'webcam_output.mp4'),
            show=not args.no_show
        )
    # Check if video
    elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"Processing video: {source}")
        output_path = output_dir / f"{Path(source).stem}_detected.mp4"
        detector.detect_video(
            source,
            output_path=str(output_path),
            show=not args.no_show
        )
        print(f"Output saved to: {output_path}")
    # Image
    else:
        print(f"Processing image: {source}")
        detections = detector.detect(source, img_size=args.img_size)
        
        print(f"\nFound {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        # Save visualization
        output_path = output_dir / f"{Path(source).stem}_detected.jpg"
        detector.visualize(
            source,
            detections,
            save_path=str(output_path),
            show=not args.no_show
        )
        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
