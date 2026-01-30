"""
YOLOv7 Data Preparation Script
Prepare and convert datasets to YOLO format
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml


class DatasetConverter:
    """Convert various dataset formats to YOLO format."""
    
    def __init__(self, output_dir: str = 'data/processed'):
        """
        Initialize converter.
        
        Args:
            output_dir: Output directory for converted dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_coco_to_yolo(
        self,
        annotations_file: str,
        images_dir: str,
        output_name: str = 'coco'
    ) -> str:
        """
        Convert COCO format to YOLO format.
        
        Args:
            annotations_file: Path to COCO annotations JSON
            images_dir: Path to images directory
            output_name: Name for output subdirectory
            
        Returns:
            Path to output directory
        """
        print(f"Converting COCO format to YOLO...")
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
        category_names = [cat['name'] for cat in coco['categories']]
        
        # Create image id to annotations mapping
        img_annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Create output structure
        output_path = self.output_dir / output_name
        images_out = output_path / 'images'
        labels_out = output_path / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_info in tqdm(coco['images'], desc="Converting"):
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_path = Path(images_dir) / img_filename
            if src_path.exists():
                shutil.copy(src_path, images_out / img_filename)
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            if img_id in img_annotations:
                for ann in img_annotations[img_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    cat_id = ann['category_id']
                    
                    # Convert to YOLO format (center_x, center_y, width, height) normalized
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    w = bbox[2] / img_width
                    h = bbox[3] / img_height
                    
                    class_id = categories[cat_id]
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Write label file
            label_filename = Path(img_filename).stem + '.txt'
            with open(labels_out / label_filename, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        # Save class names
        with open(output_path / 'classes.txt', 'w') as f:
            f.write('\n'.join(category_names))
        
        print(f"Conversion complete! Output: {output_path}")
        return str(output_path)
    
    def convert_voc_to_yolo(
        self,
        voc_dir: str,
        output_name: str = 'voc'
    ) -> str:
        """
        Convert Pascal VOC format to YOLO format.
        
        Args:
            voc_dir: Path to VOC dataset directory
            output_name: Name for output subdirectory
            
        Returns:
            Path to output directory
        """
        print(f"Converting VOC format to YOLO...")
        
        voc_path = Path(voc_dir)
        annotations_dir = voc_path / 'Annotations'
        images_dir = voc_path / 'JPEGImages'
        
        # Create output structure
        output_path = self.output_dir / output_name
        images_out = output_path / 'images'
        labels_out = output_path / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        # Get all classes
        classes = set()
        for xml_file in annotations_dir.glob('*.xml'):
            tree = ET.parse(xml_file)
            for obj in tree.findall('.//object'):
                classes.add(obj.find('name').text)
        
        classes = sorted(list(classes))
        class_to_id = {name: idx for idx, name in enumerate(classes)}
        
        # Process each annotation
        for xml_file in tqdm(list(annotations_dir.glob('*.xml')), desc="Converting"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Copy image
            src_path = images_dir / filename
            if src_path.exists():
                shutil.copy(src_path, images_out / filename)
            
            # Convert annotations
            yolo_annotations = []
            for obj in root.findall('.//object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                w = (xmax - xmin) / img_width
                h = (ymax - ymin) / img_height
                
                class_id = class_to_id[class_name]
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Write label file
            label_filename = xml_file.stem + '.txt'
            with open(labels_out / label_filename, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        # Save class names
        with open(output_path / 'classes.txt', 'w') as f:
            f.write('\n'.join(classes))
        
        print(f"Conversion complete! Output: {output_path}")
        return str(output_path)


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Path to dataset directory with images/ and labels/
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with lists of filenames for each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
    
    # Get all image files
    image_files = list(images_dir.glob('*.[jJ][pP][gG]')) + \
                  list(images_dir.glob('*.[pP][nN][gG]'))
    
    # Filter to only images with labels
    valid_files = []
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            valid_files.append(img_file.name)
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(valid_files)
    
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': valid_files[:n_train],
        'val': valid_files[n_train:n_train + n_val],
        'test': valid_files[n_train + n_val:]
    }
    
    # Create split directories
    for split_name, files in splits.items():
        split_images = data_path / split_name / 'images'
        split_labels = data_path / split_name / 'labels'
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)
        
        for filename in files:
            # Copy image
            shutil.copy(images_dir / filename, split_images / filename)
            # Copy label
            label_name = Path(filename).stem + '.txt'
            if (labels_dir / label_name).exists():
                shutil.copy(labels_dir / label_name, split_labels / label_name)
    
    print(f"\nDataset split complete:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    
    return splits


def create_data_yaml(
    data_dir: str,
    output_path: str = 'configs/custom.yaml'
) -> str:
    """
    Create data configuration YAML for YOLOv7.
    
    Args:
        data_dir: Path to processed dataset
        output_path: Path for output YAML file
        
    Returns:
        Path to created YAML file
    """
    data_path = Path(data_dir)
    
    # Load class names
    classes_file = data_path / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = ['object']  # Default
    
    # Create config
    config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    # Write YAML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Data config saved to: {output_path}")
    return str(output_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLOv7 training')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to input dataset directory')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--format', type=str, choices=['coco', 'voc', 'yolo'],
                        default='yolo', help='Input dataset format')
    
    # Split options
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    # COCO specific
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to COCO annotations JSON')
    
    return parser.parse_args()


def main():
    """Main data preparation function."""
    args = parse_args()
    
    print("=" * 50)
    print("YOLOv7 Data Preparation")
    print("=" * 50)
    
    converter = DatasetConverter(output_dir=args.output_dir)
    
    # Convert dataset if needed
    if args.format == 'coco':
        if not args.annotations:
            raise ValueError("--annotations required for COCO format")
        output_dir = converter.convert_coco_to_yolo(
            annotations_file=args.annotations,
            images_dir=args.data_dir
        )
    elif args.format == 'voc':
        output_dir = converter.convert_voc_to_yolo(voc_dir=args.data_dir)
    else:
        output_dir = args.data_dir
    
    # Split dataset
    print("\nSplitting dataset...")
    split_dataset(
        data_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create data config
    print("\nCreating data configuration...")
    create_data_yaml(data_dir=output_dir)
    
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
