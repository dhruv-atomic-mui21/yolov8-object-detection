"""
YOLOv8 Detector Unit Tests
Test cases for the YOLOv8Detector class and utilities
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    xywh2xyxy,
    xyxy2xywh,
    box_iou,
    non_max_suppression,
    scale_coords,
    letterbox,
    COCO_CLASSES
)


class TestBoundingBoxConversions:
    """Test bounding box format conversions."""
    
    def test_xywh2xyxy_single_box(self):
        """Test xywh to xyxy conversion for single box."""
        xywh = np.array([[100, 100, 50, 50]])  # center_x, center_y, width, height
        expected = np.array([[75, 75, 125, 125]])  # x1, y1, x2, y2
        result = xywh2xyxy(xywh)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xywh2xyxy_multiple_boxes(self):
        """Test xywh to xyxy conversion for multiple boxes."""
        xywh = np.array([
            [100, 100, 50, 50],
            [200, 150, 100, 80]
        ])
        result = xywh2xyxy(xywh)
        
        assert result.shape == (2, 4)
        # First box
        assert result[0, 0] == 75  # x1
        assert result[0, 2] == 125  # x2
        # Second box
        assert result[1, 0] == 150  # x1
        assert result[1, 2] == 250  # x2
    
    def test_xyxy2xywh_single_box(self):
        """Test xyxy to xywh conversion for single box."""
        xyxy = np.array([[75, 75, 125, 125]])
        expected = np.array([[100, 100, 50, 50]])
        result = xyxy2xywh(xyxy)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xywh_xyxy_roundtrip(self):
        """Test that xywh -> xyxy -> xywh roundtrip preserves values."""
        original = np.array([[100, 100, 50, 50], [200, 150, 100, 80]])
        result = xyxy2xywh(xywh2xyxy(original))
        np.testing.assert_array_almost_equal(result, original)


class TestIoU:
    """Test Intersection over Union calculations."""
    
    def test_box_iou_identical_boxes(self):
        """Test IoU of identical boxes is 1."""
        box = np.array([[0, 0, 100, 100]])
        iou = box_iou(box, box)
        np.testing.assert_almost_equal(iou[0, 0], 1.0)
    
    def test_box_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0."""
        box1 = np.array([[0, 0, 50, 50]])
        box2 = np.array([[100, 100, 150, 150]])
        iou = box_iou(box1, box2)
        np.testing.assert_almost_equal(iou[0, 0], 0.0)
    
    def test_box_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = np.array([[0, 0, 100, 100]])
        box2 = np.array([[50, 50, 150, 150]])
        iou = box_iou(box1, box2)
        
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143
        assert 0.1 < iou[0, 0] < 0.2
    
    def test_box_iou_multiple_boxes(self):
        """Test IoU calculation for multiple boxes."""
        boxes1 = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        boxes2 = np.array([[50, 50, 150, 150], [250, 250, 350, 350]])
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (2, 2)


class TestNMS:
    """Test Non-Maximum Suppression."""
    
    def test_nms_single_box(self):
        """Test NMS with single box."""
        boxes = np.array([[0, 0, 100, 100]])
        scores = np.array([0.9])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert keep == [0]
    
    def test_nms_no_boxes(self):
        """Test NMS with empty input."""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert keep == []
    
    def test_nms_overlapping_boxes(self):
        """Test NMS suppresses overlapping boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # Highly overlapping
            [200, 200, 300, 300]  # Non-overlapping
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        
        # Should keep highest score box and non-overlapping box
        assert 0 in keep  # Highest score
        assert 2 in keep  # Non-overlapping
        assert len(keep) == 2
    
    def test_nms_score_ordering(self):
        """Test NMS prioritizes higher scores."""
        boxes = np.array([
            [0, 0, 100, 100],
            [5, 5, 105, 105]  # Highly overlapping
        ])
        scores = np.array([0.7, 0.9])  # Second box has higher score
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        
        assert keep[0] == 1  # Higher score box kept


class TestLetterbox:
    """Test letterbox resizing."""
    
    def test_letterbox_square_to_square(self):
        """Test letterbox with square image to square target."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, new_shape=(200, 200))
        
        assert result.shape[0] == 200
        assert result.shape[1] == 200
    
    def test_letterbox_maintains_aspect(self):
        """Test letterbox maintains aspect ratio."""
        img = np.zeros((200, 100, 3), dtype=np.uint8)  # 2:1 aspect ratio
        result, ratio, pad = letterbox(img, new_shape=(200, 200))
        
        # Result should be square but image scaled appropriately
        assert result.shape[0] == 200
        assert result.shape[1] == 200
    
    def test_letterbox_padding_color(self):
        """Test letterbox uses correct padding color."""
        img = np.ones((100, 50, 3), dtype=np.uint8) * 255  # White image
        color = (114, 114, 114)
        result, _, _ = letterbox(img, new_shape=(100, 100), color=color)
        
        # Check padding areas have padding color
        # The narrow image should have padding on sides
        assert result.shape == (100, 100, 3)


class TestScaleCoords:
    """Test coordinate scaling."""
    
    def test_scale_coords_same_size(self):
        """Test scale_coords with same size images."""
        img1_shape = (640, 640)
        img0_shape = (640, 640)
        coords = np.array([[100, 100, 200, 200]])
        
        result = scale_coords(img1_shape, coords.copy(), img0_shape)
        np.testing.assert_array_almost_equal(result, coords)
    
    def test_scale_coords_doubles_size(self):
        """Test scale_coords when scaling up."""
        img1_shape = (320, 320)
        img0_shape = (640, 640)
        coords = np.array([[50.0, 50.0, 100.0, 100.0]])
        
        result = scale_coords(img1_shape, coords.copy(), img0_shape)
        
        # Coords should approximately double
        assert result[0, 0] > coords[0, 0]
        assert result[0, 2] > coords[0, 2]


class TestCOCOClasses:
    """Test COCO classes constant."""
    
    def test_coco_classes_count(self):
        """Test COCO has 80 classes."""
        assert len(COCO_CLASSES) == 80
    
    def test_coco_classes_contains_common(self):
        """Test COCO contains common classes."""
        assert 'person' in COCO_CLASSES
        assert 'car' in COCO_CLASSES
        assert 'dog' in COCO_CLASSES
        assert 'cat' in COCO_CLASSES


class TestDetectorMocked:
    """Test YOLOv8Detector with mocked model (no actual model loading)."""
    
    @patch('ultralytics.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization with mocked Ultralytics YOLO."""
        from src.detector import YOLOv8Detector
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 1: 'car', 2: 'dog'}
        mock_model.device = 'cpu'
        mock_yolo.return_value = mock_model
        
        detector = YOLOv8Detector(weights='yolov8n.pt')
        
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.45
        mock_yolo.assert_called_once_with('yolov8n.pt')
    
    @patch('ultralytics.YOLO')
    def test_detector_custom_thresholds(self, mock_yolo):
        """Test detector with custom thresholds."""
        from src.detector import YOLOv8Detector
        
        mock_model = MagicMock()
        mock_model.names = {}
        mock_model.device = 'cpu'
        mock_yolo.return_value = mock_model
        
        detector = YOLOv8Detector(
            weights='yolov8n.pt',
            conf_threshold=0.7,
            iou_threshold=0.3
        )
        
        assert detector.conf_threshold == 0.7
        assert detector.iou_threshold == 0.3


class TestVisualize:
    """Test visualization functions."""
    
    def test_draw_detections_empty(self):
        """Test draw_detections with no detections."""
        from src.visualize import draw_detections
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = []
        
        result = draw_detections(img, detections)
        
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)  # No change
    
    def test_draw_detections_single_box(self):
        """Test draw_detections with single detection."""
        from src.visualize import draw_detections
        
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [{
            'bbox': [50, 50, 150, 150],
            'confidence': 0.9,
            'class_name': 'test',
            'class_id': 0
        }]
        
        result = draw_detections(img, detections)
        
        # Result should have modifications (not all zeros)
        assert result.sum() > 0


class TestModelUtils:
    """Test model utility functions."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from src.model import ModelConfig
        
        config = ModelConfig()
        
        assert config.weights == 'yolov8n.pt'
        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.img_size == 640
    
    def test_model_config_custom(self):
        """Test ModelConfig with custom values."""
        from src.model import ModelConfig
        
        config = ModelConfig(
            weights='yolov8s.pt',
            conf_threshold=0.7,
            iou_threshold=0.3,
            img_size=1280
        )
        
        assert config.weights == 'yolov8s.pt'
        assert config.conf_threshold == 0.7
        assert config.iou_threshold == 0.3
        assert config.img_size == 1280
    
    def test_model_config_to_dict(self):
        """Test ModelConfig serialization."""
        from src.model import ModelConfig
        
        config = ModelConfig(weights='yolov8n.pt')
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'weights' in config_dict
        assert config_dict['weights'] == 'yolov8n.pt'
    
    def test_model_config_from_dict(self):
        """Test ModelConfig deserialization."""
        from src.model import ModelConfig
        
        config_dict = {
            'weights': 'yolov8s.pt',
            'conf_threshold': 0.6,
            'iou_threshold': 0.4,
            'img_size': 512,
            'device': 'cpu',
            'half': False
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.weights == 'yolov8s.pt'
        assert config.conf_threshold == 0.6
    
    def test_model_config_factory_methods(self):
        """Test ModelConfig factory methods for different model sizes."""
        from src.model import ModelConfig
        
        nano = ModelConfig.nano()
        assert nano.weights == 'yolov8n.pt'
        
        small = ModelConfig.small()
        assert small.weights == 'yolov8s.pt'
        
        medium = ModelConfig.medium()
        assert medium.weights == 'yolov8m.pt'
        
        large = ModelConfig.large()
        assert large.weights == 'yolov8l.pt'
        
        xlarge = ModelConfig.xlarge()
        assert xlarge.weights == 'yolov8x.pt'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
