"""
Spatial Analysis & Geometric Clearance Module
Manages 5-zone angular segmentation, monocular distance estimation, and path clearance.
"""

import numpy as np


class SpatialZoneManager:
    """Defines and computes spatial zone allocations and geometric positioning."""
    
    DEFAULT_ZONES = {
        'center': {'angle_range': (-15, 15), 'description': 'Direct path'},
        'left': {'angle_range': (-45, -15), 'description': 'Left turn zone'},
        'right': {'angle_range': (15, 45), 'description': 'Right turn zone'},
        'far_left': {'angle_range': (-90, -45), 'description': 'Sharp left zone'},
        'far_right': {'angle_range': (45, 90), 'description': 'Sharp right zone'}
    }
    
    REAL_HEIGHTS = {
        'person': 1.7,
        'car': 1.5,
        'truck': 3.5,
        'bus': 3.2,
        'bicycle': 1.2,
        'motorcycle': 1.3,
        'traffic light': 3.0,
        'stop sign': 2.5
    }
    
    def __init__(self, focal_length=700, zones=None):
        self.focal_length = focal_length
        self.zones = zones or self.DEFAULT_ZONES
    
    def estimate_real_distance(self, bbox, image_width, image_height, object_class):
        """
        Estimate Euclidean distance from camera using bounding box geometry and perspective.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_width: Width of image in pixels
            image_height: Height of image in pixels
            object_class: Class label string
            
        Returns:
            float: Estimated distance in meters (clipped 5 to 200m)
        """
        bbox_height = bbox[3] - bbox[1]
        real_height = self.REAL_HEIGHTS.get(object_class, 1.7)
        
        if bbox_height > 0:
            estimated_distance = (real_height * self.focal_length) / bbox_height
        else:
            estimated_distance = 100.0
        
        # Adjust based on vertical position (objects lower in frame are closer to ground contact)
        bbox_center_y = (bbox[1] + bbox[3]) / 2.0
        vertical_ratio = bbox_center_y / float(image_height)
        
        if vertical_ratio > 0.6:
            estimated_distance *= 0.8
        elif vertical_ratio < 0.3:
            estimated_distance *= 1.3
        
        return float(np.clip(estimated_distance, 5.0, 200.0))
    
    def calculate_lateral_position(self, bbox, image_width):
        """
        Calculate angular lateral deviation in degrees (-45° to +45° within typical FOV).
        """
        bbox_center_x = (bbox[0] + bbox[2]) / 2.0
        image_center_x = image_width / 2.0
        return ((bbox_center_x - image_center_x) / (image_width / 2.0)) * 45.0
    
    def get_zone_from_angle(self, angle):
        """Map lateral angle to corresponding discrete spatial zone."""
        for zone_name, zone_info in self.zones.items():
            if zone_info['angle_range'][0] <= angle <= zone_info['angle_range'][1]:
                return zone_name
        return 'center'


class PathClearanceAnalyzer:
    """Evaluates multi-zone path clearance and obstacle proximities."""
    
    def __init__(self, zone_manager, nav_objects):
        self.zone_manager = zone_manager
        self.nav_objects = nav_objects
    
    def analyze_path_clearance(self, detections):
        """
        Analyze clearance status across all 5 spatial zones.
        
        Returns:
            dict: Zone states with clearance status, obstacles list, and minimum distance.
        """
        path_analysis = {
            'center': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'left': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'right': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'far_left': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'far_right': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')}
        }
        
        for det in detections:
            zone = self.zone_manager.get_zone_from_angle(det['lateral_position'])
            distance = det['distance']
            
            obj_info = self.nav_objects.get(det['class'], {})
            caution_distance = obj_info.get('caution_distance', 30)
            stop_distance = obj_info.get('stop_distance', 10)
            
            if distance < caution_distance:
                path_analysis[zone]['obstacles'].append(det)
                if distance < path_analysis[zone]['closest_distance']:
                    path_analysis[zone]['closest_distance'] = distance
                
                if distance < stop_distance:
                    path_analysis[zone]['clear'] = False
        
        return path_analysis
