import cv2
import numpy as np
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pathlib import Path
import json
import math
import urllib.request
from collections import defaultdict

class RealDatasetDownloader:
    """Download real traffic and pedestrian navigation dataset"""
    
    def __init__(self):
        self.dataset_urls = [
            "http://farm4.staticflickr.com/3666/10277303256_a6a11a9d4b_z.jpg",
            "http://farm3.staticflickr.com/2538/4236286875_05b2e96ec4_z.jpg",
            "http://farm7.staticflickr.com/6141/6022871891_a601326786_z.jpg",
            "http://images.cocodataset.org/val2017/000000001268.jpg",
            "http://farm9.staticflickr.com/8118/8965896602_c68fe611bd_z.jpg",
            "http://farm8.staticflickr.com/7073/7345527746_6b25ae7ac1_z.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000001584.jpg",
            "http://images.cocodataset.org/val2017/000000002006.jpg",
        ]
    
    def download_dataset(self, output_dir):
        print("\n" + "="*80)
        print("DOWNLOADING TRAFFIC & PEDESTRIAN NAVIGATION DATASET")
        print("="*80)
        
        image_dir = Path(output_dir) / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, url in enumerate(self.dataset_urls, 1):
            filename = url.split('/')[-1]
            output_path = image_dir / filename
            
            if output_path.exists():
                print(f"[{idx}/{len(self.dataset_urls)}] ✓ Already downloaded: {filename}")
                continue
            
            try:
                print(f"[{idx}/{len(self.dataset_urls)}] ⬇ Downloading: {filename}")
                urllib.request.urlretrieve(url, output_path)
                print(f"    ✓ Saved successfully")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print("\n✓ Dataset ready for processing!")
        return image_dir


class EnhancedNavigationSystem:
    """
    Advanced navigation system with precise turn opportunity detection
    Assumes user is traveling straight and provides context-aware guidance
    """
    
    def __init__(self):
        print("\n" + "="*80)
        print("INITIALIZING ENHANCED CONTEXT-AWARE NAVIGATION SYSTEM")
        print("="*80)
        
        print("\n[1/4] Loading YOLOv8 detection model...")
        self.model = YOLO('yolov8n.pt')
        print("      ✓ YOLO model loaded")
        
        # Enhanced object classification with spatial zones
        self.nav_objects = {
            'traffic light': {'priority': 10, 'type': 'signal', 'stop_distance': 15, 'caution_distance': 50},
            'stop sign': {'priority': 10, 'type': 'signal', 'stop_distance': 10, 'caution_distance': 30},
            'car': {'priority': 8, 'type': 'vehicle', 'stop_distance': 8, 'caution_distance': 25},
            'truck': {'priority': 8, 'type': 'vehicle', 'stop_distance': 10, 'caution_distance': 30},
            'bus': {'priority': 8, 'type': 'vehicle', 'stop_distance': 10, 'caution_distance': 30},
            'person': {'priority': 10, 'type': 'pedestrian', 'stop_distance': 5, 'caution_distance': 35},
            'bicycle': {'priority': 7, 'type': 'vehicle', 'stop_distance': 6, 'caution_distance': 20},
            'motorcycle': {'priority': 7, 'type': 'vehicle', 'stop_distance': 7, 'caution_distance': 20},
        }
        
        print("\n[2/4] Setting up enhanced fuzzy logic system...")
        self.setup_advanced_fuzzy_system()
        print("      ✓ Fuzzy inference engine ready")
        
        print("\n[3/4] Initializing spatial analysis zones...")
        self.setup_spatial_zones()
        print("      ✓ Spatial zones configured")
        
        print("\n[4/4] Navigation system initialized successfully!")
        
    def setup_spatial_zones(self):
        """Define spatial zones for turn opportunity detection"""
        self.zones = {
            'center': {'angle_range': (-15, 15), 'description': 'Direct path'},
            'left': {'angle_range': (-45, -15), 'description': 'Left turn zone'},
            'right': {'angle_range': (15, 45), 'description': 'Right turn zone'},
            'far_left': {'angle_range': (-90, -45), 'description': 'Sharp left zone'},
            'far_right': {'angle_range': (45, 90), 'description': 'Sharp right zone'}
        }
        
    def setup_advanced_fuzzy_system(self):
        """Setup comprehensive fuzzy logic system with enhanced rules"""
        
        # INPUT VARIABLES
        distance = ctrl.Antecedent(np.arange(0, 201, 1), 'distance')
        distance['immediate'] = fuzz.trapmf(distance.universe, [0, 0, 8, 15])
        distance['very_close'] = fuzz.trimf(distance.universe, [10, 20, 35])
        distance['close'] = fuzz.trimf(distance.universe, [30, 50, 70])
        distance['medium'] = fuzz.trimf(distance.universe, [60, 90, 120])
        distance['far'] = fuzz.trapmf(distance.universe, [110, 150, 200, 200])
        
        confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
        confidence['very_low'] = fuzz.trapmf(confidence.universe, [0, 0, 0.2, 0.35])
        confidence['low'] = fuzz.trimf(confidence.universe, [0.25, 0.4, 0.55])
        confidence['medium'] = fuzz.trimf(confidence.universe, [0.45, 0.65, 0.8])
        confidence['high'] = fuzz.trimf(confidence.universe, [0.7, 0.85, 0.95])
        confidence['very_high'] = fuzz.trapmf(confidence.universe, [0.9, 0.95, 1.0, 1.0])
        
        priority = ctrl.Antecedent(np.arange(0, 11, 1), 'priority')
        priority['low'] = fuzz.trimf(priority.universe, [0, 3, 6])
        priority['medium'] = fuzz.trimf(priority.universe, [4, 7, 9])
        priority['high'] = fuzz.trapmf(priority.universe, [8, 9.5, 10, 10])
        
        lateral_position = ctrl.Antecedent(np.arange(-90, 91, 1), 'lateral_position')
        lateral_position['far_left'] = fuzz.trapmf(lateral_position.universe, [-90, -90, -60, -40])
        lateral_position['left'] = fuzz.trimf(lateral_position.universe, [-50, -30, -10])
        lateral_position['center'] = fuzz.trimf(lateral_position.universe, [-20, 0, 20])
        lateral_position['right'] = fuzz.trimf(lateral_position.universe, [10, 30, 50])
        lateral_position['far_right'] = fuzz.trapmf(lateral_position.universe, [40, 60, 90, 90])
        
        speed = ctrl.Antecedent(np.arange(0, 121, 1), 'speed')
        speed['stopped'] = fuzz.trapmf(speed.universe, [0, 0, 3, 8])
        speed['very_slow'] = fuzz.trimf(speed.universe, [5, 15, 25])
        speed['slow'] = fuzz.trimf(speed.universe, [20, 35, 50])
        speed['moderate'] = fuzz.trimf(speed.universe, [45, 65, 85])
        speed['fast'] = fuzz.trapmf(speed.universe, [80, 100, 120, 120])
        
        # OUTPUT VARIABLES
        urgency = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')
        urgency['none'] = fuzz.trapmf(urgency.universe, [0, 0, 5, 15])
        urgency['low'] = fuzz.trimf(urgency.universe, [10, 25, 40])
        urgency['moderate'] = fuzz.trimf(urgency.universe, [35, 50, 65])
        urgency['high'] = fuzz.trimf(urgency.universe, [60, 75, 90])
        urgency['critical'] = fuzz.trapmf(urgency.universe, [85, 92, 100, 100])
        
        safety_score = ctrl.Consequent(np.arange(0, 101, 1), 'safety_score')
        safety_score['unsafe'] = fuzz.trapmf(safety_score.universe, [0, 0, 15, 30])
        safety_score['risky'] = fuzz.trimf(safety_score.universe, [25, 40, 55])
        safety_score['moderate'] = fuzz.trimf(safety_score.universe, [50, 65, 80])
        safety_score['safe'] = fuzz.trapmf(safety_score.universe, [75, 90, 100, 100])
        
        # COMPREHENSIVE FUZZY RULES (30+ rules for precision)
        rules = []
        
        # CRITICAL IMMEDIATE DANGER RULES
        rules.append(ctrl.Rule(
            distance['immediate'] & priority['high'] & confidence['high'],
            (urgency['critical'], safety_score['unsafe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['immediate'] & lateral_position['center'] & confidence['medium'],
            (urgency['critical'], safety_score['unsafe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['very_close'] & priority['high'] & lateral_position['center'] & speed['moderate'],
            (urgency['critical'], safety_score['unsafe'])
        ))
        
        # HIGH-SPEED DANGER RULES
        rules.append(ctrl.Rule(
            distance['close'] & speed['fast'] & priority['high'],
            (urgency['critical'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['very_close'] & speed['moderate'] & confidence['high'],
            (urgency['high'], safety_score['risky'])
        ))
        
        # PEDESTRIAN SAFETY RULES (highest priority)
        rules.append(ctrl.Rule(
            distance['very_close'] & priority['high'] & confidence['very_high'],
            (urgency['critical'], safety_score['unsafe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['close'] & priority['high'] & lateral_position['center'],
            (urgency['high'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['close'] & priority['high'] & speed['slow'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        # TRAFFIC SIGNAL RULES
        rules.append(ctrl.Rule(
            distance['close'] & priority['high'] & confidence['very_high'] & lateral_position['center'],
            (urgency['high'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & priority['high'] & lateral_position['center'] & speed['moderate'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        # VEHICLE INTERACTION RULES
        rules.append(ctrl.Rule(
            distance['very_close'] & priority['medium'] & lateral_position['center'],
            (urgency['high'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['immediate'] & priority['medium'] & lateral_position['center'],
            (urgency['critical'], safety_score['unsafe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['close'] & priority['medium'] & confidence['high'] & speed['moderate'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        rules.append(ctrl.Rule(
            distance['close'] & priority['medium'] & confidence['high'] & lateral_position['center'],
            (urgency['high'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['very_close'] & priority['medium'] & confidence['very_high'],
            (urgency['high'], safety_score['risky'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & priority['medium'] & lateral_position['center'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        # SAFE PASSAGE RULES (side zones)
        rules.append(ctrl.Rule(
            distance['medium'] & lateral_position['left'] & confidence['medium'],
            (urgency['low'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & lateral_position['right'] & confidence['medium'],
            (urgency['low'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['far'] & lateral_position['far_left'],
            (urgency['none'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['far'] & lateral_position['far_right'],
            (urgency['none'], safety_score['safe'])
        ))
        
        # TURN OPPORTUNITY RULES
        rules.append(ctrl.Rule(
            distance['far'] & lateral_position['left'] & confidence['low'],
            (urgency['none'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['far'] & lateral_position['right'] & confidence['low'],
            (urgency['none'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & lateral_position['far_left'] & priority['low'],
            (urgency['low'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & lateral_position['far_right'] & priority['low'],
            (urgency['low'], safety_score['safe'])
        ))
        
        # SLOW SPEED SAFE RULES
        rules.append(ctrl.Rule(
            distance['close'] & speed['very_slow'] & priority['medium'],
            (urgency['low'], safety_score['moderate'])
        ))
        
        rules.append(ctrl.Rule(
            distance['very_close'] & speed['stopped'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        # LOW CONFIDENCE RULES
        rules.append(ctrl.Rule(
            confidence['very_low'] & distance['medium'],
            (urgency['low'], safety_score['moderate'])
        ))
        
        rules.append(ctrl.Rule(
            confidence['low'] & distance['close'],
            (urgency['moderate'], safety_score['risky'])
        ))
        
        # MIXED SCENARIO RULES
        rules.append(ctrl.Rule(
            distance['close'] & lateral_position['left'] & priority['medium'] & speed['slow'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        rules.append(ctrl.Rule(
            distance['close'] & lateral_position['right'] & priority['medium'] & speed['slow'],
            (urgency['moderate'], safety_score['moderate'])
        ))
        
        rules.append(ctrl.Rule(
            distance['medium'] & confidence['high'] & priority['low'] & speed['moderate'],
            (urgency['low'], safety_score['safe'])
        ))
        
        # FAR DISTANCE SAFE RULES
        rules.append(ctrl.Rule(
            distance['far'] & confidence['medium'],
            (urgency['none'], safety_score['safe'])
        ))
        
        rules.append(ctrl.Rule(
            distance['far'] & priority['low'],
            (urgency['none'], safety_score['safe'])
        ))
        
        # Control systems
        self.urgency_ctrl = ctrl.ControlSystem(rules)
        self.urgency_sim = ctrl.ControlSystemSimulation(self.urgency_ctrl)
        
        print(f"      ✓ {len(rules)} fuzzy rules configured")
    
    def estimate_real_distance(self, bbox, image_width, image_height, object_class):
        """Estimate distance using object size and perspective"""
        
        real_heights = {
            'person': 1.7,
            'car': 1.5,
            'truck': 3.5,
            'bus': 3.2,
            'bicycle': 1.2,
            'motorcycle': 1.3,
            'traffic light': 3.0,
            'stop sign': 2.5
        }
        
        bbox_height = bbox[3] - bbox[1]
        bbox_width = bbox[2] - bbox[0]
        bbox_area = bbox_width * bbox_height
        
        real_height = real_heights.get(object_class, 1.7)
        focal_length = 700
        
        if bbox_height > 0:
            estimated_distance = (real_height * focal_length) / bbox_height
        else:
            estimated_distance = 100
        
        # Adjust based on vertical position (objects lower in frame are closer)
        frame_height = image_height
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        vertical_ratio = bbox_center_y / frame_height
        
        # Objects in lower 60% of frame are likely closer
        if vertical_ratio > 0.6:
            estimated_distance *= 0.8
        elif vertical_ratio < 0.3:
            estimated_distance *= 1.3
        
        estimated_distance = np.clip(estimated_distance, 5, 200)
        
        return estimated_distance
    
    def calculate_lateral_position(self, bbox, image_width):
        """Calculate precise lateral position"""
        
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        image_center_x = image_width / 2
        
        lateral_angle = ((bbox_center_x - image_center_x) / (image_width / 2)) * 45
        
        return lateral_angle
    
    def get_zone_from_angle(self, angle):
        """Determine which spatial zone an object is in"""
        
        for zone_name, zone_info in self.zones.items():
            if zone_info['angle_range'][0] <= angle <= zone_info['angle_range'][1]:
                return zone_name
        return 'center'
    
    def analyze_path_clearance(self, detections):
        """Analyze which paths are clear for turns"""
        
        path_analysis = {
            'center': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'left': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'right': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'far_left': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')},
            'far_right': {'clear': True, 'obstacles': [], 'closest_distance': float('inf')}
        }
        
        for det in detections:
            zone = self.get_zone_from_angle(det['lateral_position'])
            distance = det['distance']
            
            # Check if object is blocking (based on object-specific safe distances)
            obj_info = self.nav_objects.get(det['class'], {})
            caution_distance = obj_info.get('caution_distance', 30)
            
            if distance < caution_distance:
                path_analysis[zone]['obstacles'].append(det)
                if distance < path_analysis[zone]['closest_distance']:
                    path_analysis[zone]['closest_distance'] = distance
                
                stop_distance = obj_info.get('stop_distance', 10)
                if distance < stop_distance:
                    path_analysis[zone]['clear'] = False
        
        return path_analysis
    
    def generate_detailed_instruction(self, detections, current_speed=30):
        """Generate highly detailed navigation instruction"""
        
        if not detections:
            return {
                'primary_instruction': "CONTINUE STRAIGHT - Path is clear",
                'action': 'PROCEED',
                'urgency': 'NONE',
                'reason': 'No obstacles detected',
                'details': {
                    'center_clear': True,
                    'left_turn_available': True,
                    'right_turn_available': True,
                    'safe_to_proceed': True
                },
                'turn_opportunities': {
                    'left': 'SAFE - No obstacles detected',
                    'right': 'SAFE - No obstacles detected'
                },
                'warnings': []
            }
        
        # Analyze all paths
        path_analysis = self.analyze_path_clearance(detections)
        
        # Calculate urgency for each detection
        for det in detections:
            try:
                self.urgency_sim.input['distance'] = det['distance']
                self.urgency_sim.input['confidence'] = det['confidence']
                self.urgency_sim.input['priority'] = det['priority']
                self.urgency_sim.input['lateral_position'] = det['lateral_position']
                self.urgency_sim.input['speed'] = current_speed
                
                self.urgency_sim.compute()
                
                det['urgency_score'] = self.urgency_sim.output['urgency']
                det['safety_score'] = self.urgency_sim.output['safety_score']
            except:
                det['urgency_score'] = 0
                det['safety_score'] = 100
        
        # Sort by urgency
        critical_objects = sorted(
            [d for d in detections if d['urgency_score'] > 60],
            key=lambda x: (x['urgency_score'], -x['distance']),
            reverse=True
        )
        
        # Determine primary action
        center_path = path_analysis['center']
        left_path = path_analysis['left']
        right_path = path_analysis['right']
        
        instruction = {
            'primary_instruction': '',
            'action': '',
            'urgency': '',
            'reason': '',
            'details': {},
            'turn_opportunities': {},
            'warnings': [],
            'immediate_threats': []
        }
        
        # Check for objects that MUST stop based on their stop_distance
        def should_emergency_stop(obj):
            """Check if object requires immediate stop"""
            obj_info = self.nav_objects.get(obj['class'], {})
            stop_dist = obj_info.get('stop_distance', 10)
            # Only stop if object is in CENTER path (within ±20 degrees)
            return obj['distance'] <= stop_dist and abs(obj['lateral_position']) < 20
        
        # Find all objects requiring emergency stop
        emergency_stop_objects = [d for d in detections if should_emergency_stop(d)]
        
        # CRITICAL SITUATION - Object within stop distance in CENTER path
        if emergency_stop_objects:
            threat = min(emergency_stop_objects, key=lambda x: x['distance'])
            obj_info = self.nav_objects.get(threat['class'], {})
            stop_dist = obj_info.get('stop_distance', 10)
            
            instruction['primary_instruction'] = f"⚠️ STOP - {threat['class'].upper()} at {threat['distance']:.1f}m ahead"
            instruction['action'] = 'EMERGENCY_STOP'
            instruction['urgency'] = 'CRITICAL'
            instruction['reason'] = f"{threat['class']} at {threat['distance']:.1f}m (stop threshold: {stop_dist}m)"
            instruction['immediate_threats'] = emergency_stop_objects[:3]
            
            # Check for emergency turn options
            if left_path['clear'] and left_path['closest_distance'] > 20:
                instruction['turn_opportunities']['left'] = f"✓ LEFT available - Clear for {left_path['closest_distance']:.0f}m"
            else:
                instruction['turn_opportunities']['left'] = f"✗ LEFT blocked"
            
            if right_path['clear'] and right_path['closest_distance'] > 20:
                instruction['turn_opportunities']['right'] = f"✓ RIGHT available - Clear for {right_path['closest_distance']:.0f}m"
            else:
                instruction['turn_opportunities']['right'] = f"✗ RIGHT blocked"
                
            instruction['details'] = {
                'center_clear': False,
                'obstacle_type': threat['class'],
                'obstacle_distance': threat['distance'],
                'stop_threshold': stop_dist
            }
            
            return instruction
        
        # HIGH PRIORITY - Object close in center but not at stop distance
        if not center_path['clear'] or center_path['closest_distance'] < 25:
            obstacles_ahead = center_path['obstacles']
            closest = min(obstacles_ahead, key=lambda x: x['distance']) if obstacles_ahead else None
            
            if closest:
                if closest['class'] in ['person', 'bicycle']:
                    instruction['primary_instruction'] = f"⚠️ SLOW DOWN - {closest['class'].upper()} crossing at {closest['distance']:.1f}m"
                    instruction['action'] = 'YIELD'
                    instruction['urgency'] = 'HIGH'
                    instruction['reason'] = f"{closest['class']} crossing at {closest['distance']:.1f}m"
                elif closest['class'] in ['traffic light', 'stop sign']:
                    instruction['primary_instruction'] = f"🛑 PREPARE TO STOP - {closest['class'].upper()} at {closest['distance']:.1f}m"
                    instruction['action'] = 'PREPARE_STOP'
                    instruction['urgency'] = 'HIGH'
                    instruction['reason'] = f"{closest['class']} at {closest['distance']:.1f}m"
                else:
                    instruction['primary_instruction'] = f"⚠️ REDUCE SPEED - {closest['class'].upper()} at {closest['distance']:.1f}m ahead"
                    instruction['action'] = 'SLOW_DOWN'
                    instruction['urgency'] = 'MODERATE'
                    instruction['reason'] = f"{closest['class']} at {closest['distance']:.1f}m in path"
                
                # Analyze turn options
                if left_path['clear']:
                    instruction['turn_opportunities']['left'] = f"✓ LEFT TURN SAFE - Clear path, nearest object at {left_path['closest_distance']:.0f}m"
                else:
                    instruction['turn_opportunities']['left'] = f"✗ LEFT TURN UNSAFE - Nearest obstacle at {left_path['closest_distance']:.0f}m"
                
                if right_path['clear']:
                    instruction['turn_opportunities']['right'] = f"✓ RIGHT TURN SAFE - Clear path, nearest object at {right_path['closest_distance']:.0f}m"
                else:
                    instruction['turn_opportunities']['right'] = f"✗ RIGHT TURN UNSAFE - Nearest obstacle at {right_path['closest_distance']:.0f}m"
                
                instruction['details'] = {
                    'center_clear': False,
                    'center_distance': closest['distance']
                }
                
                return instruction
        
        # MODERATE - Caution but path navigable
        if center_path['closest_distance'] < 50:
            instruction['primary_instruction'] = f"➡️ CONTINUE STRAIGHT with caution - Vehicle/object at {center_path['closest_distance']:.0f}m"
            instruction['action'] = 'PROCEED_CAUTION'
            instruction['urgency'] = 'LOW'
            instruction['reason'] = f"Object detected at {center_path['closest_distance']:.0f}m ahead"
        else:
            instruction['primary_instruction'] = "➡️ CONTINUE STRAIGHT - Path clear"
            instruction['action'] = 'PROCEED'
            instruction['urgency'] = 'NONE'
            instruction['reason'] = 'No obstacles in direct path'
        
        # Detailed turn analysis
        if left_path['clear'] and left_path['closest_distance'] > 30:
            instruction['turn_opportunities']['left'] = f"✓ LEFT TURN RECOMMENDED - Wide clearance ({left_path['closest_distance']:.0f}m)"
        elif left_path['clear'] and left_path['closest_distance'] > 15:
            instruction['turn_opportunities']['left'] = f"⚠️ LEFT TURN POSSIBLE - Tight clearance ({left_path['closest_distance']:.0f}m)"
        else:
            left_obs = left_path['obstacles']
            if left_obs:
                closest_left = min(left_obs, key=lambda x: x['distance'])
                instruction['turn_opportunities']['left'] = f"✗ LEFT TURN UNSAFE - {closest_left['class']} at {closest_left['distance']:.0f}m"
            else:
                instruction['turn_opportunities']['left'] = "✗ LEFT TURN BLOCKED"
        
        if right_path['clear'] and right_path['closest_distance'] > 30:
            instruction['turn_opportunities']['right'] = f"✓ RIGHT TURN RECOMMENDED - Wide clearance ({right_path['closest_distance']:.0f}m)"
        elif right_path['clear'] and right_path['closest_distance'] > 15:
            instruction['turn_opportunities']['right'] = f"⚠️ RIGHT TURN POSSIBLE - Tight clearance ({right_path['closest_distance']:.0f}m)"
        else:
            right_obs = right_path['obstacles']
            if right_obs:
                closest_right = min(right_obs, key=lambda x: x['distance'])
                instruction['turn_opportunities']['right'] = f"✗ RIGHT TURN UNSAFE - {closest_right['class']} at {closest_right['distance']:.0f}m"
            else:
                instruction['turn_opportunities']['right'] = "✗ RIGHT TURN BLOCKED"
        
        # Add warnings for side objects
        for zone in ['left', 'right', 'far_left', 'far_right']:
            zone_obstacles = path_analysis[zone]['obstacles']
            for obs in zone_obstacles:
                if obs['distance'] < 40:
                    instruction['warnings'].append(
                        f"{obs['class']} detected in {zone.replace('_', ' ')} zone at {obs['distance']:.0f}m"
                    )
        
        # Detailed information
        instruction['details'] = {
            'center_clear': center_path['clear'],
            'center_distance': center_path['closest_distance'],
            'left_clear': left_path['clear'],
            'left_distance': left_path['closest_distance'],
            'right_clear': right_path['clear'],
            'right_distance': right_path['closest_distance'],
            'total_objects': len(detections),
            'high_priority_objects': len([d for d in detections if d['priority'] >= 9]),
            'current_speed': current_speed
        }
        
        return instruction
    
    def visualize_instruction(self, img, instruction, detections):
        """Draw comprehensive visualization on image"""
        
        height, width = img.shape[:2]
        
        # Draw detections with zones
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            zone = self.get_zone_from_angle(det['lateral_position'])
            
            # Color by urgency
            if det.get('urgency_score', 0) > 85:
                color = (0, 0, 255)  # Red - Critical
            elif det.get('urgency_score', 0) > 60:
                color = (0, 140, 255)  # Orange - High
            elif det.get('urgency_score', 0) > 30:
                color = (0, 255, 255)  # Yellow - Moderate
            else:
                color = (0, 255, 0)  # Green - Low
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw compact label
            label = f"{det['class']}: {det['distance']:.0f}m"
            cv2.putText(img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw zone indicators at top
        zone_width = width // 5
        zone_labels = ['FAR LEFT', 'LEFT', 'CENTER', 'RIGHT', 'FAR RIGHT']
        zone_keys = ['far_left', 'left', 'center', 'right', 'far_right']
        
        path_analysis = self.analyze_path_clearance(detections)
        
        for i, (label, key) in enumerate(zip(zone_labels, zone_keys)):
            x_start = i * zone_width
            zone_info = path_analysis[key]
            
            if zone_info['clear']:
                zone_color = (0, 150, 0)
            else:
                zone_color = (0, 0, 150)
            
            cv2.rectangle(img, (x_start, 0), (x_start + zone_width, 30), zone_color, -1)
            cv2.rectangle(img, (x_start, 0), (x_start + zone_width, 30), (255, 255, 255), 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = x_start + (zone_width - label_size[0]) // 2
            cv2.putText(img, label, (label_x, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw action instruction at top center (below zone headers)
        action_y = 50
        action_text = instruction['action']
        
        urgency_colors = {
            'NONE': (0, 255, 0),
            'LOW': (0, 255, 0),
            'MODERATE': (0, 255, 255),
            'HIGH': (0, 165, 255),
            'CRITICAL': (0, 0, 255)
        }
        action_color = urgency_colors.get(instruction['urgency'], (255, 255, 255))
        
        # Draw semi-transparent background for action text
        action_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        action_x = (width - action_size[0]) // 2
        
        overlay = img.copy()
        cv2.rectangle(overlay, (action_x - 15, action_y - 25), 
                     (action_x + action_size[0] + 15, action_y + 10), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        cv2.putText(img, action_text, (action_x, action_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
        
        # Draw reason below action (if available)
        reason_text = instruction.get('reason', '')
        if reason_text:
            if len(reason_text) > 60:
                reason_text = reason_text[:57] + "..."
            
            reason_size = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            reason_x = (width - reason_size[0]) // 2
            reason_y = action_y + 25
            
            overlay = img.copy()
            cv2.rectangle(overlay, (reason_x - 10, reason_y - 18), 
                         (reason_x + reason_size[0] + 10, reason_y + 5), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            cv2.putText(img, reason_text, (reason_x, reason_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            
            turn_y_adjust = 130
        else:
            turn_y_adjust = 100
        
        # If action is STOP or critical, show turn alternatives on sides
        if instruction['action'] in ['EMERGENCY_STOP', 'YIELD', 'PREPARE_STOP']:
            # LEFT side info
            left_y = turn_y_adjust
            left_dist = path_analysis['left']['closest_distance']
            if left_dist != float('inf'):
                if left_dist > 30:
                    left_text = f"LEFT CLEAR"
                    left_detail = f"{left_dist:.0f}m"
                    left_color = (0, 255, 0)
                elif left_dist > 15:
                    left_text = f"LEFT TIGHT"
                    left_detail = f"{left_dist:.0f}m"
                    left_color = (0, 200, 255)
                else:
                    left_text = f"LEFT BLOCKED"
                    if path_analysis['left']['obstacles']:
                        obs = min(path_analysis['left']['obstacles'], key=lambda x: x['distance'])
                        left_detail = f"{obs['class']} {obs['distance']:.0f}m"
                    else:
                        left_detail = f"{left_dist:.0f}m"
                    left_color = (0, 0, 255)
            else:
                left_text = "LEFT CLEAR"
                left_detail = ""
                left_color = (0, 255, 0)
            
            # Draw left info
            overlay = img.copy()
            cv2.rectangle(overlay, (10, left_y - 20), (150, left_y + 30), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            cv2.putText(img, left_text, (15, left_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            if left_detail:
                cv2.putText(img, left_detail, (15, left_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
            
            # RIGHT side info
            right_dist = path_analysis['right']['closest_distance']
            if right_dist != float('inf'):
                if right_dist > 30:
                    right_text = f"RIGHT CLEAR"
                    right_detail = f"{right_dist:.0f}m"
                    right_color = (0, 255, 0)
                elif right_dist > 15:
                    right_text = f"RIGHT TIGHT"
                    right_detail = f"{right_dist:.0f}m"
                    right_color = (0, 200, 255)
                else:
                    right_text = f"RIGHT BLOCKED"
                    if path_analysis['right']['obstacles']:
                        obs = min(path_analysis['right']['obstacles'], key=lambda x: x['distance'])
                        right_detail = f"{obs['class']} {obs['distance']:.0f}m"
                    else:
                        right_detail = f"{right_dist:.0f}m"
                    right_color = (0, 0, 255)
            else:
                right_text = "RIGHT CLEAR"
                right_detail = ""
                right_color = (0, 255, 0)
            
            # Draw right info
            overlay = img.copy()
            cv2.rectangle(overlay, (width - 160, left_y - 20), (width - 10, left_y + 30), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            cv2.putText(img, right_text, (width - 155, left_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
            if right_detail:
                cv2.putText(img, right_detail, (width - 155, left_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
        
        # Bottom info bar - only show nearby objects
        nearby_objects = [d for d in detections if d['distance'] < 50]
        if nearby_objects:
            info_y = height - 35
            
            # Semi-transparent bottom bar
            overlay = img.copy()
            cv2.rectangle(overlay, (0, info_y - 10), (width, height), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            # Show up to 5 nearest objects
            nearby_sorted = sorted(nearby_objects, key=lambda x: x['distance'])[:5]
            info_text = "Nearby: "
            for obj in nearby_sorted:
                zone = self.get_zone_from_angle(obj['lateral_position'])
                info_text += f"{obj['class']} {obj['distance']:.0f}m ({zone}) | "
            
            info_text = info_text[:-3]  # Remove last separator
            if len(info_text) > 120:
                info_text = info_text[:117] + "..."
            
            cv2.putText(img, info_text, (10, info_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return img
    
    def process_image(self, image_path, current_speed=30):
        """Process image and generate comprehensive navigation analysis"""
        
        img_name = Path(image_path).name
        print(f"\n{'='*80}")
        print(f"PROCESSING: {img_name}")
        print(f"{'='*80}")
        
        img = cv2.imread(image_path)
        if img is None:
            return None, [], {}
        
        height, width = img.shape[:2]
        
        # Run YOLO detection
        results = self.model(img, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                class_name = self.model.names[cls_id]
                
                if class_name not in self.nav_objects or conf < 0.3:
                    continue
                
                distance = self.estimate_real_distance(bbox, width, height, class_name)
                lateral_pos = self.calculate_lateral_position(bbox, width)
                priority = self.nav_objects[class_name]['priority']
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x) for x in bbox],
                    'distance': float(distance),
                    'lateral_position': float(lateral_pos),
                    'priority': priority,
                    'object_type': self.nav_objects[class_name]['type']
                }
                
                detections.append(detection)
        
        # Generate detailed instruction
        instruction = self.generate_detailed_instruction(detections, current_speed)
        
        # Visualize
        img_annotated = self.visualize_instruction(img, instruction, detections)
        
        # Print comprehensive analysis
        print(f"\n📊 SCENE ANALYSIS:")
        print(f"   Total objects detected: {len(detections)}")
        
        if detections:
            by_type = defaultdict(list)
            for det in detections:
                by_type[det['class']].append(det)
            
            for obj_type, items in sorted(by_type.items()):
                distances = [d['distance'] for d in items]
                print(f"   • {obj_type}: {len(items)} detected, distances: {min(distances):.0f}m - {max(distances):.0f}m")
        
        print(f"\n🚦 NAVIGATION:")
        print(f"   → {instruction['primary_instruction']}")
        print(f"   Action: {instruction['action']} | Urgency: {instruction['urgency']}")
        if instruction.get('reason'):
            print(f"   Reason: {instruction['reason']}")
        
        print(f"\n🔄 TURN OPTIONS:")
        
        # Analyze paths for compact output
        path_analysis = self.analyze_path_clearance(detections)
        
        # Left
        left_dist = path_analysis['left']['closest_distance']
        if left_dist != float('inf'):
            if left_dist > 30:
                print(f"   ← LEFT:  ✓ {left_dist:.0f}m CLEAR")
            elif left_dist > 15:
                print(f"   ← LEFT:  ⚠ {left_dist:.0f}m TIGHT")
            else:
                obs_info = ""
                if path_analysis['left']['obstacles']:
                    obs = min(path_analysis['left']['obstacles'], key=lambda x: x['distance'])
                    obs_info = f" ({obs['class']})"
                print(f"   ← LEFT:  ✗ {left_dist:.0f}m BLOCKED{obs_info}")
        else:
            print(f"   ← LEFT:  ✓ CLEAR")
        
        # Right
        right_dist = path_analysis['right']['closest_distance']
        if right_dist != float('inf'):
            if right_dist > 30:
                print(f"   → RIGHT: ✓ {right_dist:.0f}m CLEAR")
            elif right_dist > 15:
                print(f"   → RIGHT: ⚠ {right_dist:.0f}m TIGHT")
            else:
                obs_info = ""
                if path_analysis['right']['obstacles']:
                    obs = min(path_analysis['right']['obstacles'], key=lambda x: x['distance'])
                    obs_info = f" ({obs['class']})"
                print(f"   → RIGHT: ✗ {right_dist:.0f}m BLOCKED{obs_info}")
        else:
            print(f"   → RIGHT: ✓ CLEAR")
        
        # Far Left
        fl_dist = path_analysis['far_left']['closest_distance']
        if fl_dist != float('inf') and fl_dist < 50:
            status = "✓ CLEAR" if fl_dist > 30 else "⚠ CAUTION" if fl_dist > 15 else "✗ BLOCKED"
            obs_info = ""
            if fl_dist <= 30 and path_analysis['far_left']['obstacles']:
                obs = min(path_analysis['far_left']['obstacles'], key=lambda x: x['distance'])
                obs_info = f" ({obs['class']})"
            print(f"   ↖ FAR L: {status} {fl_dist:.0f}m{obs_info}")
        
        # Far Right
        fr_dist = path_analysis['far_right']['closest_distance']
        if fr_dist != float('inf') and fr_dist < 50:
            status = "✓ CLEAR" if fr_dist > 30 else "⚠ CAUTION" if fr_dist > 15 else "✗ BLOCKED"
            obs_info = ""
            if fr_dist <= 30 and path_analysis['far_right']['obstacles']:
                obs = min(path_analysis['far_right']['obstacles'], key=lambda x: x['distance'])
                obs_info = f" ({obs['class']})"
            print(f"   ↗ FAR R: {status} {fr_dist:.0f}m{obs_info}")
        
        return img_annotated, detections, instruction
    
    def process_dataset(self, dataset_path):
        """Process entire dataset with varying scenarios"""
        
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE NAVIGATION ANALYSIS")
        print("="*80)
        
        image_dir = Path(dataset_path) / 'images'
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if not image_files:
            print(f"\n✗ No images found in {image_dir}")
            return
        
        print(f"\n✓ Found {len(image_files)} images to analyze")
        
        all_results = []
        
        # Simulate different speeds for varied scenarios
        speeds = [20, 30, 40, 50, 25, 35, 45, 15, 40, 30, 25, 35, 20, 30, 40]
        
        for idx, img_path in enumerate(image_files, 1):
            current_speed = speeds[idx % len(speeds)]
            
            processed_img, detections, instruction = self.process_image(str(img_path), current_speed)
            
            if processed_img is not None:
                output_path = results_dir / f"enhanced_nav_{img_path.name}"
                cv2.imwrite(str(output_path), processed_img)
                
                result = {
                    'image': img_path.name,
                    'current_speed': current_speed,
                    'detections': detections,
                    'instruction': instruction,
                    'total_objects': len(detections)
                }
                all_results.append(result)
        
        # Generate comprehensive report
        report_path = results_dir / 'enhanced_navigation_report.json'
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary statistics
        self.generate_summary_report(all_results, results_dir)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"✓ Images processed: {len(image_files)}")
        print(f"✓ Results saved in: {results_dir}")
        print(f"✓ Detailed report: {report_path}")
        print("="*80 + "\n")
        
        return all_results
    
    def generate_summary_report(self, results, output_dir):
        """Generate summary statistics report"""
        
        summary = {
            'total_images': len(results),
            'action_distribution': defaultdict(int),
            'urgency_distribution': defaultdict(int),
            'safe_turns_left': 0,
            'safe_turns_right': 0,
            'critical_situations': 0,
            'average_objects_per_scene': 0,
            'most_common_objects': defaultdict(int)
        }
        
        total_objects = 0
        
        for result in results:
            instruction = result['instruction']
            summary['action_distribution'][instruction['action']] += 1
            summary['urgency_distribution'][instruction['urgency']] += 1
            
            if '✓' in instruction['turn_opportunities'].get('left', ''):
                summary['safe_turns_left'] += 1
            if '✓' in instruction['turn_opportunities'].get('right', ''):
                summary['safe_turns_right'] += 1
            
            if instruction['urgency'] == 'CRITICAL':
                summary['critical_situations'] += 1
            
            total_objects += result['total_objects']
            
            for det in result['detections']:
                summary['most_common_objects'][det['class']] += 1
        
        summary['average_objects_per_scene'] = total_objects / len(results) if results else 0
        
        # Convert defaultdicts to regular dicts for JSON
        summary['action_distribution'] = dict(summary['action_distribution'])
        summary['urgency_distribution'] = dict(summary['urgency_distribution'])
        summary['most_common_objects'] = dict(sorted(
            summary['most_common_objects'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        summary_path = output_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total scenes analyzed: {summary['total_images']}")
        print(f"   Average objects per scene: {summary['average_objects_per_scene']:.1f}")
        print(f"   Critical situations: {summary['critical_situations']}")
        print(f"   Safe left turns available: {summary['safe_turns_left']}")
        print(f"   Safe right turns available: {summary['safe_turns_right']}")
        print(f"\n   Most common objects:")
        for obj, count in list(summary['most_common_objects'].items())[:5]:
            print(f"      • {obj}: {count}")


def main():
    print("\n" + "="*80)
    print(" "*15 + "ENHANCED CONTEXT-AWARE NAVIGATION SYSTEM")
    print(" "*20 + "With Precise Turn Opportunity Detection")
    print("="*80)
    
    print("\n🎯 SYSTEM ASSUMPTIONS:")
    print("   • User is traveling STRAIGHT")
    print("   • System analyzes surrounding for safe turn opportunities")
    print("   • Real-time object detection with distance estimation")
    print("   • Multi-zone spatial analysis (5 zones)")
    print("   • Priority-based decision making")
    print("   • Immediate threat detection and emergency response")
    
    # Download dataset
    downloader = RealDatasetDownloader()
    dataset_path = 'dataset'
    downloader.download_dataset(dataset_path)
    
    # Initialize enhanced navigation system
    nav_system = EnhancedNavigationSystem()
    
    # Process all images
    results = nav_system.process_dataset(dataset_path)
    
    print("\n" + "="*80)
    print("✓ ENHANCED NAVIGATION SYSTEM DEMO COMPLETED!")
    print("="*80)
    print("\n📁 OUTPUT FILES:")
    print("   • results/enhanced_nav_*.jpg - Annotated images with instructions")
    print("   • results/enhanced_navigation_report.json - Detailed analysis")
    print("   • results/summary_statistics.json - Statistical summary")
    
    print("\n🚀 KEY IMPROVEMENTS:")
    print("   ✓ 30+ fuzzy logic rules for precise decision making")
    print("   ✓ 5-zone spatial analysis (Far Left, Left, Center, Right, Far Right)")
    print("   ✓ Real distance estimation using object size")
    print("   ✓ Turn opportunity detection with clearance calculation")
    print("   ✓ Emergency stop detection for immediate threats")
    print("   ✓ Object-specific safe distance thresholds")
    print("   ✓ Multi-factor urgency calculation")
    print("   ✓ Detailed turn recommendations (SAFE/POSSIBLE/UNSAFE)")
    print("   ✓ Speed-aware decision making")
    print("   ✓ Visual zone indicators and comprehensive HUD")
    
    print("\n🎯 NAVIGATION ACTIONS AVAILABLE:")
    print("   • EMERGENCY_STOP - Immediate danger detected")
    print("   • YIELD - Pedestrian or high-priority crossing")
    print("   • PREPARE_STOP - Traffic signal or stop sign ahead")
    print("   • SLOW_DOWN - Reduce speed for safety")
    print("   • PROCEED_CAUTION - Continue with awareness")
    print("   • PROCEED - Clear path, normal operation")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()