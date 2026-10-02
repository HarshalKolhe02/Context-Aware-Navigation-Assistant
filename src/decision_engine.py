"""
Decision Engine & Inference Module
Implements multi-variable continuous inference, priority weighting, and real-time navigation rule generation.
"""

from collections import defaultdict
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class NavigationDecisionEngine:
    """
    Multi-factor inference and decision system.
    Evaluates distance, confidence, class priority, lateral angular offset,
    and velocity to determine safety scores, urgency metrics, and actions.
    """
    
    def __init__(self, nav_objects, voice_assistant=None):
        self.nav_objects = nav_objects
        self.voice_assistant = voice_assistant
        self.last_voice_instruction = None
        self._setup_inference_system()
        
    def _setup_inference_system(self):
        """Construct multi-variable continuous inference network and rule base."""
        # Input Antecedents
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
        
        # Consequents
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
        
        # Comprehensive Rule Network (30+ rules for precision)
        rules = [
            # Immediate Danger
            ctrl.Rule(distance['immediate'] & priority['high'] & confidence['high'], (urgency['critical'], safety_score['unsafe'])),
            ctrl.Rule(distance['immediate'] & lateral_position['center'] & confidence['medium'], (urgency['critical'], safety_score['unsafe'])),
            ctrl.Rule(distance['very_close'] & priority['high'] & lateral_position['center'] & speed['moderate'], (urgency['critical'], safety_score['unsafe'])),
            ctrl.Rule(distance['very_close'] & priority['high'] & lateral_position['center'] & speed['fast'], (urgency['critical'], safety_score['unsafe'])),
            ctrl.Rule(distance['immediate'] & speed['moderate'], (urgency['critical'], safety_score['unsafe'])),
            ctrl.Rule(distance['immediate'] & speed['fast'], (urgency['critical'], safety_score['unsafe'])),
            
            # High Urgency
            ctrl.Rule(distance['very_close'] & priority['high'] & lateral_position['center'], (urgency['high'], safety_score['unsafe'])),
            ctrl.Rule(distance['very_close'] & lateral_position['center'] & confidence['high'], (urgency['high'], safety_score['risky'])),
            ctrl.Rule(distance['close'] & priority['high'] & lateral_position['center'] & speed['fast'], (urgency['high'], safety_score['risky'])),
            ctrl.Rule(distance['very_close'] & (lateral_position['left'] | lateral_position['right']) & speed['fast'], (urgency['high'], safety_score['risky'])),
            ctrl.Rule(distance['close'] & priority['high'] & speed['moderate'], (urgency['high'], safety_score['risky'])),
            
            # Moderate Urgency
            ctrl.Rule(distance['close'] & lateral_position['center'], (urgency['moderate'], safety_score['moderate'])),
            ctrl.Rule(distance['close'] & priority['high'], (urgency['moderate'], safety_score['moderate'])),
            ctrl.Rule(distance['medium'] & priority['high'] & speed['fast'], (urgency['moderate'], safety_score['moderate'])),
            ctrl.Rule(distance['very_close'] & (lateral_position['left'] | lateral_position['right']), (urgency['moderate'], safety_score['moderate'])),
            ctrl.Rule(distance['close'] & speed['moderate'], (urgency['moderate'], safety_score['moderate'])),
            
            # Low Urgency
            ctrl.Rule(distance['medium'] & lateral_position['center'], (urgency['low'], safety_score['safe'])),
            ctrl.Rule(distance['close'] & (lateral_position['left'] | lateral_position['right']), (urgency['low'], safety_score['safe'])),
            ctrl.Rule(distance['medium'] & priority['medium'], (urgency['low'], safety_score['safe'])),
            ctrl.Rule(distance['far'] & priority['high'] & speed['fast'], (urgency['low'], safety_score['safe'])),
            ctrl.Rule(distance['very_close'] & (lateral_position['far_left'] | lateral_position['far_right']), (urgency['low'], safety_score['moderate'])),
            
            # Clear / Negligible Urgency
            ctrl.Rule(distance['far'] & ~lateral_position['center'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(distance['far'] & confidence['low'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(distance['medium'] & (lateral_position['far_left'] | lateral_position['far_right']), (urgency['none'], safety_score['safe'])),
            ctrl.Rule(confidence['very_low'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(distance['far'] & speed['slow'], (urgency['none'], safety_score['safe'])),
            
            # Speed Adaptations
            ctrl.Rule(speed['stopped'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(speed['very_slow'] & distance['close'], (urgency['low'], safety_score['safe'])),
            ctrl.Rule(speed['fast'] & distance['medium'] & lateral_position['center'], (urgency['high'], safety_score['risky'])),
            ctrl.Rule(speed['fast'] & distance['close'], (urgency['critical'], safety_score['unsafe'])),
            
            # Peripheral Context
            ctrl.Rule(lateral_position['far_left'] & ~speed['fast'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(lateral_position['far_right'] & ~speed['fast'], (urgency['none'], safety_score['safe'])),
            ctrl.Rule(priority['low'] & distance['medium'], (urgency['none'], safety_score['safe']))
        ]
        
        urgency_ctrl = ctrl.ControlSystem(rules)
        self.urgency_sim = ctrl.ControlSystemSimulation(urgency_ctrl)
    
    def evaluate_detections(self, detections, current_speed=30):
        """Compute continuous urgency and safety scores for detections."""
        for det in detections:
            try:
                self.urgency_sim.input['distance'] = det['distance']
                self.urgency_sim.input['confidence'] = det['confidence']
                self.urgency_sim.input['priority'] = det['priority']
                self.urgency_sim.input['lateral_position'] = det['lateral_position']
                self.urgency_sim.input['speed'] = current_speed
                
                self.urgency_sim.compute()
                det['urgency_score'] = float(self.urgency_sim.output['urgency'])
                det['safety_score'] = float(self.urgency_sim.output['safety_score'])
            except Exception:
                det['urgency_score'] = 0.0
                det['safety_score'] = 100.0
    
    def generate_voice_instruction(self, instruction_data):
        """Map structured navigation decisions into concise spoken sentences."""
        action = instruction_data['action']
        
        if action == 'EMERGENCY_STOP':
            return "Emergency stop! Obstacle detected ahead."
        elif action == 'YIELD':
            return "Yield to pedestrian. Slow down and prepare to stop."
        elif action == 'PREPARE_STOP':
            return "Prepare to stop. Traffic control signal ahead."
        elif action == 'SLOW_DOWN':
            return "Slow down. Obstacle in path."
        elif action == 'PROCEED_CAUTION':
            return "Proceed with caution. Object detected ahead."
        elif action == 'PROCEED':
            left_turn = instruction_data['turn_opportunities'].get('left', '')
            right_turn = instruction_data['turn_opportunities'].get('right', '')
            
            if '✓' in left_turn and '✓' in right_turn:
                return "Path clear. Both left and right turns available."
            elif '✓' in left_turn:
                return "Path clear. Left turn available."
            elif '✓' in right_turn:
                return "Path clear. Right turn available."
            else:
                return "Path clear. Continue straight ahead."
        return "Continue navigation."

    def generate_detailed_instruction(self, detections, path_analysis, current_speed=30):
        """
        Formulate comprehensive navigation decision based on multi-zone clearance and threat evaluation.
        """
        if not detections:
            instruction = {
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
                'warnings': [],
                'immediate_threats': []
            }
            self._dispatch_voice(instruction)
            return instruction
        
        self.evaluate_detections(detections, current_speed)
        
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
        
        # Check objects triggering emergency stop
        def should_emergency_stop(obj):
            obj_info = self.nav_objects.get(obj['class'], {})
            stop_dist = obj_info.get('stop_distance', 10)
            return obj['distance'] <= stop_dist and abs(obj['lateral_position']) < 20
        
        emergency_stop_objects = [d for d in detections if should_emergency_stop(d)]
        
        # Scenario 1: Critical Immediate Hazard
        if emergency_stop_objects:
            threat = min(emergency_stop_objects, key=lambda x: x['distance'])
            obj_info = self.nav_objects.get(threat['class'], {})
            stop_dist = obj_info.get('stop_distance', 10)
            
            instruction['primary_instruction'] = f"⚠️ STOP - {threat['class'].upper()} at {threat['distance']:.1f}m ahead"
            instruction['action'] = 'EMERGENCY_STOP'
            instruction['urgency'] = 'CRITICAL'
            instruction['reason'] = f"{threat['class']} at {threat['distance']:.1f}m (stop threshold: {stop_dist}m)"
            instruction['immediate_threats'] = emergency_stop_objects[:3]
            
            instruction['turn_opportunities']['left'] = (
                f"✓ LEFT available - Clear for {left_path['closest_distance']:.0f}m"
                if left_path['clear'] and left_path['closest_distance'] > 20
                else "✗ LEFT blocked"
            )
            instruction['turn_opportunities']['right'] = (
                f"✓ RIGHT available - Clear for {right_path['closest_distance']:.0f}m"
                if right_path['clear'] and right_path['closest_distance'] > 20
                else "✗ RIGHT blocked"
            )
            instruction['details'] = {
                'center_clear': False,
                'obstacle_type': threat['class'],
                'obstacle_distance': threat['distance'],
                'stop_threshold': stop_dist
            }
            self._dispatch_voice(instruction)
            return instruction
        
        # Scenario 2: Close Proximity in Center Path
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
                
                instruction['turn_opportunities']['left'] = (
                    f"✓ LEFT TURN SAFE - Clear path, nearest object at {left_path['closest_distance']:.0f}m"
                    if left_path['clear']
                    else f"✗ LEFT TURN UNSAFE - Nearest obstacle at {left_path['closest_distance']:.0f}m"
                )
                instruction['turn_opportunities']['right'] = (
                    f"✓ RIGHT TURN SAFE - Clear path, nearest object at {right_path['closest_distance']:.0f}m"
                    if right_path['clear']
                    else f"✗ RIGHT TURN UNSAFE - Nearest obstacle at {right_path['closest_distance']:.0f}m"
                )
                instruction['details'] = {
                    'center_clear': False,
                    'center_distance': closest['distance']
                }
                self._dispatch_voice(instruction)
                return instruction
        
        # Scenario 3: Moderate Caution / Clear Path
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
        
        # Turn Clearance Calculations
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
        
        # Side Obstacle Warnings
        for zone in ['left', 'right', 'far_left', 'far_right']:
            zone_obstacles = path_analysis[zone]['obstacles']
            for obs in zone_obstacles:
                if obs['distance'] < 40:
                    instruction['warnings'].append(
                        f"{obs['class']} detected in {zone.replace('_', ' ')} zone at {obs['distance']:.0f}m"
                    )
        
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
        
        self._dispatch_voice(instruction)
        return instruction
    
    def _dispatch_voice(self, instruction):
        """Helper to send synthesized speech if voice assistant is configured."""
        if not self.voice_assistant:
            return
        voice_text = self.generate_voice_instruction(instruction)
        if voice_text != self.last_voice_instruction:
            self.voice_assistant.speak_instruction(voice_text)
            self.last_voice_instruction = voice_text
