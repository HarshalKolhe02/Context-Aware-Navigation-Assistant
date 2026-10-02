"""
Visualization & HUD Overlay Module
Renders heads-up display (HUD), zone banners, bounding boxes, and routing clearance.
"""

import cv2


class NavigationVisualizer:
    """Draws augmented navigation HUD and spatial perception indicators on imagery."""
    
    URGENCY_COLORS = {
        'NONE': (0, 255, 0),        # Green
        'LOW': (0, 255, 0),         # Green
        'MODERATE': (0, 255, 255),   # Yellow
        'HIGH': (0, 165, 255),      # Orange
        'CRITICAL': (0, 0, 255)      # Red
    }
    
    ZONE_LABELS = ['FAR LEFT', 'LEFT', 'CENTER', 'RIGHT', 'FAR RIGHT']
    ZONE_KEYS = ['far_left', 'left', 'center', 'right', 'far_right']
    
    def __init__(self, zone_manager):
        self.zone_manager = zone_manager
    
    def render(self, img, instruction, detections, path_analysis):
        """
        Render complete navigation HUD onto the frame.
        
        Args:
            img: BGR image numpy array
            instruction: Decision instruction dictionary
            detections: List of detection dictionaries
            path_analysis: Path clearance dictionary
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated = img.copy()
        height, width = annotated.shape[:2]
        
        # 1. Bounding Boxes with Urgency-Color Coding
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            urgency = det.get('urgency_score', 0)
            
            if urgency > 85:
                color = (0, 0, 255)       # Red - Critical
            elif urgency > 60:
                color = (0, 140, 255)     # Orange - High
            elif urgency > 30:
                color = (0, 255, 255)     # Yellow - Moderate
            else:
                color = (0, 255, 0)       # Green - Low
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['distance']:.0f}m"
            cv2.putText(
                annotated, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # 2. Five-Zone Clearance Indicators at Top
        zone_width = width // 5
        for i, (label, key) in enumerate(zip(self.ZONE_LABELS, self.ZONE_KEYS)):
            x_start = i * zone_width
            zone_info = path_analysis[key]
            zone_color = (0, 150, 0) if zone_info['clear'] else (0, 0, 150)
            
            cv2.rectangle(annotated, (x_start, 0), (x_start + zone_width, 30), zone_color, -1)
            cv2.rectangle(annotated, (x_start, 0), (x_start + zone_width, 30), (255, 255, 255), 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = x_start + (zone_width - label_size[0]) // 2
            cv2.putText(
                annotated, label, (label_x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        # 3. Action Banner at Top Center
        action_y = 50
        action_text = instruction.get('action', 'PROCEED')
        action_color = self.URGENCY_COLORS.get(instruction.get('urgency', 'NONE'), (255, 255, 255))
        
        action_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        action_x = (width - action_size[0]) // 2
        
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (action_x - 15, action_y - 25),
            (action_x + action_size[0] + 15, action_y + 10),
            (0, 0, 0), -1
        )
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
        
        cv2.putText(
            annotated, action_text, (action_x, action_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2
        )
        
        # 4. Reason Banner (if present)
        reason_text = instruction.get('reason', '')
        if reason_text:
            if len(reason_text) > 60:
                reason_text = reason_text[:57] + "..."
            
            reason_size = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            reason_x = (width - reason_size[0]) // 2
            reason_y = action_y + 25
            
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (reason_x - 10, reason_y - 18),
                (reason_x + reason_size[0] + 10, reason_y + 5),
                (0, 0, 0), -1
            )
            annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
            
            cv2.putText(
                annotated, reason_text, (reason_x, reason_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1
            )
            turn_y_adjust = 130
        else:
            turn_y_adjust = 100
        
        # 5. Side Turn Alternative Cards (Shown for stop/yield situations)
        if instruction.get('action') in ['EMERGENCY_STOP', 'YIELD', 'PREPARE_STOP']:
            # Left Card
            left_y = turn_y_adjust
            left_dist = path_analysis['left']['closest_distance']
            if left_dist != float('inf'):
                if left_dist > 30:
                    left_text, left_detail, left_color = "LEFT CLEAR", f"{left_dist:.0f}m", (0, 255, 0)
                elif left_dist > 15:
                    left_text, left_detail, left_color = "LEFT TIGHT", f"{left_dist:.0f}m", (0, 200, 255)
                else:
                    obs_str = ""
                    if path_analysis['left']['obstacles']:
                        obs = min(path_analysis['left']['obstacles'], key=lambda x: x['distance'])
                        obs_str = f"{obs['class']} "
                    left_text, left_detail, left_color = "LEFT BLOCKED", f"{obs_str}{left_dist:.0f}m", (0, 0, 255)
            else:
                left_text, left_detail, left_color = "LEFT CLEAR", "", (0, 255, 0)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, left_y - 20), (150, left_y + 30), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
            
            cv2.putText(annotated, left_text, (15, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            if left_detail:
                cv2.putText(annotated, left_detail, (15, left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
            
            # Right Card
            right_dist = path_analysis['right']['closest_distance']
            if right_dist != float('inf'):
                if right_dist > 30:
                    right_text, right_detail, right_color = "RIGHT CLEAR", f"{right_dist:.0f}m", (0, 255, 0)
                elif right_dist > 15:
                    right_text, right_detail, right_color = "RIGHT TIGHT", f"{right_dist:.0f}m", (0, 200, 255)
                else:
                    obs_str = ""
                    if path_analysis['right']['obstacles']:
                        obs = min(path_analysis['right']['obstacles'], key=lambda x: x['distance'])
                        obs_str = f"{obs['class']} "
                    right_text, right_detail, right_color = "RIGHT BLOCKED", f"{obs_str}{right_dist:.0f}m", (0, 0, 255)
            else:
                right_text, right_detail, right_color = "RIGHT CLEAR", "", (0, 255, 0)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (width - 160, left_y - 20), (width - 10, left_y + 30), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
            
            cv2.putText(annotated, right_text, (width - 155, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
            if right_detail:
                cv2.putText(annotated, right_detail, (width - 155, left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
        
        # 6. Bottom Nearby Objects Telemetry Bar
        nearby_objects = [d for d in detections if d['distance'] < 50]
        if nearby_objects:
            info_y = height - 35
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, info_y - 10), (width, height), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
            
            nearby_sorted = sorted(nearby_objects, key=lambda x: x['distance'])[:5]
            info_text = "Nearby: "
            for obj in nearby_sorted:
                zone = self.zone_manager.get_zone_from_angle(obj['lateral_position'])
                info_text += f"{obj['class']} {obj['distance']:.0f}m ({zone}) | "
            
            info_text = info_text[:-3]
            if len(info_text) > 120:
                info_text = info_text[:117] + "..."
            
            cv2.putText(
                annotated, info_text, (10, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
            )
        
        return annotated
