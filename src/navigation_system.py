"""
Enhanced Navigation System
Master pipeline integrating object detection, spatial analysis, inference, visualization, and guidance.
"""

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from .spatial_zones import SpatialZoneManager, PathClearanceAnalyzer
from .decision_engine import NavigationDecisionEngine
from .visualizer import NavigationVisualizer
from .voice import VoiceAssistant


class EnhancedNavigationSystem:
    """
    Context-aware navigation system with turn opportunity detection,
    spatial clearance analytics, and interactive assistance.
    """
    
    DEFAULT_NAV_OBJECTS = {
        'traffic light': {'priority': 10, 'type': 'signal', 'stop_distance': 15, 'caution_distance': 50},
        'stop sign': {'priority': 10, 'type': 'signal', 'stop_distance': 10, 'caution_distance': 30},
        'car': {'priority': 8, 'type': 'vehicle', 'stop_distance': 8, 'caution_distance': 25},
        'truck': {'priority': 8, 'type': 'vehicle', 'stop_distance': 10, 'caution_distance': 30},
        'bus': {'priority': 8, 'type': 'vehicle', 'stop_distance': 10, 'caution_distance': 30},
        'person': {'priority': 10, 'type': 'pedestrian', 'stop_distance': 5, 'caution_distance': 35},
        'bicycle': {'priority': 7, 'type': 'vehicle', 'stop_distance': 6, 'caution_distance': 20},
        'motorcycle': {'priority': 7, 'type': 'vehicle', 'stop_distance': 7, 'caution_distance': 20},
    }
    
    def __init__(self, model_path='yolov8n.pt', enable_voice=True, nav_objects=None):
        print("\n" + "=" * 80)
        print("INITIALIZING ENHANCED CONTEXT-AWARE NAVIGATION SYSTEM")
        print("=" * 80)
        
        print("\n[1/4] Loading YOLOv8 detection model...")
        self.model = YOLO(model_path)
        print("      ✓ YOLO model loaded")
        
        self.nav_objects = nav_objects or self.DEFAULT_NAV_OBJECTS
        
        print("\n[2/4] Initializing spatial analysis & clearance zones...")
        self.zone_manager = SpatialZoneManager()
        self.zones = self.zone_manager.zones
        self.clearance_analyzer = PathClearanceAnalyzer(self.zone_manager, self.nav_objects)
        print("      ✓ Spatial zones configured")
        
        print("\n[3/4] Initializing voice assistant...")
        self.voice_assistant = VoiceAssistant() if enable_voice else None
        if self.voice_assistant and self.voice_assistant.enabled:
            print("      ✓ Voice assistant ready")
        else:
            print("      ✓ Voice assistant offline (visual mode active)")
            
        print("\n[4/4] Setting up multi-factor decision engine & visualizer...")
        self.decision_engine = NavigationDecisionEngine(self.nav_objects, self.voice_assistant)
        self.visualizer = NavigationVisualizer(self.zone_manager)
        print("      ✓ Decision engine and HUD visualizer initialized")
        
        # State tracking
        self.camera_active = False
        self.fps = 0.0
        self.latest_instruction = None
        self.latest_detections = []
        
        print("\n✓ Navigation system initialized successfully!")
    
    def estimate_real_distance(self, bbox, image_width, image_height, object_class):
        """Estimate distance in meters."""
        return self.zone_manager.estimate_real_distance(bbox, image_width, image_height, object_class)
    
    def calculate_lateral_position(self, bbox, image_width):
        """Calculate lateral angle."""
        return self.zone_manager.calculate_lateral_position(bbox, image_width)
    
    def get_zone_from_angle(self, angle):
        """Map lateral angle to zone name."""
        return self.zone_manager.get_zone_from_angle(angle)
    
    def analyze_path_clearance(self, detections):
        """Compute zone clearance status."""
        return self.clearance_analyzer.analyze_path_clearance(detections)
    
    def generate_detailed_instruction(self, detections, current_speed=30):
        """Synthesize navigation recommendations."""
        path_analysis = self.analyze_path_clearance(detections)
        return self.decision_engine.generate_detailed_instruction(detections, path_analysis, current_speed)
    
    def visualize_instruction(self, img, instruction, detections):
        """Overlay HUD on image."""
        path_analysis = self.analyze_path_clearance(detections)
        return self.visualizer.render(img, instruction, detections, path_analysis)
    
    def process_frame(self, frame, current_speed=30):
        """
        Process a single image frame for real-time analysis.
        
        Returns:
            tuple: (annotated_frame, detections, instruction)
        """
        if frame is None:
            return None, [], {}
        
        height, width = frame.shape[:2]
        results = self.model(frame, verbose=False)
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
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x) for x in bbox],
                    'distance': float(distance),
                    'lateral_position': float(lateral_pos),
                    'priority': priority,
                    'object_type': self.nav_objects[class_name]['type']
                })
        
        path_analysis = self.analyze_path_clearance(detections)
        instruction = self.decision_engine.generate_detailed_instruction(detections, path_analysis, current_speed)
        annotated_frame = self.visualizer.render(frame, instruction, detections, path_analysis)
        
        return annotated_frame, detections, instruction
    
    def process_image(self, image_path, current_speed=30):
        """Process a single image file and print comprehensive terminal diagnostics."""
        img_name = Path(image_path).name
        print(f"\n{'=' * 80}")
        print(f"PROCESSING: {img_name}")
        print(f"{'=' * 80}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"✗ Failed to load image: {image_path}")
            return None, [], {}
        
        processed_img, detections, instruction = self.process_frame(img, current_speed)
        
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
        path_analysis = self.analyze_path_clearance(detections)
        
        for side, symbol in [('left', '← LEFT: '), ('right', '→ RIGHT:')]:
            dist = path_analysis[side]['closest_distance']
            if dist != float('inf'):
                if dist > 30:
                    status = f"✓ {dist:.0f}m CLEAR"
                elif dist > 15:
                    status = f"⚠ {dist:.0f}m TIGHT"
                else:
                    obs_str = ""
                    if path_analysis[side]['obstacles']:
                        obs = min(path_analysis[side]['obstacles'], key=lambda x: x['distance'])
                        obs_str = f" ({obs['class']})"
                    status = f"✗ {dist:.0f}m BLOCKED{obs_str}"
            else:
                status = "✓ CLEAR"
            print(f"   {symbol} {status}")
            
        return processed_img, detections, instruction
    
    def start_camera_feed(self, camera_id=0, window_name="Enhanced Navigation System"):
        """Run real-time video stream with camera input and interactive keyboard controls."""
        print(f"\n🎥 STARTING REAL-TIME CAMERA FEED (Camera {camera_id})")
        print("   Press 'q' to quit, 's' to save current frame, 'p' for info, 'v' for voice")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.camera_active = True
        results_dir = Path('camera_results')
        results_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        start_time = time.time()
        
        print("\n🚀 REAL-TIME PROCESSING STARTED...")
        
        try:
            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Error: Could not read frame from camera")
                    break
                
                frame_count += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    self.fps = frame_count / (current_time - start_time)
                    frame_count = 0
                    start_time = current_time
                
                simulated_speed = 30 + (10 * math.sin(time.time() * 0.5))
                processed_frame, detections, instruction = self.process_frame(frame, simulated_speed)
                
                self.latest_instruction = instruction
                self.latest_detections = detections
                
                # Render FPS overlay
                fps_text = f"FPS: {self.fps:.1f} | Speed: {simulated_speed:.0f} km/h"
                cv2.putText(
                    processed_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
                
                cv2.imshow(window_name, processed_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting camera feed...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = results_dir / f"camera_capture_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), processed_frame)
                    print(f"✓ Frame saved: {save_path}")
                elif key == ord('p'):
                    print(f"\n[CURRENT] {instruction['primary_instruction']}")
                    print(f"          Action: {instruction['action']} | Urgency: {instruction['urgency']}")
                elif key == ord('v') and self.voice_assistant:
                    spoken = self.decision_engine.generate_voice_instruction(instruction)
                    self.voice_assistant.speak_instruction(spoken, force=True)
                    print(f"🔊 Spoken: {spoken}")
        finally:
            self.camera_active = False
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Camera feed closed successfully")
            
    def process_dataset(self, dataset_path, output_dir='results'):
        """Process image directory, export annotated frames, and save summary reports."""
        dataset_dir = Path(dataset_path) / 'images'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dataset_dir.glob(ext)))
        
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE NAVIGATION ANALYSIS")
        print("=" * 80)
        print(f"\n✓ Found {len(image_files)} images to analyze")
        
        results = []
        speeds = [30, 40, 50, 25, 35, 45, 15, 40, 30]  # Representative vehicle speeds
        
        for idx, img_path in enumerate(sorted(image_files)):
            speed = speeds[idx % len(speeds)]
            processed_img, detections, instruction = self.process_image(img_path, current_speed=speed)
            
            if processed_img is not None:
                output_filename = f"enhanced_nav_{img_path.name}"
                output_filepath = output_dir / output_filename
                cv2.imwrite(str(output_filepath), processed_img)
                
                results.append({
                    'image': img_path.name,
                    'current_speed': speed,
                    'detections': detections,
                    'instruction': instruction,
                    'total_objects': len(detections)
                })
        
        report_path = output_dir / 'enhanced_navigation_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.generate_summary_report(results, output_dir)
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"✓ Images processed: {len(results)}")
        print(f"✓ Results saved in: {output_dir}")
        print(f"✓ Detailed report: {report_path}")
        print("=" * 80 + "\n")
        
        return results
    
    def generate_summary_report(self, results, output_dir):
        """Produce statistical breakdown of evaluated scenes."""
        output_dir = Path(output_dir)
        summary = {
            'total_images': len(results),
            'action_distribution': defaultdict(int),
            'urgency_distribution': defaultdict(int),
            'safe_turns_left': 0,
            'safe_turns_right': 0,
            'critical_situations': 0,
            'average_objects_per_scene': 0.0,
            'most_common_objects': defaultdict(int)
        }
        
        total_objects = 0
        for result in results:
            instruction = result['instruction']
            summary['action_distribution'][instruction['action']] += 1
            summary['urgency_distribution'][instruction['urgency']] += 1
            
            turns = instruction.get('turn_opportunities', {})
            if '✓' in turns.get('left', ''):
                summary['safe_turns_left'] += 1
            if '✓' in turns.get('right', ''):
                summary['safe_turns_right'] += 1
            
            if instruction.get('urgency') == 'CRITICAL':
                summary['critical_situations'] += 1
            
            total_objects += result['total_objects']
            for det in result['detections']:
                summary['most_common_objects'][det['class']] += 1
        
        summary['average_objects_per_scene'] = total_objects / len(results) if results else 0.0
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
