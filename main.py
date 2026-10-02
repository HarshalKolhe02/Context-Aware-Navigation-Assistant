#!/usr/bin/env python3
"""
Context-Aware Navigation Assistant
Main Entry Point
"""

import argparse
import sys
from pathlib import Path

from src import RealDatasetDownloader, EnhancedNavigationSystem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Context-Aware Navigation Assistant - Visual Perception & Real-Time Decision Guidance"
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'dataset', 'camera', 'both'],
        default='interactive',
        help="Operation mode: 'dataset', 'camera', 'both', or 'interactive' (default)"
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='dataset',
        help="Path to evaluation dataset directory (default: 'dataset')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help="Directory to store processed results and JSON reports (default: 'results')"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help="YOLO model checkpoint path (default: 'yolov8n.pt')"
    )
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help="Disable spoken audio voice assistant output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 80)
    print(" " * 15 + "CONTEXT-AWARE NAVIGATION ASSISTANT")
    print(" " * 18 + "Perception & Spatial Decision Engine")
    print("=" * 80)
    
    print("\n🎯 SYSTEM SPECIFICATIONS:")
    print("   • Primary Assumption: Observer traveling straight ahead")
    print("   • Perception: YOLOv8 real-time object detection")
    print("   • Spatial Mapping: 5-zone angular segmentation (-90° to +90°)")
    print("   • Monocular Depth: Geometry and perspective-based distance estimation")
    print("   • Decision Engine: Multi-factor continuous rule-based inference matrix")
    print("   • Routing Analysis: Safe turn opportunity and obstacle clearance scoring")
    print("   • Multimodal Output: Augmented HUD overlay and non-blocking voice guidance")
    
    # Ensure sample evaluation dataset exists
    dataset_images_dir = Path(args.dataset_path) / 'images'
    if not dataset_images_dir.exists() or not any(dataset_images_dir.iterdir()):
        downloader = RealDatasetDownloader()
        downloader.download_dataset(args.dataset_path)
    
    # Initialize the navigation assistant
    enable_voice = not args.no_voice
    nav_system = EnhancedNavigationSystem(
        model_path=args.model,
        enable_voice=enable_voice
    )
    
    # Dispatch according to selected mode
    if args.mode == 'dataset':
        nav_system.process_dataset(args.dataset_path, output_dir=args.output_dir)
        return
    
    elif args.mode == 'camera':
        nav_system.start_camera_feed(camera_id=args.camera_id)
        return
        
    elif args.mode == 'both':
        nav_system.process_dataset(args.dataset_path, output_dir=args.output_dir)
        print("\n" + "=" * 50)
        print("SWITCHING TO REAL-TIME CAMERA MODE")
        print("=" * 50)
        nav_system.start_camera_feed(camera_id=args.camera_id)
        return
    
    # Interactive mode (default)
    while True:
        print("\n🔧 SELECT PROCESSING MODE:")
        print("   1. Real-time Camera Feed (with voice)")
        print("   2. Process Downloaded Dataset")
        print("   3. Both (Camera + Dataset)")
        print("   4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if choice == '1':
            camera_input = input("Enter camera ID (default 0 for webcam): ").strip()
            cam_id = int(camera_input) if camera_input.isdigit() else 0
            nav_system.start_camera_feed(camera_id=cam_id)
            
        elif choice == '2':
            nav_system.process_dataset(args.dataset_path, output_dir=args.output_dir)
            
        elif choice == '3':
            nav_system.process_dataset(args.dataset_path, output_dir=args.output_dir)
            print("\n" + "=" * 50)
            print("SWITCHING TO REAL-TIME CAMERA MODE")
            print("=" * 50)
            nav_system.start_camera_feed(camera_id=args.camera_id)
            
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            
    print("\n" + "=" * 80)
    print("✓ NAVIGATION ASSISTANT SESSION CONCLUDED")
    print("=" * 80)


if __name__ == "__main__":
    main()