#!/usr/bin/env python3
"""
Context-Aware Navigation Assistant - Legacy Test Suite 2
Demonstrates multi-zone visual navigation with real-time camera feed and text-to-speech voice instructions.
"""

from src import RealDatasetDownloader, EnhancedNavigationSystem


def main():
    print("\n" + "=" * 80)
    print(" " * 15 + "ENHANCED CONTEXT-AWARE NAVIGATION SYSTEM")
    print(" " * 18 + "Vision, Spatial Analysis & Voice Guidance")
    print("=" * 80)
    
    print("\n🎯 SYSTEM CAPABILITIES:")
    print("   • Real-time camera feed processing")
    print("   • Image dataset processing")
    print("   • Multi-zone spatial analysis (5 zones)")
    print("   • Priority-based decision making")
    print("   • Turn opportunity detection")
    print("   • Emergency threat detection")
    print("   • Voice instructions for navigation")
    
    # Download dataset if necessary
    downloader = RealDatasetDownloader()
    dataset_path = 'dataset'
    downloader.download_dataset(dataset_path)
    
    # Initialize navigation system with voice assistant
    nav_system = EnhancedNavigationSystem(enable_voice=True)
    
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
            camera_id = input("Enter camera ID (default 0 for webcam): ").strip()
            cam_id = int(camera_id) if camera_id.isdigit() else 0
            nav_system.start_camera_feed(cam_id)
            
        elif choice == '2':
            nav_system.process_dataset(dataset_path)
            
        elif choice == '3':
            nav_system.process_dataset(dataset_path)
            print("\n" + "=" * 50)
            print("SWITCHING TO REAL-TIME CAMERA MODE")
            print("=" * 50)
            nav_system.start_camera_feed(0)
            
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            
    print("\n" + "=" * 80)
    print("✓ NAVIGATION SYSTEM DEMO COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()