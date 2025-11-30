"""
YOLO Object Tracking and Movement Analysis

This script uses YOLOv8 to detect and track objects in dashboard camera videos.
It analyzes object movement relative to the ego vehicle and generates:
1. Annotated video with bounding boxes, IDs, and movement directions
2. Summary report of detected objects and their movement patterns

Tracked Objects:
    - Vehicles (car, truck, bus)
    - Pedestrians (person)
    - Animals (dog, cat, cow)

Movement Analysis:
    - Approaching (moving toward ego vehicle)
    - Moving away (moving away from ego vehicle)  
    - Moving left/right

Usage:
    python yolo_object_tracker.py --video path/to/video.mp4 --output output.mp4

Installation:
    pip install ultralytics
"""

import cv2
import argparse
from collections import defaultdict, deque

from ultralytics import YOLO
from config import DEVICE, ALLOWED_CLASSES, YOLO_MODEL_PATH



def track_and_analyze(video_path: str, output_path: str = "output.mp4") -> bool:
    """
    Track objects in video and analyze their movements.
    
    Args:
        video_path: Path to input video
        output_path: Path to save annotated output video
    
    Returns:
        True if successful, False otherwise
    """
    print("=" * 70)
    print("YOLO OBJECT TRACKING AND MOVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SUCCESS] Created output directory: {output_dir}")
    
    # Load model
    print("\n[INFO] Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
        print("[SUCCESS] Model loaded")
    except Exception as e:
        print(f"[ERROR] Error loading YOLO model: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {video_path}")
        print("[HINT] Check if the video file is corrupted or requires additional codecs.")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n[INFO] Video: {width}x{height} @ {fps} FPS")
    
    # Ego vehicle reference point (bottom center)
    ego_x, ego_y = width // 2, height
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking data structures
    unique_objects = defaultdict(set)
    trajectories = defaultdict(lambda: deque(maxlen=5))
    object_directions = defaultdict(dict)
    
    print("\n[INFO] Processing video...\n")
    
    # Process video
    frame_count = 0
    try:
        for result in model.track(
            source=video_path,
            stream=True,
            tracker="bytetrack.yaml",
            conf=0.6,
            verbose=False,
            show=False
        ):
            frame = result.orig_img
            frame_count += 1
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                name = model.names[cls]
                
                if conf < 0.6 or name not in ALLOWED_CLASSES or track_id is None:
                    continue
                
                unique_objects[name].add(track_id)
                
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Track trajectory
                trajectories[track_id].append((cx, cy))
                direction = ""
                
                if len(trajectories[track_id]) >= 2:
                    (px, py) = trajectories[track_id][0]
                    (lx, ly) = trajectories[track_id][-1]
                    dx, dy = lx - px, ly - py
                    
                    # Determine movement direction
                    if abs(dx) > abs(dy):
                        direction = "→ right" if dx > 0 else "← left"
                    else:
                        if dy > 0:
                            direction = "↓ approaching"
                        else:
                            direction = "↑ moving away"
                    
                    object_directions[name][track_id] = direction
                
                # Draw annotations
                color = (0, 255, 0) if name in ["car", "truck", "bus"] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} #{track_id} {direction}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Write frame
            out.write(frame)
    
    except KeyboardInterrupt:
        print("\n[WARNING] Processing interrupted by user")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error during processing: {e}")
        return False
    finally:
        cap.release()
        out.release()
        print(f"\n[INFO] Processed {frame_count} frames")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    
    for name, ids in unique_objects.items():
        print(f"{name}: {len(ids)} unique objects")
    
    # Movement summary
    summary_directions = defaultdict(lambda: defaultdict(int))
    for name, objs in object_directions.items():
        for _, direction in objs.items():
            summary_directions[name][direction] += 1
    
    print("\n" + "=" * 70)
    print("TRAFFIC NARRATIVE")
    print("=" * 70)
    print("Ego car is driving in traffic.\n")
    
    for name, ids in unique_objects.items():
        n = len(ids)
        if n == 0:
            continue
        
        directions = summary_directions[name]
        if directions:
            dir_text = ", ".join([f"{v} {k}" for k, v in directions.items()])
            print(f"Detected {n} {name}(s), with movements: {dir_text}.")
        else:
            print(f"Detected {n} {name}(s).")
    
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Analysis complete!")
    print(f"[SUCCESS] Annotated video saved to: {output_path}")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and analyze objects in video")
    parser.add_argument(
        "--video",
        type=str,
        default="Video/525.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="yolo_output/tracked_output.mp4",
        help="Path to save annotated output video"
    )
    
    args = parser.parse_args()
    
    # Run tracking and analysis
    success = track_and_analyze(args.video, args.output)
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)
