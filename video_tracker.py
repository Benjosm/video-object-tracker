import argparse
import time
import cv2
import torch
import numpy as np
from sort_tracker import Sort

def main():
    parser = argparse.ArgumentParser(description="Video object tracker")
    parser.add_argument("--input", required=True, help="Input video file path or camera index")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    args = parser.parse_args()

    # Load YOLOv5 nano model for CPU efficiency
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.input}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    # Initialize SORT tracker for all classes
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Initialize timing and frame counter
    frame_count = 0
    start_time = time.time()

    print("Starting video processing... Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Perform object detection
            results = model(frame)

            # Extract detections: results.xyxy[0] returns [x1, y1, x2, y2, conf, cls]
            detections_xyxy = results.xyxy[0].cpu().numpy()

            # Prepare detections for SORT: convert xyxy to xywh and filter by conf threshold (use CLI arg)
            # Also, preserve class id
            detections = []
            for *xyxy, conf, cls in detections_xyxy:
                if conf >= args.conf:  # Use confidence threshold from arguments
                    # Convert xyxy to xywh
                    x1, y1, x2, y2 = map(int, xyxy)
                    w = x2 - x1
                    h = y2 - y1
                    # Include confidence and class ID
                    detections.append([x1, y1, w, h, conf, int(cls)])

            # Update tracker with current frame's detections (all classes)
            tracked_objects = tracker.update(np.array(detections))

            # Draw bounding boxes and track IDs on the frame
            for obj in tracked_objects:
                x, y, w, h, track_id, cls = map(int, obj[:6])  # Extract xywh, track ID, and class
                # Use a color based on the class ID for better visualization
                color = (255, 0, 0) if cls % 3 == 0 else (0, 255, 0) if cls % 3 == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'ID {track_id} | Cls {cls}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Write the frame to the output video
            out.write(frame)

            # Calculate and print FPS and frame number every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                print(f"Frame: {frame_count}, FPS: {current_fps:.2f}")

            # Display the resulting frame (optional)
            # cv2.imshow('Tracked Video', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Final output
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processing complete. Total frames: {frame_count}, Total time: {total_time:.2f}s, Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
