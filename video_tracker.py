import cv2
import torch
import numpy as np
from sort_tracker import Sort  # Use correct class name Sort instead of SortTracker
import argparse
import time

def main(input_path, output_path, conf_threshold=0.5):
    # Initialize the YOLOv5n model as specified in project plan
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.conf = conf_threshold  # Set confidence threshold

    # Initialize tracker
    tracker = Sort()

    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Fixed typo: outputNewton's -> output_path

    if not out.isOpened():
        print(f"Error: Cannot create output video file {output_path}")
        return

    frame_count = 0
    total_processing_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # Perform detection
        results = model(frame)

        # Extract detections from results using xyxy format
        # results.xyxy[0] contains tensor of detections: (x1, y1, x2, y2, confidence, class)
        detections_tensor = results.xyxy[0]
        detections_numpy = detections_tensor.cpu().numpy()

        # Convert detections to numpy array of [x, y, w, h, confidence]
        detections = []
        for detection in detections_numpy:
            x1, y1, x2, y2, conf, cls = detection
            w = x2 - x1
            h = y2 - y1
            # Track all object classes (remove cls == 0 filter), use conf threshold from args
            if conf > conf_threshold:
                detections.append([x1, y1, w, h, conf])

        # Update tracker with current detections
        tracked_objects = tracker.update(np.array(detections))

        # Draw tracking results
        for track in tracked_objects:
            x, y, w, h, track_id = track.astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Calculate processing time
        end_time = time.time()
        frame_processing_time = end_time - start_time
        total_processing_time += frame_processing_time

        # Print frame counter and FPS to stdout as required
        print(f"Frame {frame_count} processed in {frame_processing_time:.3f}s, "
              f"FPS: {1.0/max(frame_processing_time, 1e-6):.1f}")

    # Release resources
    cap.release()
    out.release()

    # Final statistics
    if frame_count > 0:
        avg_fps = frame_count / total_processing_time
        print(f"Processing complete. Total frames: {frame_count}, Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Object Tracker")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    args = parser.parse_args()

    main(args.input, args.output, args.conf)
