import cv2
import torch
import numpy as np
import argparse
from sort_tracker import Sort
import os

def main():
    parser = argparse.ArgumentParser(description="Video Object Tracker using YOLOv5 and SORT")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    args = parser.parse_args()

    # Load YOLOv5 model from ultralytics
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Initialize SORT tracker
    tracker = Sort()

    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 detection
        results = model(frame)
        detections = results.pred[0].cpu().numpy()  # x1, y1, x2, y2, conf, class

        # Filter by confidence
        high_conf_detections = detections[detections[:, 4] >= args.conf]
        # Convert to [x1, y1, x2, y2, score]
        bboxes = high_conf_detections[:, :5]

        # Update tracker
        if len(bboxes) > 0:
            tracked_objects = tracker.update(bboxes)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

        # Draw bounding boxes and IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            x1, y1, x2, y2, obj_id = map(int, [x1, y1, x2, y2, obj_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write frame to output
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
