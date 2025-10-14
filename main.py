import cv2
import torch
import numpy as np
import argparse
import os
from sort_tracker import Sort

def main():
    parser = argparse.ArgumentParser(description="Video object tracking with YOLOv5 and SORT")
    
    parser.add_argument(
        'input',
        help='Input video file path or device ID (e.g., 0 for webcam)'
    )
    
    parser.add_argument(
        'output',
        help='Output video file path (e.g., output.mp4)'
    )

    parser.add_argument(
        '--model',
        default='yolov5s',
        help='YOLOv5 model variant (e.g., yolov5s, yolov5m, yolov5l)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for detection'
    )

    args = parser.parse_args()

    # Handle both file paths and webcam/device IDs
    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)

    # Check input file existence (unless it's a webcam)
    if isinstance(input_source, str) and not os.path.exists(input_source):
        raise FileNotFoundError(f"Input video not found: {input_source}")

    # Load YOLOv5 model
    print(f"Loading YOLOv5 model: {args.model}")
    model = torch.hub.load('ultralytics/yolov5', args.model, pretrained=True)
    model.conf = args.conf  # Set confidence threshold

    # Initialize SORT tracker
    tracker = Sort()

    # Open video file or capture device
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {input_source}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video: {args.input}")
    print(f"Saving result to: {args.output}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...", end='\r')

        # Perform inference
        results = model(frame)

        # Extract detections: [x1, y1, x2, y2, confidence, class]
        detections = results.pred[0].cpu().numpy()

        # Filter only for 'person' class (COCO class 0)
        person_detections = detections[detections[:, 5] == 0]
        
        # Convert to [x, y, w, h, score] for SORT
        if len(person_detections) > 0:
            dets_xywh = person_detections[:, :5].copy()  # x1, y1, x2, y2, conf
            dets_xywh[:, 2] -= dets_xywh[:, 0]  # width (x2 - x1)
            dets_xywh[:, 3] -= dets_xywh[:, 1]  # height (y2 - y1)
        else:
            dets_xywh = np.empty((0, 5))

        # Update tracker
        tracked_objects = tracker.update(dets_xywh)

        # Draw bounding boxes and IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames. Output saved to {args.output}")

if __name__ == "__main__":
    main()
