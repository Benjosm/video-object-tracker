import cv2
import torch
import numpy as np
import argparse
from sort_tracker import SortTracker

def main(video_path, output_path, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    tracker = SortTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Use xyxy format

        tracked_objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf > conf_threshold and cls == 0:  # Use dynamic threshold and person class
                w = x2 - x1
                h = y2 - y1
                # SORT expects [x, y, x+w, y+h, conf], but tracker may expect [x1, y1, x2, y2, conf]
                tracked_objects.append([x1, y1, x2, y2, conf])

        tracked_ids = tracker.update(np.array(tracked_objects))

        for tid in tracked_ids:
            x1, y1, x2, y2, obj_id = tid
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {int(obj_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    args = parser.parse_args()

    main(args.input, args.output)
