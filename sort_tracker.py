import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize the SORT tracker.
        :param max_age: Maximum number of frames to keep a track without detection.
        :param min_hits: Minimum number of detections to confirm a track.
        :param iou_threshold: Minimum IOU for association.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        """
        Update the tracker with new detections.
        :param detections: List of [x1, y1, x2, y2, score] detections.
        :return: List of tracked objects as [x1, y1, x2, y2, track_id].
        """
        # Predict new positions for existing tracks
        predicted_tracks = []
        for track in self.tracks:
            track['kalman'].predict()
            predicted_tracks.append(track['kalman'].x)

        # Associate detections with tracks
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(detections, predicted_tracks)

        # Update matched tracks with detections
        for d, t in matched:
            self.tracks[t]['kalman'].update(detections[d][:4])
            self.tracks[t]['hits'] += 1
            self.tracks[t]['age'] = 0

        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            self._initiate_track(detections[d])

        # Mark unmatched tracks for removal
        for t in unmatched_tracks:
            self.tracks[t]['age'] += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age and t['hits'] >= self.min_hits]

        # Generate output
        outputs = []
        for track in self.tracks:
            bbox = track['kalman'].x[:4].flatten()
            outputs.append([bbox[0], bbox[1], bbox[2], bbox[3], track['id']])
        
        return np.array(outputs) if len(outputs) > 0 else np.empty((0, 5))

    def _associate_detections_to_tracks(self, detections, predicted_tracks):
        """
        Associate detections to tracked objects using IOU as cost.
        """
        if len(predicted_tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0), dtype=int), np.arange(len(predicted_tracks))

        # Compute IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(predicted_tracks)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(predicted_tracks):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])

        # Use Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched = []
        unmatched_dets = list(set(range(len(detections))) - set(row_ind))
        unmatched_tracks = list(set(range(len(predicted_tracks))) - set(col_ind))

        # Filter matches using IOU threshold
        for row, col in zip(row_ind, col_ind):
            if iou_matrix[row, col] < self.iou_threshold:
                unmatched_dets.append(row)
                unmatched_tracks.append(col)
            else:
                matched.append((row, col))

        return np.array(matched), np.array(unmatched_dets), np.array(unmatched_tracks)

    def _initiate_track(self, detection):
        """
        Create a new track for a detection.
        """
        kalman = KalmanFilter(dim_x=8, dim_z=4)
        kalman.x[:4] = detection[:4].reshape(4, 1)
        kalman.P[:4, :4] *= 1000.0
        kalman.P[4:, 4:] *= 1000.0
        kalman.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]])
        kalman.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0]])
        kalman.R[2:, 2:] *= 10.0
        kalman.Q[-1, -1] *= 0.01
        kalman.Q[4:, 4:] *= 0.01

        self.tracks.append({
            'id': self.next_id,
            'kalman': kalman,
            'hits': 1,
            'age': 0
        })
        self.next_id += 1

    def _iou(self, bbox_a, bbox_b):
        """
        Compute Intersection over Union (IOU) of two bounding boxes.
        """
        x1, y1, x2, y2 = bbox_a
        x1p, y1p, x2p, y2p = bbox_b

        inter_x1 = max(x1, x1p)
        inter_y1 = max(y1, y1p)
        inter_x2 = min(x2, x2p)
        inter_y2 = min(y2, y2p)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (x2p - x1p) * (y2p - y1p)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area
