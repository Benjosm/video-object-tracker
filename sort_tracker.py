import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def iou(bb_test, bb_gt):
    """
    Computes the IoU between two bounding boxes in xywh format.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
    yy2 = np.minimum(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = bb_test[2] * bb_test[3]
    area_gt = bb_gt[2] * bb_gt[3]
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using an initial bounding box in [x, y, w, h] format.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4) # state: [x, y, s, r, x_dot, y_dot, s_dot]
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:,2:] *= 10. # measurement uncertainty
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = bbox.reshape((4,1)) # initialize state with the provided bbox (xywh)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the tracker with an observed bounding box in xywh format.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

    def predict(self):
        """
        Returns the predicted bounding box in xywh format.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].reshape((4,)))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate in xywh format.
        """
        return self.kf.x[:4].reshape((4,))

class Sort:
    """
    SORT: A Simple, Online, Real-time Tracker.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the Sort object with maximum age and minimum hits parameters.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
        dets - a numpy array of detections in the format [[x,y,w,h,conf,cls], [x,y,w,h,conf,cls],...]
        Modified version to return [x,y,w,h,track_id,class] for per-class tracking.
        Returns a numpy array of tracked objects in the format [[x,y,w,h,track_id,cls], [x,y,w,h,track_id,cls],...]
        """
        self.frame_count += 1
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections with trackers using IoU
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4]) # Update with xywh

        # Create and init new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4]) # Initialize with xywh
            # Include the class ID from the detection
            trk.class_id = dets[i, 5]
            self.trackers.append(trk)

        # Remove dead tracker predictions
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (self.frame_count >= self.min_hits or self.frame_count <= self.min_hits):
                # Append the track ID and class ID, using the tracker's stored class ID
                ret.append(np.concatenate((d, [trk.id], [trk.class_id]))) # [x,y,w,h,track_id,cls]
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.array(ret)

    def associate_detections_to_trackers(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections, and unmatched_trackers
        """
        if len(trackers)==0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)),dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d,t] = iou(det, trk)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.asarray(matched_indices).T
        else:
            matched_indices = np.empty((0,2),dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if len(matches)==0:
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
