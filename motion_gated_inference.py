#!/usr/bin/env python3
"""
motion_gated_inference.py — Stage 5: Motion-Gated Inference with SORT Tracking

Standalone script for benchmarking motion-gated + SORT track propagation
against full-frame (always-detect) baseline on UAV surveillance video.

Pipeline per frame:
  1. Motion gate: cheap frame differencing (~0.3 ms on ARM)
  2. If motion detected OR fallback interval reached:
       → Run quantized ONNX detector
       → Hungarian-match detections to existing SORT tracks
  3. If no motion AND tracks are active:
       → Kalman-predict track positions (microseconds, no inference)
  4. Output annotated video + benchmark JSON

Usage:
  # Benchmark on a video (gated+SORT vs full-frame baseline)
  python motion_gated_inference.py \
      --model model_mixed_precision_quantized.onnx \
      --video drone_surveillance.mp4 \
      --output gated_output.mp4 \
      --benchmark

  # With MOTA/IDF1 evaluation (provide COCO ground-truth JSON)
  python motion_gated_inference.py \
      --model model_mixed_precision_quantized.onnx \
      --video drone_surveillance.mp4 \
      --coco-annotations val.json \
      --image-dir /path/to/frames \
      --eval

  # Process a directory of frame images instead of a video
  python motion_gated_inference.py \
      --model model_mixed_precision_quantized.onnx \
      --image-dir /path/to/frames \
      --benchmark
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required: pip install onnxruntime")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# SECTION 1: SORT TRACKER (Kalman Filter + Hungarian Assignment)
# ============================================================================

class KalmanBoxTracker:
    """
    Single-object Kalman filter tracker using a constant-velocity model.

    State vector (8-dim): [cx, cy, aspect_ratio, height, vcx, vcy, var, vh]
    Measurement (4-dim):  [cx, cy, aspect_ratio, height]

    This is the standard SORT formulation (Bewley et al., 2016).
    """

    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """
        Args:
            bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        """
        # Convert to [cx, cy, aspect_ratio, height]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect = w / max(h, 1e-6)

        # Initialize state
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = [cx, cy, aspect, h]

        # Covariance: higher uncertainty for velocity components
        self.covariance = np.eye(8, dtype=np.float64) * 10.0
        self.covariance[4:, 4:] *= 100.0  # More uncertain about velocity

        # Transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01
        self.Q[4:, 4:] += np.eye(4) * 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0  # Noisier aspect ratio and height

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.cls = None  # Will be set by detector

    def predict(self) -> Tuple[float, float, float, float]:
        """Predict next state and return predicted bounding box (x1, y1, x2, y2)."""
        self.mean = self.F @ self.mean
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

        # Handle negative height/aspect (numerical instability safeguard)
        if self.mean[3] <= 1:
            self.mean[3] = 1.0
        if self.mean[2] <= 0:
            self.mean[2] = 0.01

        return self._to_xyxy()

    def update(self, bbox: Tuple[float, float, float, float],
               cls: Optional[int] = None, conf: Optional[float] = None):
        """Update with a new measurement."""
        z = np.zeros(4, dtype=np.float64)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z[0] = cx
        z[1] = cy
        z[2] = w / max(h, 1e-6)
        z[3] = h

        # Kalman update
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.mean
        self.mean = self.mean + K @ innovation
        I = np.eye(8)
        self.covariance = (I - K @ self.H) @ self.covariance

        self.time_since_update = 0
        self.hits += 1

        if cls is not None:
            self.cls = cls
        if conf is not None:
            self.conf = conf

    def _to_xyxy(self) -> Tuple[float, float, float, float]:
        cx, cy, aspect, h = self.mean[:4]
        w = aspect * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class SortTracker:
    """
    SORT (Simple Online and Realtime Tracking) multi-object tracker.

    Maintains a set of KalmanBoxTracker instances. On each frame:
    1. Predict all track positions
    2. Match new detections to predicted tracks via IoU + Hungarian assignment
    3. Update matched tracks, create new tracks for unmatched detections
    4. Remove tracks that haven't been updated for too long

    Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Args:
            max_age: Max frames a track can survive without detection before deletion.
            min_hits: Min consecutive hits before a track is confirmed.
            iou_threshold: Min IoU for a detection to match a track.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self):
        """Reset tracker state (call at the start of each video)."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Process one frame of detections.

        Args:
            detections: List of dicts with 'bbox' [x1,y1,x2,y2], 'confidence', 'class'.

        Returns:
            List of active tracks with 'bbox', 'id', 'class', 'confidence'.
        """
        self.frame_count += 1

        # Step 1: Predict all existing trackers
        predicted_boxes = []
        for tr in self.trackers:
            box = tr.predict()
            predicted_boxes.append(box)

        # Step 2: If no detections, return predicted positions (track propagation)
        if len(detections) == 0:
            # Remove stale tracks
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return self._get_active_tracks()

        # Step 3: Build IoU cost matrix
        det_boxes = np.array([d['bbox'] for d in detections], dtype=np.float64)
        if len(predicted_boxes) > 0:
            trk_boxes = np.array(predicted_boxes, dtype=np.float64)
            iou_matrix = self._iou_batch(trk_boxes, det_boxes)

            # Hungarian: minimize negative IoU
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    det = detections[c]
                    self.trackers[r].update(det['bbox'], det.get('class'), det.get('confidence'))
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # Unmatched detections → new tracks
            for c in range(len(detections)):
                if c not in matched_dets:
                    new_trk = KalmanBoxTracker(detections[c]['bbox'])
                    new_trk.cls = detections[c].get('class')
                    new_trk.conf = detections[c].get('confidence')
                    self.trackers.append(new_trk)

            # Unmatched tracks are left to age (handled below)
        else:
            # No existing trackers — all detections become new tracks
            for det in detections:
                new_trk = KalmanBoxTracker(det['bbox'])
                new_trk.cls = det.get('class')
                new_trk.conf = det.get('confidence')
                self.trackers.append(new_trk)

        # Step 4: Remove stale tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return self._get_active_tracks()

    def predict_only(self) -> List[Dict]:
        """
        Kalman-predict track positions WITHOUT running the detector.
        This is called during motion-gated skip frames to propagate tracks.

        Returns:
            List of predicted tracks with 'bbox', 'id', 'class'.
        """
        self.frame_count += 1
        for tr in self.trackers:
            tr.predict()

        # Remove stale tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Dict]:
        """Return confirmed tracks (hits >= min_hits or recently updated)."""
        results = []
        for tr in self.trackers:
            # Report track if confirmed, or if it was just updated this frame
            if tr.hits >= self.min_hits or self.frame_count <= self.min_hits:
                if tr.time_since_update == 0 or tr.time_since_update <= self.max_age:
                    box = tr._to_xyxy()
                    results.append({
                        'bbox': box,
                        'id': tr.id,
                        'class': tr.cls,
                        'confidence': getattr(tr, 'conf', 0.0),
                        'time_since_update': tr.time_since_update,
                    })
        return results

    @staticmethod
    def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of [x1,y1,x2,y2] boxes."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)))

        # Expand for broadcasting
        a = boxes_a[:, None, :]  # [Na, 1, 4]
        b = boxes_b[None, :, :]  # [1, Nb, 4]

        inter_x1 = np.maximum(a[..., 0], b[..., 0])
        inter_y1 = np.maximum(a[..., 1], b[..., 1])
        inter_x2 = np.minimum(a[..., 2], b[..., 2])
        inter_y2 = np.minimum(a[..., 3], b[..., 3])

        inter_w = np.clip(inter_x2 - inter_x1, 0, None)
        inter_h = np.clip(inter_y2 - inter_y1, 0, None)
        inter_area = inter_w * inter_h

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        union = area_a[:, None] + area_b[None, :] - inter_area
        iou = inter_area / np.clip(union, 1e-6, None)
        return iou


# ============================================================================
# SECTION 2: ONNX YOLO DETECTOR
# ============================================================================

class ONNXDetector:
    """
    YOLOv8/v11 ONNX inference wrapper for CPU-only deployment.

    Handles preprocessing (resize, normalize, HWC→CHW), inference via
    onnxruntime, and postprocessing (NMS, coordinate rescaling).
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path: Path to .onnx model file.
            imgsz: Input size the model was trained/exported at.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            num_classes: Number of classes in the model.
            class_names: Optional list of class name strings.
        """
        # Create ONNX Runtime session (CPU-only, single thread for edge realism)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

        # Auto-detect input size if possible
        if len(self.input_shape) == 4 and self.input_shape[2] is not None:
            self.imgsz = int(self.input_shape[2])
        else:
            self.imgsz = imgsz

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a BGR frame (OpenCV format).

        Args:
            frame: BGR image, HWC, uint8.

        Returns:
            List of detections: {'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': int}
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize, normalize, HWC→CHW
        input_data = self._preprocess(frame)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Postprocess: NMS, rescale to original size
        detections = self._postprocess(outputs, orig_w, orig_h)
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize with letterbox, normalize to [0,1], HWC→CHW, add batch dim."""
        h, w = frame.shape[:2]
        target = self.imgsz

        # Letterbox resize (preserves aspect ratio)
        scale = min(target / h, target / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_top = (target - new_h) // 2
        pad_bottom = target - new_h - pad_top
        pad_left = (target - new_w) // 2
        pad_right = target - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Store for coordinate rescaling
        self._scale = scale
        self._pad_top = pad_top
        self._pad_left = pad_left

        # Normalize and convert to CHW
        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)  # HWC → CHW
        padded = padded[np.newaxis, ...]  # Add batch dim

        return padded

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        orig_w: int,
        orig_h: int,
    ) -> List[Dict]:
        """
        Postprocess ONNX output: apply NMS, rescale boxes to original image.

        Handles both standard YOLOv8 output [1, 4+nc, num_anchors] and
        NMS-included output formats.
        """
        output = outputs[0]

        # Detect output format
        # Standard YOLOv8/v11: [1, 4+nc, num_anchors] or [1, num_anchors, 4+nc]
        if output.ndim == 3:
            if output.shape[1] == 4 + self.num_classes:
                # [1, 4+nc, num_anchors] → transpose
                output = output[0].T  # [num_anchors, 4+nc]
            elif output.shape[2] == 4 + self.num_classes:
                output = output[0]  # [num_anchors, 4+nc]
            elif output.shape[1] < output.shape[2]:
                # [1, small, large] → likely [1, 4+nc, num_anchors]
                output = output[0].T
            else:
                output = output[0]

        # Now output should be [num_anchors, 4+nc]
        # Box coords: cx, cy, w, h (in letterbox-padded input space)
        # Class scores: columns 4 to 4+nc

        boxes_raw = output[:, :4]
        scores_raw = output[:, 4:4 + self.num_classes]

        # Get max class score per anchor
        max_scores = scores_raw.max(axis=1)
        class_ids = scores_raw.argmax(axis=1)

        # Confidence filter
        mask = max_scores >= self.conf_threshold
        boxes_raw = boxes_raw[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS (per-class to avoid suppressing cross-class detections)
        keep = self._nms(boxes, max_scores, class_ids)
        boxes = boxes[keep]
        scores = max_scores[keep]
        classes = class_ids[keep]

        # Rescale from letterbox space to original image space
        scale = getattr(self, '_scale', 1.0)
        pad_top = getattr(self, '_pad_top', 0)
        pad_left = getattr(self, '_pad_left', 0)

        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox': boxes[i].tolist(),
                'confidence': float(scores[i]),
                'class': int(classes[i]),
            })

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[int]:
        """Per-class Non-Maximum Suppression."""
        keep = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            order = cls_scores.argsort()[::-1]
            cls_keep = []

            while len(order) > 0:
                i = order[0]
                cls_keep.append(i)

                if len(order) == 1:
                    break

                # Compute IoU with remaining boxes
                remaining = order[1:]
                iou = self._iou_single(
                    cls_boxes[i],
                    cls_boxes[remaining],
                )

                # Keep boxes with IoU below threshold
                order = remaining[iou < self.iou_threshold]

            for idx in cls_keep:
                keep.append(cls_indices[idx])

        return sorted(keep)

    @staticmethod
    def _iou_single(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """IoU between one box and a set of boxes."""
        inter_x1 = np.maximum(box[0], boxes[:, 0])
        inter_y1 = np.maximum(box[1], boxes[:, 1])
        inter_x2 = np.minimum(box[2], boxes[:, 2])
        inter_y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.clip(inter_x2 - inter_x1, 0, None)
        inter_h = np.clip(inter_y2 - inter_y1, 0, None)
        inter_area = inter_w * inter_h

        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter_area

        return inter_area / np.clip(union, 1e-6, None)


# ============================================================================
# SECTION 3: MOTION GATE
# ============================================================================

class MotionGate:
    """
    Temporal motion gate using frame differencing or MOG2 background subtraction.

    On CPU-only edge hardware, frame differencing costs ~0.3 ms per frame
    at 1920×1080 — negligible compared to detector inference (~50-100 ms).

    The gate decides whether to run the expensive detector or skip and
    rely on Kalman-predicted track positions.
    """

    def __init__(
        self,
        method: str = 'diff',  # 'diff' or 'mog2'
        threshold: int = 25,
        min_area: int = 100,
        dilation_kernel: Tuple[int, int] = (15, 15),
        motion_ratio_threshold: float = 0.003,
        mog2_history: int = 500,
        mog2_var_threshold: int = 16,
        resize_width: int = 320,  # Downsample for speed
    ):
        """
        Args:
            method: 'diff' for frame differencing, 'mog2' for MOG2.
            threshold: Pixel difference threshold (0-255).
            min_area: Minimum contour area to count as motion.
            dilation_kernel: Morphological dilation kernel size.
            motion_ratio_threshold: Min fraction of changed pixels to trigger.
            resize_width: Downsample width for motion computation (speed).
        """
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.kernel = np.ones(dilation_kernel, np.uint8)
        self.motion_ratio_threshold = motion_ratio_threshold
        self.resize_width = resize_width

        if method == 'mog2':
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history,
                varThreshold=mog2_var_threshold,
                detectShadows=False,
            )
        else:
            self.bg_sub = None

        self.prev_gray = None

    def reset(self):
        """Reset state for a new video."""
        self.prev_gray = None
        if self.bg_sub is not None:
            # Re-create MOG2 to clear learned background
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False,
            )

    def has_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if there is significant motion between two frames.

        Args:
            prev_frame, curr_frame: BGR frames (full resolution).

        Returns:
            (has_motion: bool, motion_ratio: float)
        """
        h, w = curr_frame.shape[:2]

        # Downsample for speed
        if self.resize_width < w:
            scale = self.resize_width / w
            small_prev = cv2.resize(prev_frame, (self.resize_width, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            small_curr = cv2.resize(curr_frame, (self.resize_width, int(h * scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            small_prev = prev_frame
            small_curr = curr_frame

        sh, sw = small_curr.shape[:2]

        # Convert to grayscale
        if self.method == 'mog2':
            mask = self.bg_sub.apply(small_curr)
        else:
            prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Clean up mask
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                significant = True
                break

        motion_ratio = cv2.countNonZero(mask) / (sh * sw)

        if not significant and motion_ratio < self.motion_ratio_threshold:
            return False, motion_ratio

        return True, motion_ratio


# ============================================================================
# SECTION 4: MOTION-GATED INFERENCE PIPELINE
# ============================================================================

class MotionGatedInference:
    """
    Motion-gated detection pipeline with SORT track propagation.

    Architecture:
        ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
        │  Motion Gate │────▶│  ONNX Detect │────▶│  SORT Update │
        │  (~0.3 ms)   │     │  (~50-100ms) │     │  (~0.1 ms)   │
        └─────────────┘     └──────────────┘     └──────────────┘
               │                                       ▲
               ▼ No motion                              │
        ┌─────────────┐                                 │
        │  Kalman     │─────────────────────────────────┘
        │  Predict    │   (propagate tracks without inference)
        │  (~0.01 ms) │
        └─────────────┘

    Safety net: full-frame detection every N frames to catch slow-moving
    or hovering drones that frame differencing misses.
    """

    def __init__(
        self,
        model_path: str,
        motion_method: str = 'diff',
        motion_threshold: int = 25,
        motion_ratio_threshold: float = 0.003,
        fallback_interval: int = 30,
        max_track_age: int = 30,
        min_track_hits: int = 3,
        iou_threshold: float = 0.3,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
    ):
        self.detector = ONNXDetector(
            model_path=model_path,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=nms_iou_threshold,
            num_classes=num_classes,
            class_names=class_names,
        )

        self.gate = MotionGate(
            method=motion_method,
            threshold=motion_threshold,
            motion_ratio_threshold=motion_ratio_threshold,
        )

        self.tracker = SortTracker(
            max_age=max_track_age,
            min_hits=min_track_hits,
            iou_threshold=iou_threshold,
        )

        self.fallback_interval = fallback_interval

    def _reset(self):
        """Reset state for a new video/sequence."""
        self.gate.reset()
        self.tracker.reset()

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Process a video with motion-gated inference + SORT tracking.

        Returns statistics dict with frame counts, timing, and track info.
        """
        self._reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        stats = {
            'total_frames': 0,
            'skipped': 0,
            'detected': 0,
            'fallback': 0,
            'gate_times_ms': [],
            'detect_times_ms': [],
            'track_times_ms': [],
            'total_times_ms': [],
            'total_detections': 0,
            'track_ids': set(),
        }

        prev_frame = None
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Motion-Gated")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.perf_counter()

            # First frame: always detect
            if prev_frame is None:
                t_detect = time.perf_counter()
                dets = self.detector.detect(frame)
                stats['detect_times_ms'].append((time.perf_counter() - t_detect) * 1000)

                tracks = self.tracker.update(dets)
                mode = 'detected'
                stats['detected'] += 1
            else:
                # Fallback: full detection every N frames
                if frame_idx % self.fallback_interval == 0:
                    t_detect = time.perf_counter()
                    dets = self.detector.detect(frame)
                    stats['detect_times_ms'].append((time.perf_counter() - t_detect) * 1000)

                    tracks = self.tracker.update(dets)
                    mode = 'fallback'
                    stats['fallback'] += 1
                else:
                    # Stage 1: Motion gate
                    t_gate = time.perf_counter()
                    has_motion, motion_ratio = self.gate.has_motion(prev_frame, frame)
                    stats['gate_times_ms'].append((time.perf_counter() - t_gate) * 1000)

                    if has_motion:
                        # Stage 2: Run detector
                        t_detect = time.perf_counter()
                        dets = self.detector.detect(frame)
                        stats['detect_times_ms'].append((time.perf_counter() - t_detect) * 1000)

                        # Stage 3: Update SORT with detections
                        t_track = time.perf_counter()
                        tracks = self.tracker.update(dets)
                        stats['track_times_ms'].append((time.perf_counter() - t_track) * 1000)

                        mode = 'detected'
                        stats['detected'] += 1
                    else:
                        # No motion: Kalman-predict only (skip detector)
                        t_track = time.perf_counter()
                        tracks = self.tracker.predict_only()
                        stats['track_times_ms'].append((time.perf_counter() - t_track) * 1000)

                        mode = 'skipped'
                        stats['skipped'] += 1

            total_time = (time.perf_counter() - t_start) * 1000
            stats['total_times_ms'].append(total_time)
            stats['total_detections'] += len(tracks)
            for t in tracks:
                stats['track_ids'].add(t['id'])

            stats['total_frames'] += 1
            frame_idx += 1

            if writer:
                annotated = self._draw(frame, tracks, mode, frame_idx)
                writer.write(annotated)

            prev_frame = frame.copy()
            pbar.update(1)

        cap.release()
        if writer:
            writer.release()
        pbar.close()

        stats['track_ids'] = len(stats['track_ids'])
        self._compute_stats(stats)
        self._print_stats(stats, "MOTION-GATED + SORT")
        return stats

    def process_full_frame_baseline(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Run full-frame detection on every frame (always-detect baseline).

        This is the comparison baseline: no motion gating, detector runs
        on every single frame. SORT tracking is still applied to maintain
        track IDs for fair MOTA comparison.
        """
        self._reset()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        stats = {
            'total_frames': 0,
            'detect_times_ms': [],
            'track_times_ms': [],
            'total_times_ms': [],
            'total_detections': 0,
            'track_ids': set(),
        }

        pbar = tqdm(total=total_frames, desc="Full-Frame Baseline")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.perf_counter()

            t_detect = time.perf_counter()
            dets = self.detector.detect(frame)
            stats['detect_times_ms'].append((time.perf_counter() - t_detect) * 1000)

            t_track = time.perf_counter()
            tracks = self.tracker.update(dets)
            stats['track_times_ms'].append((time.perf_counter() - t_track) * 1000)

            total_time = (time.perf_counter() - t_start) * 1000
            stats['total_times_ms'].append(total_time)
            stats['total_detections'] += len(tracks)
            for t in tracks:
                stats['track_ids'].add(t['id'])

            stats['total_frames'] += 1

            if writer:
                annotated = self._draw(frame, tracks, 'full', stats['total_frames'])
                writer.write(annotated)

            pbar.update(1)

        cap.release()
        if writer:
            writer.release()
        pbar.close()

        stats['track_ids'] = len(stats['track_ids'])
        self._compute_stats(stats)
        self._print_stats(stats, "FULL-FRAME BASELINE")
        return stats

    def process_image_dir(
        self,
        image_dir: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Process a directory of frame images (sorted by filename).
        Useful when you've already extracted frames from video.
        """
        self._reset()

        files = sorted([f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        if not files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        stats = {
            'total_frames': 0,
            'skipped': 0,
            'detected': 0,
            'fallback': 0,
            'gate_times_ms': [],
            'detect_times_ms': [],
            'track_times_ms': [],
            'total_times_ms': [],
            'total_detections': 0,
            'track_ids': set(),
        }

        prev_frame = None

        for frame_idx, fname in enumerate(tqdm(files, desc="Motion-Gated")):
            frame = cv2.imread(os.path.join(image_dir, fname))
            if frame is None:
                continue

            t_start = time.perf_counter()

            if prev_frame is None:
                dets = self.detector.detect(frame)
                tracks = self.tracker.update(dets)
                mode = 'detected'
                stats['detected'] += 1
            elif frame_idx % self.fallback_interval == 0:
                dets = self.detector.detect(frame)
                tracks = self.tracker.update(dets)
                mode = 'fallback'
                stats['fallback'] += 1
            else:
                t_gate = time.perf_counter()
                has_motion, _ = self.gate.has_motion(prev_frame, frame)
                stats['gate_times_ms'].append((time.perf_counter() - t_gate) * 1000)

                if has_motion:
                    dets = self.detector.detect(frame)
                    tracks = self.tracker.update(dets)
                    mode = 'detected'
                    stats['detected'] += 1
                else:
                    tracks = self.tracker.predict_only()
                    mode = 'skipped'
                    stats['skipped'] += 1

            total_time = (time.perf_counter() - t_start) * 1000
            stats['total_times_ms'].append(total_time)
            stats['total_detections'] += len(tracks)
            for t in tracks:
                stats['track_ids'].add(t['id'])

            stats['total_frames'] += 1
            prev_frame = frame.copy()

        stats['track_ids'] = len(stats['track_ids'])
        self._compute_stats(stats)
        self._print_stats(stats, "MOTION-GATED + SORT (image dir)")
        return stats

    @staticmethod
    def _compute_stats(stats: Dict):
        """Compute aggregate statistics."""
        total_times = np.array(stats.get('total_times_ms', [1]))
        detect_times = np.array(stats.get('detect_times_ms', [0]))
        gate_times = np.array(stats.get('gate_times_ms', [0]))
        track_times = np.array(stats.get('track_times_ms', [0]))

        total = stats.get('total_frames', 1)

        stats['avg_total_time_ms'] = float(total_times.mean())
        stats['avg_fps'] = float(1000.0 / max(total_times.mean(), 1e-6))
        stats['p95_total_time_ms'] = float(np.percentile(total_times, 95))
        stats['p99_total_time_ms'] = float(np.percentile(total_times, 99))

        if len(detect_times) > 0:
            stats['avg_detect_time_ms'] = float(detect_times.mean())
            stats['p95_detect_time_ms'] = float(np.percentile(detect_times, 95))

        if len(gate_times) > 0:
            stats['avg_gate_time_ms'] = float(gate_times.mean())

        if len(track_times) > 0:
            stats['avg_track_time_ms'] = float(track_times.mean())

        stats['skip_ratio'] = stats.get('skipped', 0) / max(total, 1)
        stats['detect_ratio'] = stats.get('detected', 0) / max(total, 1)
        stats['fallback_ratio'] = stats.get('fallback', 0) / max(total, 1)

    @staticmethod
    def _print_stats(stats: Dict, title: str):
        total = stats.get('total_frames', 1)
        print(f"\n{'='*55}")
        print(f" {title}")
        print(f"{'='*55}")
        print(f" Total frames:         {total}")
        if 'skipped' in stats:
            print(f"   Skipped (no motion): {stats['skipped']} ({stats['skipped']/total*100:.1f}%)")
            print(f"   Detected (motion):   {stats['detected']} ({stats['detected']/total*100:.1f}%)")
            print(f"   Fallback:            {stats['fallback']} ({stats['fallback']/total*100:.1f}%)")
        print(f" Avg total time/frame: {stats.get('avg_total_time_ms', 0):.2f} ms")
        print(f" Avg FPS:              {stats.get('avg_fps', 0):.1f}")
        print(f" P95 latency:          {stats.get('p95_total_time_ms', 0):.2f} ms")
        if 'avg_detect_time_ms' in stats:
            print(f" Avg detect time:      {stats['avg_detect_time_ms']:.2f} ms")
        if 'avg_gate_time_ms' in stats:
            print(f" Avg gate time:        {stats['avg_gate_time_ms']:.2f} ms")
        if 'avg_track_time_ms' in stats:
            print(f" Avg track time:       {stats['avg_track_time_ms']:.3f} ms")
        print(f" Unique tracks:        {stats.get('track_ids', 0)}")
        print(f" Total detections:     {stats.get('total_detections', 0)}")

    @staticmethod
    def _draw(
        frame: np.ndarray,
        tracks: List[Dict],
        mode: str,
        frame_idx: int,
    ) -> np.ndarray:
        """Draw tracking results on frame."""
        annotated = frame.copy()

        colors = {
            'skipped': (128, 128, 128),   # Gray for predicted-only
            'detected': (0, 255, 0),       # Green for detected
            'fallback': (0, 165, 255),     # Orange for fallback
            'full': (0, 255, 255),         # Yellow for full-frame
        }
        color = colors.get(mode, (255, 255, 255))

        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            cls = t.get('class', 0)
            cls_name = f"cls{cls}" if cls is not None else "?"
            label = f"ID:{t['id']} {cls_name} {t.get('confidence', 0):.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Mark predicted (unobserved) tracks differently
            tsu = t.get('time_since_update', 0)
            if tsu > 0:
                cv2.putText(annotated, f"[P{tsu}]", (x2 - 30, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Mode indicator
        mode_text = f"Frame {frame_idx} | Mode: {mode.upper()}"
        cv2.putText(annotated, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated


# ============================================================================
# SECTION 5: BENCHMARK COMPARISON
# ============================================================================

def run_benchmark(
    model_path: str,
    video_path: str,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict:
    """
    Run full benchmark: motion-gated+SORT vs full-frame baseline.

    Produces the comparison table for the paper:
    - Avg FPS, latency, P95
    - Energy proxy (Raspberry Pi 5: ~5W)
    - Skip ratio (frames where detector was skipped)
    - Speedup factor
    """
    pipeline = MotionGatedInference(model_path=model_path, **kwargs)

    print("\n" + "=" * 60)
    print(" BENCHMARK: Motion-Gated+SORT vs Full-Frame Baseline")
    print("=" * 60)

    # Gated run
    gated_out = None
    if output_dir:
        gated_out = os.path.join(output_dir, "gated_output.mp4")

    print("\n--- Running Motion-Gated + SORT ---")
    gated_stats = pipeline.process_video(video_path, output_path=gated_out)

    # Full-frame baseline
    full_out = None
    if output_dir:
        full_out = os.path.join(output_dir, "baseline_output.mp4")

    print("\n--- Running Full-Frame Baseline ---")
    full_stats = pipeline.process_full_frame_baseline(video_path, output_path=full_out)

    # Energy proxy (Raspberry Pi 5 idle ~3W, peak ~5W)
    power_w = 5.0

    gated_fps = gated_stats['avg_fps']
    full_fps = full_stats['avg_fps']

    result = {
        'model_path': model_path,
        'video_path': video_path,
        'full_frame': {
            'avg_fps': full_fps,
            'avg_latency_ms': full_stats['avg_total_time_ms'],
            'p95_latency_ms': full_stats['p95_total_time_ms'],
            'avg_detect_time_ms': full_stats.get('avg_detect_time_ms', 0),
            'total_detections': full_stats['total_detections'],
            'unique_tracks': full_stats['track_ids'],
            'energy_per_frame_mj': full_stats['avg_total_time_ms'] / 1000 * power_w * 1000,
        },
        'motion_gated': {
            'avg_fps': gated_fps,
            'avg_latency_ms': gated_stats['avg_total_time_ms'],
            'p95_latency_ms': gated_stats['p95_total_time_ms'],
            'avg_detect_time_ms': gated_stats.get('avg_detect_time_ms', 0),
            'avg_gate_time_ms': gated_stats.get('avg_gate_time_ms', 0),
            'avg_track_time_ms': gated_stats.get('avg_track_time_ms', 0),
            'total_detections': gated_stats['total_detections'],
            'unique_tracks': gated_stats['track_ids'],
            'frames_skipped': gated_stats['skipped'],
            'frames_detected': gated_stats['detected'],
            'frames_fallback': gated_stats['fallback'],
            'skip_ratio': gated_stats['skip_ratio'],
            'energy_per_frame_mj': gated_stats['avg_total_time_ms'] / 1000 * power_w * 1000,
        },
        'speedup': gated_fps / max(full_fps, 1e-6),
        'energy_reduction': 1.0 - (gated_stats['avg_total_time_ms'] /
                                   max(full_stats['avg_total_time_ms'], 1e-6)),
    }

    # Print comparison table
    print(f"\n{'='*60}")
    print(f" BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Full-Frame':>12} {'Gated+SORT':>12}")
    print(f"{'-'*60}")
    print(f"{'Avg FPS':<35} {full_fps:>12.1f} {gated_fps:>12.1f}")
    print(f"{'Avg latency (ms)':<35} {full_stats['avg_total_time_ms']:>12.2f} "
          f"{gated_stats['avg_total_time_ms']:>12.2f}")
    print(f"{'P95 latency (ms)':<35} {full_stats['p95_total_time_ms']:>12.2f} "
          f"{gated_stats['p95_total_time_ms']:>12.2f}")
    print(f"{'Avg detect time (ms)':<35} "
          f"{full_stats.get('avg_detect_time_ms',0):>12.2f} "
          f"{gated_stats.get('avg_detect_time_ms',0):>12.2f}")
    print(f"{'Avg gate time (ms)':<35} {'N/A':>12} "
          f"{gated_stats.get('avg_gate_time_ms',0):>12.2f}")
    print(f"{'Avg track time (ms)':<35} {'N/A':>12} "
          f"{gated_stats.get('avg_track_time_ms',0):>12.3f}")
    print(f"{'Energy/frame (mJ)':<35} "
          f"{full_stats['avg_total_time_ms']/1000*power_w*1000:>12.1f} "
          f"{gated_stats['avg_total_time_ms']/1000*power_w*1000:>12.1f}")
    print(f"{'Skip ratio':<35} {'0.0%':>12} "
          f"{gated_stats['skip_ratio']*100:>11.1f}%")
    print(f"{'Unique tracks':<35} {full_stats['track_ids']:>12} "
          f"{gated_stats['track_ids']:>12}")
    print(f"{'Total detections':<35} {full_stats['total_detections']:>12} "
          f"{gated_stats['total_detections']:>12}")
    print(f"{'Speedup':<35} {'1.00x':>12} "
          f"{result['speedup']:>11.2f}x")
    print(f"{'Energy reduction':<35} {'0.0%':>12} "
          f"{result['energy_reduction']*100:>11.1f}%")
    print(f"{'='*60}")

    return result


# ============================================================================
# SECTION 6: OPTIONAL MOTA/IDF1 EVALUATION
# ============================================================================

def evaluate_tracking(
    model_path: str,
    video_path: str,
    coco_annotations: str,
    image_dir: str,
    **kwargs,
) -> Dict:
    """
    Evaluate tracking accuracy with MOTA, IDF1, IDSw metrics.

    Uses COCO ground-truth annotations (as provided by the benchmark repo)
    to compute standard MOT metrics.

    Args:
        model_path: Path to ONNX model.
        video_path: Path to video (or None to use image_dir).
        coco_annotations: Path to COCO-format JSON with ground-truth.
        image_dir: Directory with frame images (for frame-level evaluation).
    """
    pipeline = MotionGatedInference(model_path=model_path, **kwargs)

    # Load ground truth
    with open(coco_annotations) as f:
        gt_data = json.load(f)

    # Build per-frame ground truth: {frame_id: [{bbox, id}, ...]}
    gt_by_frame = defaultdict(list)
    img_id_to_frame = {}

    for img in gt_data['images']:
        # Extract frame number from filename (e.g., "video_0001.jpg" → 1)
        fname = img['file_name']
        parts = fname.rsplit('_', 1)
        if len(parts) == 2:
            try:
                frame_num = int(parts[1].split('.')[0])
            except ValueError:
                frame_num = img['id']
        else:
            frame_num = img['id']
        img_id_to_frame[img['id']] = frame_num

    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_frame:
            frame_num = img_id_to_frame[img_id]
            x, y, w, h = ann['bbox']
            gt_by_frame[frame_num].append({
                'bbox': [x, y, x + w, y + h],
                'id': ann.get('instance_id', ann['id']),
            })

    # Run gated inference and collect tracking results
    pipeline._reset()

    if video_path:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Collect predictions per frame
        pred_by_frame = {}
        prev_frame = None
        frame_idx = 0

        for _ in tqdm(range(total), desc="Eval (video)"):
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is None or frame_idx % pipeline.fallback_interval == 0:
                dets = pipeline.detector.detect(frame)
                tracks = pipeline.tracker.update(dets)
            else:
                has_motion, _ = pipeline.gate.has_motion(prev_frame, frame)
                if has_motion:
                    dets = pipeline.detector.detect(frame)
                    tracks = pipeline.tracker.update(dets)
                else:
                    tracks = pipeline.tracker.predict_only()

            pred_by_frame[frame_idx] = tracks
            prev_frame = frame.copy()
            frame_idx += 1

        cap.release()
    else:
        # Use image directory
        files = sorted(os.listdir(image_dir))
        pred_by_frame = {}
        prev_frame = None

        for frame_idx, fname in enumerate(tqdm(files, desc="Eval (images)")):
            frame = cv2.imread(os.path.join(image_dir, fname))
            if frame is None:
                continue

            if prev_frame is None or frame_idx % pipeline.fallback_interval == 0:
                dets = pipeline.detector.detect(frame)
                tracks = pipeline.tracker.update(dets)
            else:
                has_motion, _ = pipeline.gate.has_motion(prev_frame, frame)
                if has_motion:
                    dets = pipeline.detector.detect(frame)
                    tracks = pipeline.tracker.update(dets)
                else:
                    tracks = pipeline.tracker.predict_only()

            pred_by_frame[frame_idx] = tracks
            prev_frame = frame.copy()

    # Compute MOTA, IDF1, IDSw
    metrics = compute_mota_idf1(gt_by_frame, pred_by_frame)

    print(f"\n{'='*50}")
    print(f" TRACKING EVALUATION (vs Ground Truth)")
    print(f"{'='*50}")
    print(f" MOTA:   {metrics['mota']:.4f}")
    print(f" IDF1:   {metrics['idf1']:.4f}")
    print(f" TP:     {metrics['tp']}")
    print(f" FP:     {metrics['fp']}")
    print(f" FN:     {metrics['fn']}")
    print(f" IDSw:   {metrics['idsw']}")
    print(f" GT:     {metrics['gt_count']}")
    print(f" Pred:   {metrics['pred_count']}")
    print(f"{'='*50}")

    return metrics


def compute_mota_idf1(
    gt_by_frame: Dict[int, List[Dict]],
    pred_by_frame: Dict[int, List[Dict]],
    iou_match_threshold: float = 0.5,
) -> Dict:
    """
    Compute MOTA and IDF1 from frame-level ground truth and predictions.

    MOTA = 1 - (FN + FP + IDSw) / GT
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Uses a simplified global-hungarian ID matching approach.
    """
    # Collect all unique GT and pred track IDs
    all_gt_ids = set()
    all_pred_ids = set()
    for frame_gt in gt_by_frame.values():
        for gt in frame_gt:
            all_gt_ids.add(gt['id'])
    for frame_pred in pred_by_frame.values():
        for pred in frame_pred:
            all_pred_ids.add(pred['id'])

    # Build co-occurrence matrix: how many frames each (gt_id, pred_id) pair overlaps
    cooccur = defaultdict(int)
    tp_frames = 0
    fp_frames = 0
    fn_frames = 0
    idsw = 0

    prev_matches = {}  # gt_id → pred_id from previous frame

    for frame_idx in sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys()))):
        gt_frame = gt_by_frame.get(frame_idx, [])
        pred_frame = pred_by_frame.get(frame_idx, [])

        if len(gt_frame) == 0 and len(pred_frame) == 0:
            continue

        # Build IoU matrix
        if len(gt_frame) > 0 and len(pred_frame) > 0:
            gt_boxes = np.array([g['bbox'] for g in gt_frame])
            pred_boxes = np.array([p['bbox'] for p in pred_frame])

            iou_matrix = np.zeros((len(gt_frame), len(pred_frame)))
            for i, gb in enumerate(gt_boxes):
                for j, pb in enumerate(pred_boxes):
                    iou_matrix[i, j] = _compute_iou(gb, pb)

            # Hungarian assignment
            gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

            matched_gt = set()
            matched_pred = set()
            for gi, pi in zip(gt_indices, pred_indices):
                if iou_matrix[gi, pi] >= iou_match_threshold:
                    tp_frames += 1
                    gt_id = gt_frame[gi]['id']
                    pred_id = pred_frame[pi]['id']
                    cooccur[(gt_id, pred_id)] += 1
                    matched_gt.add(gi)
                    matched_pred.add(pi)

                    # Check ID switch
                    if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                        idsw += 1
                    prev_matches[gt_id] = pred_id

            fp_frames += len(pred_frame) - len(matched_pred)
            fn_frames += len(gt_frame) - len(matched_gt)
        elif len(gt_frame) > 0:
            fn_frames += len(gt_frame)
        elif len(pred_frame) > 0:
            fp_frames += len(pred_frame)

    gt_total = sum(len(v) for v in gt_by_frame.values())
    pred_total = sum(len(v) for v in pred_by_frame.values())

    # Global ID matching for IDF1
    gt_ids_list = sorted(all_gt_ids)
    pred_ids_list = sorted(all_pred_ids)

    if len(gt_ids_list) > 0 and len(pred_ids_list) > 0:
        cost_matrix = np.zeros((len(gt_ids_list), len(pred_ids_list)))
        gt_idx_map = {gid: i for i, gid in enumerate(gt_ids_list)}
        pred_idx_map = {pid: i for i, pid in enumerate(pred_ids_list)}

        for (gt_id, pred_id), count in cooccur.items():
            i = gt_idx_map[gt_id]
            j = pred_idx_map[pred_id]
            cost_matrix[i, j] = count

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        idtp = 0
        matched_gt_ids = set()
        matched_pred_ids = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > 0:
                idtp += cost_matrix[r, c]
                matched_gt_ids.add(gt_ids_list[r])
                matched_pred_ids.add(pred_ids_list[c])

        idfp = pred_total - idtp
        idfn = gt_total - idtp
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    else:
        idf1 = 0.0
        idtp = 0

    mota = 1.0 - (fn_frames + fp_frames + idsw) / max(gt_total, 1)

    return {
        'mota': float(mota),
        'idf1': float(idf1),
        'tp': tp_frames,
        'fp': fp_frames,
        'fn': fn_frames,
        'idsw': idsw,
        'gt_count': gt_total,
        'pred_count': pred_total,
        'idtp': int(idtp),
    }


def _compute_iou(box_a, box_b) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / max(union, 1e-6)


# ============================================================================
# SECTION 7: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Motion-Gated Inference with SORT Tracking for UAV Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Benchmark on a video
  python motion_gated_inference.py \\
      --model model_mixed_precision_quantized.onnx \\
      --video drone_surveillance.mp4 \\
      --benchmark

  # Process image directory with annotated output
  python motion_gated_inference.py \\
      --model model_quant.onnx \\
      --image-dir /path/to/frames \\
      --output annotated_output.mp4

  # Evaluate with MOTA/IDF1
  python motion_gated_inference.py \\
      --model model_quant.onnx \\
      --video drone.mp4 \\
      --coco-annotations val.json \\
      --eval
        """,
    )

    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--video', default=None, help='Path to input video')
    parser.add_argument('--image-dir', default=None, help='Directory of frame images')
    parser.add_argument('--output', default=None, help='Output video path')
    parser.add_argument('--output-dir', default=None, help='Directory for output videos')

    parser.add_argument('--benchmark', action='store_true',
                        help='Run gated vs full-frame benchmark comparison')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate MOTA/IDF1 with ground truth')

    parser.add_argument('--coco-annotations', default=None,
                        help='COCO JSON with ground-truth annotations (for --eval)')

    # Detector parameters
    parser.add_argument('--imgsz', type=int, default=640, help='Model input size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms-iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes in model')
    parser.add_argument('--class-names', nargs='+', default=None,
                        help='Class name strings (e.g., drone bird)')

    # Motion gate parameters
    parser.add_argument('--motion-method', choices=['diff', 'mog2'],
                        default='diff', help='Motion detection method')
    parser.add_argument('--motion-threshold', type=int, default=25,
                        help='Pixel difference threshold')
    parser.add_argument('--motion-ratio', type=float, default=0.003,
                        help='Min motion ratio to trigger detection')
    parser.add_argument('--fallback-interval', type=int, default=30,
                        help='Full-frame detection every N frames (safety net)')

    # SORT parameters
    parser.add_argument('--max-age', type=int, default=30,
                        help='Max frames track survives without detection')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='Min consecutive hits to confirm a track')
    parser.add_argument('--match-iou', type=float, default=0.3,
                        help='Min IoU for detection-track matching')

    # Output
    parser.add_argument('--save-json', default=None,
                        help='Save benchmark results to JSON file')

    args = parser.parse_args()

    # Validate inputs
    if not args.video and not args.image_dir:
        parser.error("Either --video or --image-dir is required")

    if args.eval and not args.coco_annotations:
        parser.error("--eval requires --coco-annotations")

    kwargs = dict(
        motion_method=args.motion_method,
        motion_threshold=args.motion_threshold,
        motion_ratio_threshold=args.motion_ratio,
        fallback_interval=args.fallback_interval,
        max_track_age=args.max_age,
        min_track_hits=args.min_hits,
        iou_threshold=args.match_iou,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        nms_iou_threshold=args.nms_iou,
        num_classes=args.num_classes,
        class_names=args.class_names,
    )

    if args.benchmark:
        if not args.video:
            parser.error("--benchmark requires --video")
        result = run_benchmark(
            model_path=args.model,
            video_path=args.video,
            output_dir=args.output_dir,
            **kwargs,
        )
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.save_json}")

    elif args.eval:
        result = evaluate_tracking(
            model_path=args.model,
            video_path=args.video,
            coco_annotations=args.coco_annotations,
            image_dir=args.image_dir or '',
            **kwargs,
        )
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.save_json}")

    else:
        # Simple gated inference
        pipeline = MotionGatedInference(model_path=args.model, **kwargs)

        if args.video:
            stats = pipeline.process_video(args.video, output_path=args.output)
        else:
            stats = pipeline.process_image_dir(args.image_dir, output_path=args.output)

        if args.save_json:
            # Convert sets to lists for JSON
            stats_json = {k: (list(v) if isinstance(v, set) else v)
                         for k, v in stats.items()}
            with open(args.save_json, 'w') as f:
                json.dump(stats_json, f, indent=2, default=str)
            print(f"\nResults saved to {args.save_json}")


if __name__ == '__main__':
    main()
