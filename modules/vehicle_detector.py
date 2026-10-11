"""
Vehicle Detector Module
Provides multi-backend vehicle detection, tracking, and traffic metric estimation
for traffic surveillance video streams.

Key Capabilities:
1. Real Pretrained DNN Object Detection: YOLOv8 Nano ONNX running natively via OpenCV DNN (cv2.dnn).
   Detects supported road vehicles: Cars, Motorcycles, Buses, and Trucks.
2. Classical Computer Vision: OpenCV MOG2 background subtractor and contour classifier.
3. Procedural Synthetic Engine: Ground-truth vehicle simulation for testing and zero-dependency demo.
4. Multi-Object Tracking: IoU + Centroid distance Hungarian tracker maintaining persistent track IDs.
5. Unique Vehicle Counting: Distinguishes unique tracked vehicles from instantaneous frame density
   to prevent repeated counting of stationary or slow-moving vehicles.
6. CPU Efficiency: Configurable confidence threshold, Non-Maximum Suppression (NMS), and inference input size.
7. Robust Error Handling: Explicit exceptions for missing weights, corrupted data, and empty frames.
   Transparently isolates real inference results from synthetic demo modes.

Model Details:
- Model Name: YOLOv8 Nano (yolov8n.onnx)
- Input Resolution: 640x640 pixels (RGB, normalized [0, 1])
- Architecture: Ultralytics YOLOv8 Nano anchor-free convolutional network
- COCO Vehicle Classes: Car (ID 2), Motorcycle (ID 3), Bus (ID 5), Truck (ID 7)
- Execution Engine: OpenCV DNN CPU runtime (cv2.dnn.readNetFromONNX)

Limitations:
- Severe vehicle occlusions (e.g. motorcycles completely obscured by articulated buses).
- Nighttime glare / headlight bloom causing expanded bounding boxes.
- Extreme oblique or fisheye surveillance camera angles without perspective correction.
- Boundary truncation when vehicles enter or exit frame edges.
"""

from dataclasses import dataclass, field
import os
import time
import urllib.request
import urllib.error
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class ModelNotLoadedError(RuntimeError):
    """Raised when real neural network inference is requested but model weights are not loaded."""
    pass


@dataclass
class Detection:
    """Represents a detected vehicle in a video frame."""
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in pixel coordinates
    class_name: str                    # 'car', 'bus', 'truck', 'motorcycle'
    confidence: float                  # Detection confidence score (0.0 to 1.0)
    center: Tuple[int, int]            # Centroid (cx, cy)
    track_id: Optional[int] = None     # Persistent tracking identifier across frames
    lane_id: str = "Unknown"           # Assigned traffic lane approach ('North', 'South', 'East', 'West')
    pcu_weight: float = 1.0            # Passenger Car Unit equivalent


@dataclass
class VehicleTrack:
    """Represents a persistently tracked vehicle across consecutive video frames."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    hits: int = 1
    consecutive_misses: int = 0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    lane_id: str = "Unknown"
    pcu_weight: float = 1.0


class VehicleTracker:
    """
    Multi-Object Tracker for traffic video streams.
    Uses Hungarian algorithm on IoU overlap and Euclidean centroid distance to maintain
    persistent vehicle track IDs across consecutive frames.
    
    Prevents counting the same vehicle repeatedly as a new vehicle and provides
    clear distinction between unique tracks, vehicles visible in a frame, and estimated traffic flow.
    """
    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 1,
        iou_threshold: float = 0.25,
        max_distance: float = 85.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

        self.next_track_id: int = 1
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.unique_track_ids: Set[int] = set()
        self.unique_counts_by_class: Dict[str, Set[int]] = {
            "car": set(),
            "bus": set(),
            "truck": set(),
            "motorcycle": set(),
        }
        self.first_frame_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None

    def reset(self) -> None:
        """Resets tracker state, clearing all active tracks and cumulative counts."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.unique_track_ids.clear()
        for k in self.unique_counts_by_class:
            self.unique_counts_by_class[k].clear()
        self.first_frame_time = None
        self.last_frame_time = None

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Computes 2D Intersection-over-Union (IoU) between two bounding boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def _compute_distance(pt_a: Tuple[int, int], pt_b: Tuple[int, int]) -> float:
        """Computes Euclidean distance between two centroid coordinates."""
        dx = float(pt_a[0] - pt_b[0])
        dy = float(pt_a[1] - pt_b[1])
        return (dx * dx + dy * dy) ** 0.5

    def update(self, detections: List[Detection], timestamp: float = 0.0) -> List[Detection]:
        """
        Associates new frame detections with existing tracks and assigns persistent track IDs.
        Returns the detections updated with persistent track_id.
        """
        if self.first_frame_time is None:
            self.first_frame_time = timestamp
        self.last_frame_time = timestamp

        active_track_list = list(self.active_tracks.values())
        num_tracks = len(active_track_list)
        num_dets = len(detections)

        # Handle edge cases: No existing tracks
        if num_tracks == 0:
            for det in detections:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                new_track = VehicleTrack(
                    track_id=tid,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    hits=1,
                    consecutive_misses=0,
                    trajectory=[det.center],
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                    lane_id=det.lane_id,
                    pcu_weight=det.pcu_weight,
                )
                self.active_tracks[tid] = new_track
                self.unique_track_ids.add(tid)
                c_key = det.class_name.lower()
                if c_key in self.unique_counts_by_class:
                    self.unique_counts_by_class[c_key].add(tid)
            return detections

        # Handle edge cases: No detections in this frame
        if num_dets == 0:
            to_remove = []
            for tid, track in self.active_tracks.items():
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.active_tracks[tid]
            return detections

        # Compute cost matrix based on IoU and centroid distance
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, track in enumerate(active_track_list):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det.bbox)
                dist = self._compute_distance(track.center, det.center)

                # Prioritize IoU overlap; fall back to distance when moving fast
                if iou > 0.0:
                    cost = 1.0 - iou
                elif dist < self.max_distance:
                    cost = 1.0 + (dist / self.max_distance)
                else:
                    cost = 10.0  # Large penalty for disjoint and distant boxes

                # Class consistency bonus/penalty
                if track.class_name != det.class_name:
                    cost += 0.5

                cost_matrix[i, j] = cost

        # Solve assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            # Match condition: acceptable IoU or close centroid distance
            iou = self._compute_iou(active_track_list[r].bbox, detections[c].bbox)
            dist = self._compute_distance(active_track_list[r].center, detections[c].center)

            if iou >= self.iou_threshold or dist <= self.max_distance:
                track = active_track_list[r]
                det = detections[c]

                det.track_id = track.track_id
                track.bbox = det.bbox
                track.center = det.center
                track.confidence = det.confidence
                track.lane_id = det.lane_id
                track.pcu_weight = det.pcu_weight
                track.hits += 1
                track.consecutive_misses = 0
                track.last_seen_time = timestamp
                track.trajectory.append(det.center)
                if len(track.trajectory) > 30:
                    track.trajectory.pop(0)

                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched detections become new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                new_track = VehicleTrack(
                    track_id=tid,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    hits=1,
                    consecutive_misses=0,
                    trajectory=[det.center],
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                    lane_id=det.lane_id,
                    pcu_weight=det.pcu_weight,
                )
                self.active_tracks[tid] = new_track
                self.unique_track_ids.add(tid)
                c_key = det.class_name.lower()
                if c_key in self.unique_counts_by_class:
                    self.unique_counts_by_class[c_key].add(tid)

        # Unmatched tracks age and expire
        to_remove = []
        for i, track in enumerate(active_track_list):
            if i not in matched_tracks:
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_age:
                    to_remove.append(track.track_id)

        for tid in to_remove:
            if tid in self.active_tracks:
                del self.active_tracks[tid]

        return detections

    def get_unique_vehicle_count(self) -> int:
        """Returns total cumulative unique vehicles tracked across the video."""
        return len(self.unique_track_ids)

    def get_unique_counts_by_class(self) -> Dict[str, int]:
        """Returns cumulative unique vehicle counts broken down by class."""
        return {k: len(v) for k, v in self.unique_counts_by_class.items()}

    def get_active_vehicle_count(self) -> int:
        """Returns number of vehicles actively tracked in the current frame."""
        return len(self.active_tracks)

    def compute_traffic_flow_rate(self, elapsed_seconds: Optional[float] = None) -> float:
        """
        Estimates traffic flow rate in Vehicles Per Hour (veh/h).
        q = (N_unique / delta_t) * 3600
        """
        if elapsed_seconds is None:
            if self.first_frame_time is not None and self.last_frame_time is not None:
                elapsed_seconds = max(0.1, self.last_frame_time - self.first_frame_time)
            else:
                elapsed_seconds = 1.0

        if elapsed_seconds <= 0.05:
            return 0.0

        unique_count = self.get_unique_vehicle_count()
        rate_vph = (unique_count / elapsed_seconds) * 3600.0
        return round(rate_vph, 1)

    @staticmethod
    def explain_traffic_metrics() -> Dict[str, str]:
        """
        Explains the fundamental theoretical distinction between unique tracks,
        instantaneous visible vehicles, and estimated traffic flow.
        """
        return {
            "unique_tracks": (
                "Cumulative count of distinct, persistent physical vehicle entities observed "
                "over the entire surveillance window (|T_unique|). Each vehicle is assigned a "
                "persistent track ID upon entering the frame and is counted exactly once. "
                "This prevents vehicles waiting at red lights or queueing slowly from inflating "
                "the total vehicle count."
            ),
            "vehicles_visible_in_frame": (
                "Instantaneous spatial occupancy count (density) of active vehicles currently "
                "visible inside the camera's field of view at frame timestamp t. Reflects acute "
                "roadway demand and queue length, directly driving real-time green-time split allocation."
            ),
            "estimated_traffic_flow": (
                "Dynamic throughput rate of vehicles per unit time, expressed in Vehicles Per Hour (veh/h) "
                "or Passenger Car Units per hour (PCU/h). Calculated as q = (ΔN_unique / Δt) * 3600 over "
                "the elapsed observation window. Used in civil engineering for Highway Capacity Manual (HCM) "
                "Level of Service (LOS) and Bureau of Public Roads (BPR) volume-to-capacity (V/C) impedance."
            ),
        }


class BaseDetector:
    """Abstract interface for vehicle detectors."""
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class OpenCVContourDetector(BaseDetector):
    """
    Classical Computer Vision vehicle detector using Background Subtraction (MOG2),
    Morphological operations, and geometric contour classification.
    Ideal for fixed surveillance cameras without requiring neural network downloads.
    """
    def __init__(self, min_area: int = 1200, max_area: int = 40000):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )
        self.min_area = min_area
        self.max_area = max_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return []

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply background subtraction
        fg_mask = self.subtractor.apply(blurred)

        # Remove shadows (gray pixels 127 in MOG2) and apply morphology
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, self.kernel, iterations=2)
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 1.0

                # Classification heuristic based on area and aspect ratio
                if area > 10000 or (w > 120 and h > 80):
                    class_name = "bus" if aspect_ratio > 1.2 else "truck"
                    pcu = 3.0
                    conf = min(0.92, 0.75 + (area / self.max_area) * 0.15)
                elif area < 3000 and (aspect_ratio < 0.9 or h > 1.4 * w):
                    class_name = "motorcycle"
                    pcu = 0.5
                    conf = 0.82
                else:
                    class_name = "car"
                    pcu = 1.0
                    conf = min(0.95, 0.80 + (area / 10000) * 0.15)

                cx = int(x + w / 2)
                cy = int(y + h / 2)
                detections.append(
                    Detection(
                        bbox=(x, y, x + w, y + h),
                        class_name=class_name,
                        confidence=round(conf, 2),
                        center=(cx, cy),
                        pcu_weight=pcu,
                    )
                )

        return detections


class PretrainedDNNVehicleDetector(BaseDetector):
    """
    OpenCV DNN Pretrained Object Detector using YOLOv8 Nano ONNX.
    Detects supported road vehicles (cars, motorcycles, buses, trucks) using
    real convolutional neural network inference.
    
    Adheres strictly to academic integrity:
    If the model weights are missing or unreadable, does NOT return fake/mock detections.
    Instead, exposes an explicit error state and raises ModelNotLoadedError on inference.
    """
    # COCO 80-Class indices for supported road vehicles
    COCO_VEHICLE_CLASSES = {
        2: ("car", 1.0),
        3: ("motorcycle", 0.5),
        5: ("bus", 3.0),
        7: ("truck", 3.0),
    }

    DEFAULT_MODEL_FILENAME = "yolov8n.onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net: Optional[cv2.dnn.Net] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        # Determine model path
        self.model_path = self._resolve_model_path(model_path)
        self._load_network()

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Finds model weights file across common workspace locations."""
        if model_path is not None:
            return model_path

        candidates = [
            os.path.join("models", self.DEFAULT_MODEL_FILENAME),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", self.DEFAULT_MODEL_FILENAME)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), self.DEFAULT_MODEL_FILENAME)),
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                return p

        # Default path
        return os.path.join("models", self.DEFAULT_MODEL_FILENAME)

    def _load_network(self) -> None:
        """Loads ONNX network into OpenCV DNN module."""
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            self.load_error = (
                f"Model weights file not found at '{self.model_path}'. "
                f"Please ensure '{self.DEFAULT_MODEL_FILENAME}' is located in the 'models/' folder."
            )
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load ONNX model '{self.model_path}': {str(e)}"

    @classmethod
    def download_weights_if_missing(cls, target_path: Optional[str] = None, timeout: int = 45) -> Tuple[bool, str]:
        """
        Attempts to download the official YOLOv8 Nano ONNX model weights (12.8 MB)
        from Ultralytics GitHub release assets if not already present.

        Network Requirement:
            Requires outbound HTTPS access to github.com and objects.githubusercontent.com.

        Failure Handling:
            Handles HTTP errors, network timeouts, and partial file corruption gracefully.
            Leaves existing files untouched upon failure.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        dest = target_path or os.path.join("models", cls.DEFAULT_MODEL_FILENAME)
        dest_abs = os.path.abspath(dest)

        if os.path.exists(dest_abs) and os.path.getsize(dest_abs) > 10 * 1024 * 1024:
            return True, f"Model weights already present at {dest_abs}."

        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        urls = [
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        ]

        temp_dest = dest_abs + ".download.tmp"
        err_msg = "Unknown error"
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SmartTraffic-AutoSetup/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    with open(temp_dest, "wb") as f_out:
                        while True:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)

                if os.path.exists(temp_dest) and os.path.getsize(temp_dest) > 10 * 1024 * 1024:
                    if os.path.exists(dest_abs):
                        os.remove(dest_abs)
                    os.rename(temp_dest, dest_abs)
                    return True, f"Successfully downloaded YOLOv8 Nano weights to {dest_abs}."
            except urllib.error.HTTPError as e:
                err_msg = f"HTTP Error {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                err_msg = f"Network connection failed: {e.reason}. Check internet/firewall settings."
            except TimeoutError:
                err_msg = f"Download timed out after {timeout}s. Network may be constrained."
            except Exception as e:
                err_msg = f"Download error ({type(e).__name__}): {str(e)}"
            finally:
                if os.path.exists(temp_dest):
                    try:
                        os.remove(temp_dest)
                    except Exception:
                        pass

        return False, (
            f"Failed to download model weights. Detail: {err_msg}. "
            f"Network Requirement: Outbound HTTPS to github.com is required. "
            f"You can manually place 'yolov8n.onnx' in the 'models/' folder or run 'python setup_models.py'."
        )

    def reload(self) -> bool:
        """Attempts to re-resolve and load the neural network weights from disk."""
        self.model_path = self._resolve_model_path(None)
        self._load_network()
        return self.is_loaded

    def set_conf_threshold(self, threshold: float) -> None:
        """Configures confidence threshold (0.0 to 1.0)."""
        self.conf_threshold = max(0.01, min(0.99, threshold))

    def set_input_size(self, size: Tuple[int, int]) -> None:
        """Configures input inference size (width, height)."""
        self.input_size = size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Executes real YOLOv8 object detection on the frame.
        Raises ModelNotLoadedError if model weights are missing or unreadable.
        """
        if not self.is_loaded or self.net is None:
            raise ModelNotLoadedError(
                self.load_error or "Neural network model weights are not loaded. Cannot perform real inference."
            )

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return []

        orig_h, orig_w = frame.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return []

        # YOLOv8 expects 640x640 normalized RGB input
        in_w, in_h = self.input_size
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (in_w, in_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        # Parse YOLOv8 output: shape (1, 84, 8400) -> transpose to (8400, 84)
        predictions = np.squeeze(output).T
        if predictions.ndim != 2 or predictions.shape[1] < 84:
            return []

        # Coordinates [cx, cy, w, h] in inference scale
        boxes_raw = predictions[:, :4]
        # Class probabilities [80 classes]
        class_scores = predictions[:, 4:]

        # Extract best class per candidate box
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Scale factors to map inference coordinates back to original frame dimensions
        scale_x = orig_w / float(in_w)
        scale_y = orig_h / float(in_h)

        candidate_boxes: List[List[int]] = []
        candidate_confs: List[float] = []
        candidate_classes: List[int] = []

        for i in range(len(class_ids)):
            conf = float(confidences[i])
            cid = int(class_ids[i])

            if conf >= self.conf_threshold and cid in self.COCO_VEHICLE_CLASSES:
                cx = boxes_raw[i, 0] * scale_x
                cy = boxes_raw[i, 1] * scale_y
                bw = boxes_raw[i, 2] * scale_x
                bh = boxes_raw[i, 3] * scale_y

                x1 = int(cx - bw / 2.0)
                y1 = int(cy - bh / 2.0)
                w = int(bw)
                h = int(bh)

                candidate_boxes.append([x1, y1, w, h])
                candidate_confs.append(conf)
                candidate_classes.append(cid)

        if not candidate_boxes:
            return []

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            candidate_boxes, candidate_confs, self.conf_threshold, self.nms_threshold
        )

        detections: List[Detection] = []
        if len(indices) == 0:
            return []

        indices_flat = indices.flatten() if hasattr(indices, "flatten") else [indices[0]]

        for idx in indices_flat:
            box = candidate_boxes[idx]
            conf = candidate_confs[idx]
            cid = candidate_classes[idx]

            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(orig_w, box[0] + box[2])
            y2 = min(orig_h, box[1] + box[3])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name, pcu = self.COCO_VEHICLE_CLASSES[cid]
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_name=class_name,
                    confidence=round(conf, 2),
                    center=(cx, cy),
                    pcu_weight=pcu,
                )
            )

        return detections


class ProceduralSyntheticDetector(BaseDetector):
    """
    Extracts vehicles from procedurally synthesized traffic simulation frames.
    Provides ground-truth coordinates and metrics for zero-dependency testing and demonstration.
    """
    def __init__(self, ground_truth_provider=None):
        self.provider = ground_truth_provider

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.provider and hasattr(self.provider, "get_detections"):
            return self.provider.get_detections(frame)
        return []


class VehicleDetector:
    """
    Unified Vehicle Detection and Multi-Object Tracking Pipeline.
    
    Provides:
    - Real YOLOv8 ONNX object detection
    - Classical OpenCV MOG2 background subtractor
    - Procedural synthetic benchmark simulator
    - Multi-Object Tracking with persistent track IDs
    - Lane boundary classification
    - High-contrast HUD annotation rendering
    - Transparent reporting of real inference vs demo mode
    """
    BACKENDS = ["Pretrained DNN / YOLO", "OpenCV MOG2 (Classical CV)", "Procedural Simulation"]

    def __init__(
        self,
        backend: str = "Pretrained DNN / YOLO",
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        model_path: Optional[str] = None,
        enable_tracking: bool = True,
    ):
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.enable_tracking = enable_tracking

        self.contour_detector = OpenCVContourDetector()
        self.dnn_detector = PretrainedDNNVehicleDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
        )
        self.synthetic_detector = ProceduralSyntheticDetector()
        self.tracker = VehicleTracker()

        # If user defaulted to DNN but DNN weights are missing, set backend safely without crashing
        if self.backend == "Pretrained DNN / YOLO" and not self.dnn_detector.is_loaded:
            # We preserve the chosen backend for transparency, but record the load error
            pass

    def set_backend(self, backend: str) -> None:
        """Switches active detector backend."""
        if backend in self.BACKENDS:
            self.backend = backend

    def set_conf_threshold(self, threshold: float) -> None:
        """Updates confidence threshold across detectors."""
        self.conf_threshold = threshold
        self.dnn_detector.set_conf_threshold(threshold)

    def set_input_size(self, size: Tuple[int, int]) -> None:
        """Updates model input inference size."""
        self.input_size = size
        self.dnn_detector.set_input_size(size)

    def reset_tracker(self) -> None:
        """Resets tracking IDs and cumulative metrics."""
        self.tracker.reset()

    def download_model_weights(self, timeout: int = 45) -> Tuple[bool, str]:
        """
        Attempts to download pretrained YOLOv8 Nano weights at runtime if missing.
        Reloads the DNN detector upon successful download.
        """
        success, msg = PretrainedDNNVehicleDetector.download_weights_if_missing(timeout=timeout)
        if success:
            self.reload_dnn()
        return success, msg

    def reload_dnn(self) -> bool:
        """Reloads the pretrained DNN detector from disk."""
        return self.dnn_detector.reload()

    def get_status(self) -> Dict[str, Any]:
        """
        Returns full diagnostic status of the detection and tracking pipeline.
        Enables explicit UI messaging regarding real vs simulated inference.
        """
        is_simulated = self.backend == "Procedural Simulation"
        is_dnn_loaded = self.dnn_detector.is_loaded
        return {
            "backend": self.backend,
            "is_simulated": is_simulated,
            "is_dnn_model_loaded": is_dnn_loaded,
            "dnn_load_error": self.dnn_detector.load_error,
            "model_path": self.dnn_detector.model_path,
            "confidence_threshold": self.conf_threshold,
            "input_size": self.input_size,
            "tracking_enabled": self.enable_tracking,
            "unique_tracked_vehicles": self.tracker.get_unique_vehicle_count(),
            "active_tracked_vehicles": self.tracker.get_active_vehicle_count(),
            "supported_classes": ["car", "bus", "truck", "motorcycle"],
            "pcu_standards": {"car": 1.0, "bus": 3.0, "truck": 3.0, "motorcycle": 0.5},
        }

    def detect(
        self,
        frame: np.ndarray,
        synthetic_stream=None,
        timestamp: float = 0.0,
        apply_tracking: bool = True,
    ) -> List[Detection]:
        """
        Detects vehicles in frame, assigns approach lanes, and optionally updates persistent track IDs.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return []

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []

        detections: List[Detection] = []

        if synthetic_stream is not None:
            detections = synthetic_stream.get_detections(frame)
        elif self.backend == "Procedural Simulation":
            detections = self.synthetic_detector.detect(frame)
        elif self.backend == "Pretrained DNN / YOLO":
            if not self.dnn_detector.is_loaded:
                raise ModelNotLoadedError(
                    f"Real detection requested via 'Pretrained DNN / YOLO', but weights could not be loaded: "
                    f"{self.dnn_detector.load_error}. Please ensure '{self.dnn_detector.model_path}' exists, "
                    f"or select 'Procedural Simulation' demo mode."
                )
            detections = self.dnn_detector.detect(frame)
        else:  # Classical OpenCV MOG2
            detections = self.contour_detector.detect(frame)

        # Filter by confidence threshold
        detections = [d for d in detections if d.confidence >= self.conf_threshold]

        # Assign lane approaches (North, South, East, West)
        for det in detections:
            cx, cy = det.center
            det.lane_id = self._classify_lane(cx, cy, w, h)

        # Apply multi-object tracking to assign persistent track IDs
        if apply_tracking and self.enable_tracking:
            detections = self.tracker.update(detections, timestamp=timestamp)

        return detections

    @staticmethod
    def _classify_lane(cx: int, cy: int, width: int, height: int) -> str:
        """Classifies a vehicle's approach lane based on its spatial quadrant relative to center."""
        mid_x = width // 2
        mid_y = height // 2

        dx = cx - mid_x
        dy = cy - mid_y

        if abs(dy) >= abs(dx):
            return "North" if dy < 0 else "South"
        else:
            return "West" if dx < 0 else "East"

    @staticmethod
    def explain_traffic_metrics() -> Dict[str, str]:
        """Returns comprehensive explanation of unique tracks vs visible count vs flow rate."""
        return VehicleTracker.explain_traffic_metrics()

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Detection],
        draw_lanes: bool = True,
        show_labels: bool = True,
        show_tracks: bool = True,
        show_trails: bool = True,
        tracker: Optional[VehicleTracker] = None,
        header_text: Optional[str] = None,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        tripwire_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        tripwire_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws high-contrast HUD annotations on the frame.
        Uses Cyan / Amber / Coral / Emerald Green palette with bounding boxes,
        persistent track IDs, velocity vectors, ROI regions, and tripwire counting lines.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros((540, 720, 3), dtype=np.uint8)

        out_frame = frame.copy()
        h, w = out_frame.shape[:2]

        # Professional BGR color palette
        class_colors = {
            "car": (238, 211, 34),          # Cyan
            "bus": (11, 158, 245),          # Amber
            "truck": (68, 68, 239),         # Coral Red
            "motorcycle": (129, 185, 16),   # Emerald Green
        }

        # 1. Lane Corridor Guidelines
        if draw_lanes:
            overlay = out_frame.copy()
            mid_x, mid_y = w // 2, h // 2
            box_r = min(w, h) // 7

            # Central intersection conflict box
            cv2.rectangle(
                overlay,
                (mid_x - box_r, mid_y - box_r),
                (mid_x + box_r, mid_y + box_r),
                (238, 211, 34),
                1,
            )
            # Dividing corridors
            cv2.line(overlay, (mid_x, 0), (mid_x, mid_y - box_r), (70, 60, 50), 1)
            cv2.line(overlay, (mid_x, mid_y + box_r), (mid_x, h), (70, 60, 50), 1)
            cv2.line(overlay, (0, mid_y), (mid_x - box_r, mid_y), (70, 60, 50), 1)
            cv2.line(overlay, (mid_x + box_r, mid_y), (w, mid_y), (70, 60, 50), 1)

            # Lane labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, "NORTH (N)", (mid_x - 45, 30), font, 0.52, (238, 211, 34), 1)
            cv2.putText(overlay, "SOUTH (S)", (mid_x - 45, h - 20), font, 0.52, (238, 211, 34), 1)
            cv2.putText(overlay, "WEST (W)", (20, mid_y - 10), font, 0.52, (238, 211, 34), 1)
            cv2.putText(overlay, "EAST (E)", (w - 110, mid_y - 10), font, 0.52, (238, 211, 34), 1)

            cv2.addWeighted(overlay, 0.5, out_frame, 0.5, 0, out_frame)

        # 2. Region of Interest (ROI) Overlay
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            rx1, rx2 = max(0, min(rx1, rx2)), min(w, max(rx1, rx2))
            ry1, ry2 = max(0, min(ry1, ry2)), min(h, max(ry1, ry2))

            roi_overlay = out_frame.copy()
            cv2.rectangle(roi_overlay, (rx1, ry1), (rx2, ry2), (0, 165, 255), -1)  # Translucent amber/orange
            cv2.addWeighted(roi_overlay, 0.18, out_frame, 0.82, 0, out_frame)
            cv2.rectangle(out_frame, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            cv2.putText(
                out_frame,
                "ZONE ROI",
                (rx1 + 6, max(18, ry1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Virtual Counting Line (Tripwire) Overlay
        if tripwire_line is not None:
            p1, p2 = tripwire_line
            # Glowing double-line
            cv2.line(out_frame, p1, p2, (30, 200, 255), 3, cv2.LINE_AA)
            cv2.line(out_frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out_frame, p1, 4, (30, 200, 255), -1)
            cv2.circle(out_frame, p2, 4, (30, 200, 255), -1)

            tw_label = tripwire_text or "VIRTUAL COUNTING TRIPWIRE"
            mx = int((p1[0] + p2[0]) / 2)
            my = max(20, int((p1[1] + p2[1]) / 2) - 8)
            cv2.putText(
                out_frame,
                tw_label,
                (mx - 80, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 200, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Historical Motion Trajectory Trails
        if show_trails and tracker is not None:
            for track in tracker.active_tracks.values():
                if len(track.trajectory) > 1:
                    color = class_colors.get(track.class_name, (238, 211, 34))
                    for k in range(1, len(track.trajectory)):
                        pt1 = track.trajectory[k - 1]
                        pt2 = track.trajectory[k]
                        thickness = max(1, int(k / len(track.trajectory) * 2.5))
                        cv2.line(out_frame, pt1, pt2, color, thickness)

        # 5. Vehicle Bounding Boxes & Tags
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = class_colors.get(det.class_name, (238, 211, 34))

            # Bounding box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out_frame, det.center, 3, color, -1)

            if show_labels:
                # Format label with persistent Track ID where available
                if show_tracks and det.track_id is not None:
                    label_text = f"{det.class_name.upper()} #{det.track_id} {int(det.confidence * 100)}% [{det.lane_id}]"
                else:
                    label_text = f"{det.class_name.upper()} {int(det.confidence * 100)}% [{det.lane_id}]"

                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, y1 - th - 6)
                cv2.rectangle(out_frame, (x1, tag_y1), (x1 + tw + 6, y1), (15, 23, 42), -1)
                cv2.rectangle(out_frame, (x1, tag_y1), (x1 + tw + 6, y1), color, 1)

                cv2.putText(
                    out_frame,
                    label_text,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Optional HUD Header Banner
        if header_text:
            banner_overlay = out_frame.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, 32), (15, 23, 42), -1)
            cv2.addWeighted(banner_overlay, 0.75, out_frame, 0.25, 0, out_frame)
            cv2.putText(
                out_frame,
                header_text,
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (238, 211, 34),
                1,
                cv2.LINE_AA,
            )

        return out_frame
