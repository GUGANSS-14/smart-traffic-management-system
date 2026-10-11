"""
Unit Tests for Vehicle Detector & Multi-Object Tracking Pipeline.
Uses synthetic frames and mocked detection inputs. Zero real video files required.
"""

import os
import pytest
import numpy as np

from modules.vehicle_detector import (
    Detection,
    VehicleTrack,
    VehicleTracker,
    OpenCVContourDetector,
    PretrainedDNNVehicleDetector,
    ProceduralSyntheticDetector,
    VehicleDetector,
    ModelNotLoadedError,
)


class TestDetectionAndLaneClassification:
    def test_detection_dataclass(self):
        det = Detection(
            bbox=(10, 20, 110, 120),
            class_name="car",
            confidence=0.92,
            center=(60, 70),
            track_id=1,
            lane_id="North",
            pcu_weight=1.0,
        )
        assert det.class_name == "car"
        assert det.confidence == 0.92
        assert det.track_id == 1
        assert det.lane_id == "North"
        assert det.pcu_weight == 1.0

    def test_classify_lane_quadrants(self):
        # Frame size 800x600 (center at 400, 300)
        w, h = 800, 600

        # Point high up center -> North
        assert VehicleDetector._classify_lane(400, 100, w, h) == "North"
        # Point low down center -> South
        assert VehicleDetector._classify_lane(400, 500, w, h) == "South"
        # Point far left -> West
        assert VehicleDetector._classify_lane(100, 300, w, h) == "West"
        # Point far right -> East
        assert VehicleDetector._classify_lane(700, 300, w, h) == "East"


class TestVehicleTracker:
    def test_persistent_track_id_and_movement(self):
        tracker = VehicleTracker(max_age=5, iou_threshold=0.2)

        # Frame 1: Vehicle 1 appears
        d1 = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9, center=(150, 150))]
        res1 = tracker.update(d1, timestamp=0.0)
        assert len(res1) == 1
        assert res1[0].track_id == 1
        assert tracker.get_unique_vehicle_count() == 1
        assert tracker.get_active_vehicle_count() == 1

        # Frame 2: Vehicle 1 moves slightly
        d2 = [Detection(bbox=(105, 105, 205, 205), class_name="car", confidence=0.92, center=(155, 155))]
        res2 = tracker.update(d2, timestamp=0.033)
        assert len(res2) == 1
        # Track ID must remain persistent
        assert res2[0].track_id == 1
        # Must not increment unique count for the same moving vehicle
        assert tracker.get_unique_vehicle_count() == 1

    def test_no_double_counting_stopped_vehicle(self):
        tracker = VehicleTracker(max_age=10)

        # Vehicle stops at red light for 15 frames
        for f in range(15):
            d = [Detection(bbox=(200, 200, 300, 300), class_name="truck", confidence=0.88, center=(250, 250))]
            res = tracker.update(d, timestamp=f * 0.033)
            assert res[0].track_id == 1
            # Unique vehicle count must strictly stay 1
            assert tracker.get_unique_vehicle_count() == 1
            assert tracker.get_active_vehicle_count() == 1

        counts_by_class = tracker.get_unique_counts_by_class()
        assert counts_by_class["truck"] == 1
        assert counts_by_class["car"] == 0

    def test_multiple_unique_vehicles_and_class_separation(self):
        tracker = VehicleTracker(max_age=5)

        # Frame 1: Car and Motorcycle appear
        d1 = [
            Detection(bbox=(50, 50, 150, 150), class_name="car", confidence=0.9, center=(100, 100)),
            Detection(bbox=(400, 100, 440, 160), class_name="motorcycle", confidence=0.85, center=(420, 130)),
        ]
        res1 = tracker.update(d1, timestamp=0.0)
        assert len(res1) == 2
        id_car = res1[0].track_id
        id_moto = res1[1].track_id
        assert id_car != id_moto
        assert tracker.get_unique_vehicle_count() == 2

        # Frame 2: New Bus appears along with existing vehicles
        d2 = [
            Detection(bbox=(55, 55, 155, 155), class_name="car", confidence=0.9, center=(105, 105)),
            Detection(bbox=(402, 102, 442, 162), class_name="motorcycle", confidence=0.85, center=(422, 132)),
            Detection(bbox=(250, 300, 350, 480), class_name="bus", confidence=0.95, center=(300, 390)),
        ]
        res2 = tracker.update(d2, timestamp=0.033)
        assert len(res2) == 3
        assert res2[0].track_id == id_car
        assert res2[1].track_id == id_moto
        assert res2[2].track_id == 3  # New persistent ID
        assert tracker.get_unique_vehicle_count() == 3

        breakdown = tracker.get_unique_counts_by_class()
        assert breakdown["car"] == 1
        assert breakdown["motorcycle"] == 1
        assert breakdown["bus"] == 1
        assert breakdown["truck"] == 0

    def test_tracker_reset(self):
        tracker = VehicleTracker()
        d = [Detection(bbox=(10, 10, 50, 50), class_name="car", confidence=0.9, center=(30, 30))]
        tracker.update(d)
        assert tracker.get_unique_vehicle_count() == 1

        tracker.reset()
        assert tracker.get_unique_vehicle_count() == 0
        assert tracker.get_active_vehicle_count() == 0

    def test_traffic_flow_rate_calculation(self):
        tracker = VehicleTracker()
        # 10 distinct vehicles (e.g. far apart) in 30 seconds -> (10 / 30) * 3600 = 1200 veh/h
        dets = [
            Detection(bbox=(i * 120, 50, i * 120 + 50, 100), class_name="car", confidence=0.9, center=(i * 120 + 25, 75))
            for i in range(10)
        ]
        tracker.update(dets, timestamp=0.0)
        assert tracker.get_unique_vehicle_count() == 10

        flow_rate = tracker.compute_traffic_flow_rate(elapsed_seconds=30.0)
        assert flow_rate == 1200.0

    def test_explain_traffic_metrics(self):
        explanations = VehicleTracker.explain_traffic_metrics()
        assert "unique_tracks" in explanations
        assert "vehicles_visible_in_frame" in explanations
        assert "estimated_traffic_flow" in explanations
        assert len(explanations["unique_tracks"]) > 50


class TestOpenCVContourDetector:
    def test_empty_frame(self):
        det = OpenCVContourDetector()
        assert det.detect(None) == []
        assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_blank_frame_no_false_positives(self):
        det = OpenCVContourDetector()
        frame = np.full((300, 400, 3), (30, 30, 30), dtype=np.uint8)
        # Background subtractor initializes
        for _ in range(5):
            results = det.detect(frame)
        assert len(results) == 0


class TestPretrainedDNNVehicleDetector:
    def test_missing_weights_handling(self):
        det = PretrainedDNNVehicleDetector(model_path="nonexistent_test_weights.onnx")
        assert det.is_loaded is False
        assert det.load_error is not None
        assert "not found" in det.load_error.lower()

        # Calling detect on missing weights must raise ModelNotLoadedError
        with pytest.raises(ModelNotLoadedError) as exc_info:
            det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert "not found" in str(exc_info.value).lower()

    def test_empty_frame_handling(self):
        # Test with loaded model if available, or verified instance
        det = PretrainedDNNVehicleDetector()
        if det.is_loaded:
            assert det.detect(None) == []
            assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_configurable_thresholds_and_input_size(self):
        det = PretrainedDNNVehicleDetector(conf_threshold=0.35, input_size=(416, 416))
        assert det.conf_threshold == 0.35
        assert det.input_size == (416, 416)

        det.set_conf_threshold(0.60)
        assert det.conf_threshold == 0.60

        det.set_input_size((640, 640))
        assert det.input_size == (640, 640)


class TestVehicleDetectorFacade:
    def test_status_reporting(self):
        det = VehicleDetector(backend="Procedural Simulation", conf_threshold=0.50)
        status = det.get_status()

        assert status["backend"] == "Procedural Simulation"
        assert status["is_simulated"] is True
        assert status["confidence_threshold"] == 0.50
        assert "car" in status["supported_classes"]
        assert status["pcu_standards"]["car"] == 1.0
        assert status["pcu_standards"]["bus"] == 3.0

    def test_backend_switching(self):
        det = VehicleDetector()
        assert det.backend == "Pretrained DNN / YOLO"

        det.set_backend("OpenCV MOG2 (Classical CV)")
        assert det.backend == "OpenCV MOG2 (Classical CV)"

        det.set_backend("Procedural Simulation")
        assert det.backend == "Procedural Simulation"

    def test_draw_detections_hud(self):
        frame = np.full((540, 720, 3), (25, 30, 40), dtype=np.uint8)
        detections = [
            Detection(bbox=(50, 50, 150, 150), class_name="car", confidence=0.91, center=(100, 100), track_id=1, lane_id="North"),
            Detection(bbox=(300, 200, 420, 320), class_name="bus", confidence=0.89, center=(360, 260), track_id=2, lane_id="South"),
        ]

        annotated = VehicleDetector.draw_detections(
            frame,
            detections,
            draw_lanes=True,
            show_labels=True,
            show_tracks=True,
            header_text="Active: 2 | Flow: 120 veh/h",
        )

        assert annotated is not None
        assert annotated.shape == frame.shape
        # Ensure pixels have been modified (HUD drawn)
        assert not np.array_equal(annotated, frame)
