"""
Unit Tests for Video Processor, Frame Sampling & Synthetic Simulation Engine.
Zero real video files required.
"""

import os
import pytest
import numpy as np

from modules.video_processor import (
    SyntheticVehicle,
    SyntheticTrafficStream,
    VideoProcessor,
    FrameAnalysisResult,
    UnsupportedVideoFormatError,
    CorruptedVideoError,
)
from modules.vehicle_detector import VehicleDetector


class TestSyntheticEngine:
    def test_synthetic_vehicle_dimensions_and_pcu(self):
        car = SyntheticVehicle(1, "North", "car", 0.2, 0.01, (200, 200, 40))
        assert car.pcu == 1.0
        assert car.width == 24

        bus = SyntheticVehicle(2, "South", "bus", 0.4, 0.005, (30, 150, 240))
        assert bus.pcu == 3.0
        assert bus.width == 28

        truck = SyntheticVehicle(3, "East", "truck", 0.1, 0.006, (60, 60, 220))
        assert truck.pcu == 3.0

        bike = SyntheticVehicle(4, "West", "motorcycle", 0.5, 0.009, (80, 220, 120))
        assert bike.pcu == 0.5

    def test_synthetic_traffic_stream(self):
        stream = SyntheticTrafficStream(width=640, height=480)
        assert len(stream.vehicles) > 0

        # Step simulation physics
        stream.step()
        frame = stream.render_frame()
        assert frame.shape == (480, 640, 3)

        # Ground truth detections
        detections = stream.get_detections(frame)
        assert len(detections) > 0
        for d in detections:
            assert d.track_id is not None  # Must have persistent track IDs
            assert d.class_name in ["car", "bus", "truck", "motorcycle"]
            assert d.lane_id in ["North", "South", "East", "West"]

    def test_signal_phase_override(self):
        stream = SyntheticTrafficStream()
        stream.set_signal_phase("East-West")
        assert stream.active_phase == "East-West"

        stream.set_signal_phase("North-South")
        assert stream.active_phase == "North-South"


class TestVideoProcessor:
    def test_synthetic_frame_processing(self):
        detector = VehicleDetector(backend="Procedural Simulation")
        vp = VideoProcessor(detector=detector)

        info = vp.get_video_info()
        assert info["is_synthetic"] is True
        assert info["width"] == 720
        assert info["height"] == 540

        frame, raw_dets = vp.read_synthetic_frame()
        assert frame is not None
        assert len(raw_dets) > 0

        result = vp.process_frame(frame, is_synthetic=True)
        assert isinstance(result, FrameAnalysisResult)
        assert result.active_vehicles_count > 0
        assert result.unique_vehicles_count > 0
        assert result.pcu_total > 0
        assert "car" in result.detection_counts
        assert result.annotated_frame.shape == frame.shape

    def test_empty_frame_handling(self):
        vp = VideoProcessor()

        # None frame
        res_none = vp.process_frame(None)
        assert res_none.active_vehicles_count == 0
        assert res_none.detections == []
        assert res_none.annotated_frame.shape == (540, 720, 3)

        # Zero sized frame
        empty_mat = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = vp.process_frame(empty_mat)
        assert res_empty.active_vehicles_count == 0
        assert res_empty.detections == []

    def test_unsupported_video_formats(self, tmp_path):
        vp = VideoProcessor()

        # Dummy text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a video")

        with pytest.raises(UnsupportedVideoFormatError) as exc_info:
            vp.open_video_file(str(txt_file))
        assert "unsupported" in str(exc_info.value).lower()

        # Dummy CSV file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2")

        with pytest.raises(UnsupportedVideoFormatError):
            vp.open_video_file(str(csv_file))

    def test_corrupted_or_empty_video_file(self, tmp_path):
        vp = VideoProcessor()

        # 0-byte MP4 file
        empty_mp4 = tmp_path / "empty.mp4"
        empty_mp4.touch()

        with pytest.raises(CorruptedVideoError) as exc_info:
            vp.open_video_file(str(empty_mp4))
        assert "0 bytes" in str(exc_info.value) or "corrupted" in str(exc_info.value).lower()

    def test_nonexistent_file(self):
        vp = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            vp.open_video_file("nonexistent_traffic_video.mp4")

    def test_batch_frame_processing(self):
        vp = VideoProcessor(detector=VehicleDetector(backend="Procedural Simulation"))

        # Create small batch of synthetic frames
        f1, _ = vp.read_synthetic_frame()
        f2, _ = vp.read_synthetic_frame()
        f3, _ = vp.read_synthetic_frame()

        results = vp.process_batch([f1, f2, f3], start_index=10, fps=30.0)
        assert len(results) == 3
        assert results[0].frame_index == 10
        assert results[1].frame_index == 11
        assert results[2].frame_index == 12

        # Timestamps should be increasing
        assert results[0].timestamp < results[1].timestamp < results[2].timestamp


class TestVideoHelpers:
    def test_check_tripwire_crossing_horizontal(self):
        crossed_ids = set()
        p1 = (100, 200)
        p2 = (300, 200)

        # Vehicle crosses near line
        assert VideoProcessor.check_tripwire_crossing((200, 205), p1, p2, track_id=42, crossed_set=crossed_ids)
        assert 42 in crossed_ids

        # Deduplication: already crossed vehicle returns True and set retains single entry
        assert VideoProcessor.check_tripwire_crossing((200, 205), p1, p2, track_id=42, crossed_set=crossed_ids)
        assert len(crossed_ids) == 1

        # Far away point outside tolerance
        assert not VideoProcessor.check_tripwire_crossing((200, 250), p1, p2, track_id=99, crossed_set=crossed_ids)
        assert 99 not in crossed_ids

        # None track id returns False
        assert not VideoProcessor.check_tripwire_crossing((200, 205), p1, p2, track_id=None, crossed_set=crossed_ids)

    def test_check_tripwire_crossing_vertical(self):
        crossed_ids = set()
        p1 = (200, 100)
        p2 = (200, 300)

        # Vehicle crosses vertical line
        assert VideoProcessor.check_tripwire_crossing((205, 200), p1, p2, track_id=10, crossed_set=crossed_ids)
        assert 10 in crossed_ids

        # Far off x coordinate
        assert not VideoProcessor.check_tripwire_crossing((250, 200), p1, p2, track_id=11, crossed_set=crossed_ids)

    def test_is_inside_roi(self):
        roi = (50, 50, 200, 200)
        # Inside
        assert VideoProcessor.is_inside_roi((100, 100), roi) is True
        # On boundary
        assert VideoProcessor.is_inside_roi((50, 50), roi) is True
        assert VideoProcessor.is_inside_roi((200, 200), roi) is True
        # Outside
        assert VideoProcessor.is_inside_roi((20, 20), roi) is False
        assert VideoProcessor.is_inside_roi((250, 100), roi) is False
        # None ROI returns True for all points
        assert VideoProcessor.is_inside_roi((999, 999), None) is True

    def test_get_video_metadata_invalid_file(self):
        # Nonexistent file returns None
        assert VideoProcessor.get_video_metadata("nonexistent_path_xyz.mp4") is None
        # Empty string returns None
        assert VideoProcessor.get_video_metadata("") is None
