"""
Unit Tests for Camera Stream Abstraction.
Tests local hardware camera handling, lifecycle resource release,
and MockCameraStream frame generators without physical hardware.
"""

import pytest
import numpy as np

from modules.camera_stream import (
    BaseCameraStream,
    OpenCVCameraStream,
    MockCameraStream,
    CameraUnavailableError,
    CameraPermissionError,
)


class TestMockCameraStream:
    def test_mock_camera_stream_lifecycle(self):
        stream = MockCameraStream(fps=30.0)
        assert not stream.is_connected()

        # Explicit start
        success = stream.start()
        assert success is True
        assert stream.is_connected() is True

        # Read frames
        ret, frame, ts = stream.read_frame()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert ts > 0

        # Read second frame
        ret2, frame2, ts2 = stream.read_frame()
        assert ret2 is True
        assert ts2 >= ts

        # Status inspection
        status = stream.get_status()
        assert status["is_connected"] is True
        assert status["total_frames_captured"] == 2
        assert status["resolution"] == (640, 480)

        # Explicit stop
        stream.stop()
        assert stream.is_connected() is False

        # Reading after stop returns False
        ret_after, frame_after, _ = stream.read_frame()
        assert ret_after is False
        assert frame_after is None

    def test_mock_camera_custom_frames(self):
        # Custom synthetic frames
        f1 = np.zeros((100, 200, 3), dtype=np.uint8)
        f2 = np.ones((100, 200, 3), dtype=np.uint8) * 255
        stream = MockCameraStream(frames=[f1, f2])

        stream.start()
        _, r1, _ = stream.read_frame()
        _, r2, _ = stream.read_frame()
        _, r3, _ = stream.read_frame()  # Loops back to f1

        assert np.array_equal(r1, f1)
        assert np.array_equal(r2, f2)
        assert np.array_equal(r3, f1)
        stream.stop()

    def test_mock_camera_simulated_start_failure(self):
        stream = MockCameraStream(should_fail_start=True)
        success = stream.start()
        assert success is False
        assert stream.is_connected() is False
        status = stream.get_status()
        assert "failure" in status["error_message"].lower()

    def test_mock_camera_simulated_disconnect(self):
        stream = MockCameraStream(fail_after_frames=3)
        stream.start()

        # Read 3 frames successfully
        for _ in range(3):
            ret, frame, _ = stream.read_frame()
            assert ret is True

        # 4th frame triggers simulated disconnect
        ret, frame, _ = stream.read_frame()
        assert ret is False
        assert frame is None
        assert stream.is_connected() is False
        status = stream.get_status()
        assert "disconnect" in status["error_message"].lower()

    def test_context_manager(self):
        with MockCameraStream() as cam:
            assert cam.is_connected() is True
            ret, frame, _ = cam.read_frame()
            assert ret is True

        # Automatically released on exit
        assert cam.is_connected() is False


class TestOpenCVCameraStream:
    def test_nonexistent_hardware_index_handled_gracefully(self):
        # High index that does not exist
        cam = OpenCVCameraStream(device_index=9999)
        assert not cam.is_connected()

        # Start should return False gracefully, not raise an unhandled crash
        success = cam.start()
        assert success is False
        assert cam.is_connected() is False

        status = cam.get_status()
        assert status["is_connected"] is False
        assert status["error_message"] is not None

        # Clean release
        cam.stop()
        assert cam.is_connected() is False

    def test_opencv_context_manager_cleanup(self):
        cam = OpenCVCameraStream(device_index=9999)
        with cam:
            pass
        assert cam.is_connected() is False
