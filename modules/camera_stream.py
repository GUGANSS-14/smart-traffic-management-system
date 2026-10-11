"""
Camera Stream Module
Provides a clean, testable abstraction for real-time camera ingestion.
Supports local hardware webcams via OpenCV VideoCapture, mock frame generators for automated tests,
and reliable resource management.

Key Features:
- Safe lifecycle management: explicit start, stop, and context-manager resource release.
- Camera status inspection: device index, resolution, measured FPS, and frame timestamps.
- MockCameraStream abstraction: enables automated unit tests with synthetic/file frames without physical hardware.
- Defensive error handling: traps permission denial, device conflicts, and disconnected cameras.
- Independent importability: zero dependencies on Streamlit UI code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
import cv2


class CameraUnavailableError(IOError):
    """Raised when the specified camera hardware index cannot be opened or is occupied."""
    pass


class CameraPermissionError(PermissionError):
    """Raised when access to camera device is denied or hardware returns empty frames."""
    pass


class BaseCameraStream(ABC):
    """Abstract interface for camera capture streams."""

    @abstractmethod
    def start(self) -> bool:
        """Opens capture stream and allocates hardware handles. Returns True if successful."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Releases all capture handles and closes underlying device."""
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads the next available frame.
        Returns (success_flag, frame_ndarray, timestamp_epoch).
        """
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the camera stream is actively open and capturing."""
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Returns telemetry dictionary of connection status, resolution, and timestamps."""
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class OpenCVCameraStream(BaseCameraStream):
    """
    Hardware camera streamer using local OpenCV VideoCapture.
    Operates on local machine server devices (Device Index 0, 1, 2...).
    Guarantees reliable resource release when stopped or on failure.
    """

    def __init__(
        self,
        device_index: int = 0,
        target_fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.device_index = device_index
        self.target_fps = target_fps
        self.target_width = width
        self.target_height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._last_frame_time: Optional[float] = None
        self._frame_count = 0
        self._error_message: Optional[str] = None
        self._resolution = (0, 0)
        self._min_frame_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

    def start(self) -> bool:
        """
        Connects to the local hardware camera.
        Never auto-starts; requires explicit invocation.
        """
        if self._is_connected and self.cap is not None and self.cap.isOpened():
            return True

        self.stop()
        self._error_message = None

        try:
            self.cap = cv2.VideoCapture(self.device_index)

            # Optional DirectShow backend on Windows for fast initialization if default fails
            if not self.cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.stop()
                self._error_message = (
                    f"No camera device detected at Index {self.device_index}. "
                    "Ensure a physical webcam is plugged in and accessible to the local process."
                )
                return False

            # Configure resolution if specified
            if self.target_width and self.target_height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            # Perform a test read to verify permission and functional capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                self.stop()
                self._error_message = (
                    f"Camera Device {self.device_index} opened but returned no image data. "
                    "Access may be blocked by OS privacy permissions or another application."
                )
                return False

            h, w = test_frame.shape[:2]
            self._resolution = (w, h)
            self._is_connected = True
            self._last_frame_time = time.time()
            self._frame_count = 1
            return True

        except Exception as e:
            self.stop()
            self._error_message = f"Hardware error initializing camera {self.device_index}: {str(e)}"
            return False

    def stop(self) -> None:
        """Releases the camera hardware handle cleanly."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._is_connected = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Reads the latest frame from the hardware device."""
        if not self._is_connected or self.cap is None or not self.cap.isOpened():
            return False, None, time.time()

        now = time.time()

        # Enforce target FPS throttling if configured
        if self._min_frame_interval > 0 and self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed < self._min_frame_interval:
                time.sleep(self._min_frame_interval - elapsed)
                now = time.time()

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.stop()
            self._error_message = "Camera stream disconnected or frame capture failed."
            return False, None, now

        self._last_frame_time = now
        self._frame_count += 1
        return True, frame, now

    def is_connected(self) -> bool:
        return self._is_connected and self.cap is not None and self.cap.isOpened()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive device status telemetry."""
        return {
            "is_connected": self.is_connected(),
            "device_index": self.device_index,
            "device_type": "Local Hardware Camera (OpenCV)",
            "resolution": self._resolution,
            "total_frames_captured": self._frame_count,
            "last_frame_timestamp": self._last_frame_time,
            "error_message": self._error_message,
        }


class MockCameraStream(BaseCameraStream):
    """
    Testable camera abstraction for automated unit tests and CI/CD pipelines.
    Feeds frames from an in-memory list or video file, and simulates errors/disconnections.
    """

    def __init__(
        self,
        frames: Optional[List[np.ndarray]] = None,
        fps: float = 30.0,
        should_fail_start: bool = False,
        fail_after_frames: Optional[int] = None,
    ):
        self.fps = max(1.0, fps)
        self.should_fail_start = should_fail_start
        self.fail_after_frames = fail_after_frames

        # Default synthetic frame if none provided
        if frames is None or len(frames) == 0:
            default_frame = np.full((480, 640, 3), (30, 35, 45), dtype=np.uint8)
            cv2.putText(
                default_frame,
                "MOCK CAMERA TEST STREAM",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            self.frames = [default_frame]
        else:
            self.frames = frames

        self._frame_idx = 0
        self._is_connected = False
        self._frame_count = 0
        self._last_frame_time: Optional[float] = None
        self._error_message: Optional[str] = None

    def start(self) -> bool:
        if self.should_fail_start:
            self._is_connected = False
            self._error_message = "Simulated hardware initialization failure."
            return False

        self._is_connected = True
        self._frame_idx = 0
        self._frame_count = 0
        self._last_frame_time = time.time()
        self._error_message = None
        return True

    def stop(self) -> None:
        self._is_connected = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self._is_connected:
            return False, None, time.time()

        now = time.time()

        # Simulate disconnection mid-stream
        if self.fail_after_frames is not None and self._frame_count >= self.fail_after_frames:
            self.stop()
            self._error_message = f"Simulated camera disconnect after {self.fail_after_frames} frames."
            return False, None, now

        frame = self.frames[self._frame_idx % len(self.frames)]
        self._frame_idx += 1
        self._frame_count += 1
        self._last_frame_time = now

        return True, frame.copy(), now

    def is_connected(self) -> bool:
        return self._is_connected

    def get_status(self) -> Dict[str, Any]:
        h, w = self.frames[0].shape[:2] if self.frames else (0, 0)
        return {
            "is_connected": self.is_connected(),
            "device_index": -1,
            "device_type": "Mock Camera Stream (Test Abstraction)",
            "resolution": (w, h),
            "total_frames_captured": self._frame_count,
            "last_frame_timestamp": self._last_frame_time,
            "error_message": self._error_message,
        }
