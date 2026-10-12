
"""
Smart Traffic Management System - Core Modules
AI-based traffic analytics and graph-theoretic optimization.
"""


from .vehicle_detector import VehicleDetector, Detection
from .video_processor import VideoProcessor, SyntheticTrafficStream
from .traffic_analyzer import TrafficAnalyzer, LaneMetrics
from .signal_optimizer import SignalOptimizer, SignalPlan, SignalComparison
from .graph_model import IntersectionGraphNetwork, RoadSegment, FourWayIntersectionGraph
from .discrete_math import DiscreteMathLab
from .camera_stream import (
    BaseCameraStream,
    OpenCVCameraStream,
    MockCameraStream,
    CameraUnavailableError,
    CameraPermissionError,
)

__all__ = [
    "VehicleDetector",
    "Detection",
    "VideoProcessor",
    "SyntheticTrafficStream",
    "TrafficAnalyzer",
    "LaneMetrics",
    "SignalOptimizer",
    "SignalPlan",
    "SignalComparison",
    "IntersectionGraphNetwork",
    "FourWayIntersectionGraph",
    "RoadSegment",
    "DiscreteMathLab",
    "BaseCameraStream",
    "OpenCVCameraStream",
    "MockCameraStream",
    "CameraUnavailableError",
    "CameraPermissionError",
]
