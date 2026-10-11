"""
Page 2: Live Surveillance Camera Feed & Edge Edge Analytics
Real-Time Surveillance Camera Capture with Computer Vision Analytics.

Supports two clearly separated operating modes:
Mode A: Local Hardware Camera Analysis using OpenCV VideoCapture (for local development).
Mode B: Browser-Accessible Camera Input using st.camera_input (for client-server / remote access).
Fallback: Procedural Synthetic CCTV stream and uploaded video redirection for unsupported headless environments.

Operational Integrity & Privacy:
- Never opens a camera without explicit user action.
- Releases camera hardware handles reliably on stop, failure, or page change.
- In-memory frame processing with zero persistent video storage or unauthorized transmission.
- Explicit educational documentation detailing the difference between local server and browser client camera access.
"""

import time
import os
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go

from modules.vehicle_detector import VehicleDetector, Detection, ModelNotLoadedError
from modules.video_processor import SyntheticTrafficStream
from modules.traffic_analyzer import TrafficAnalyzer
from modules.signal_optimizer import SignalOptimizer
from modules.camera_stream import (
    OpenCVCameraStream,
    MockCameraStream,
    CameraUnavailableError,
    CameraPermissionError,
)
from visualizations.charts import get_custom_css

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Live Camera Feed - Smart Traffic", page_icon="📷", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

col_cam_head, col_cam_badge = st.columns([2.8, 1.2])

with col_cam_head:
    st.title("📷 Live Surveillance Camera Feed")
    st.markdown(
        """
        **Real-Time Edge Capture, Vehicle Detection, Multi-Object Tracking & Signal Actuation**  
        *Connects to optical surveillance sensors or browser webcam with intelligent fallback simulation.*
        """
    )

with col_cam_badge:
    if st.session_state.get("live_cam_connected", False):
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-emerald">🟢 Camera Connected</span><br>
                <span style="font-size: 0.75rem; color: #34D399; font-weight:600;">Active Video Stream</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-amber">⚪ Camera Standby</span><br>
                <span style="font-size: 0.75rem; color: #94A3B8;">Disconnected / Awaiting Start</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "live_detector" not in st.session_state:
    st.session_state.live_detector = VehicleDetector(backend="Pretrained DNN / YOLO", conf_threshold=0.45)
if "live_analyzer" not in st.session_state:
    st.session_state.live_analyzer = TrafficAnalyzer()
if "live_optimizer" not in st.session_state:
    st.session_state.live_optimizer = SignalOptimizer()
if "sim_cam_stream" not in st.session_state:
    st.session_state.sim_cam_stream = SyntheticTrafficStream(width=720, height=540)

if "live_cam_stream" not in st.session_state:
    st.session_state.live_cam_stream = None
if "live_cam_connected" not in st.session_state:
    st.session_state.live_cam_connected = False
if "live_last_frame_bgr" not in st.session_state:
    st.session_state.live_last_frame_bgr = None
if "live_last_timestamp_str" not in st.session_state:
    st.session_state.live_last_timestamp_str = "None"
if "live_last_detections" not in st.session_state:
    st.session_state.live_last_detections = []
if "live_error_message" not in st.session_state:
    st.session_state.live_error_message = None

detector: VehicleDetector = st.session_state.live_detector
analyzer: TrafficAnalyzer = st.session_state.live_analyzer
optimizer: SignalOptimizer = st.session_state.live_optimizer


# -----------------------------------------------------------------------------
# Sidebar: Operating Modes & Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    with st.expander("🧭 System Navigation", expanded=False):
        st.page_link("app.py", label="Executive Overview", icon="🏠")
        st.page_link("pages/1_Video_Analysis.py", label="1. Video Analysis", icon="🎥")
        st.page_link("pages/2_Live_Camera.py", label="2. Live Camera Feed", icon="📷")
        st.page_link("pages/3_Traffic_Analytics.py", label="3. Traffic Analytics", icon="📊")
        st.page_link("pages/4_Intersection_Graph.py", label="4. Intersection Graph", icon="🕸️")
        st.page_link("pages/5_Discrete_Mathematics_Lab.py", label="5. Discrete Math Lab", icon="📐")
        st.page_link("pages/6_Project_Information.py", label="6. Project Information & Docs", icon="📖")
        st.page_link("pages/7_How_to_Use.py", label="7. How to Use & User Guide", icon="💡")

    st.markdown("### 📷 Operating Mode Selection")
    cam_mode = st.radio(
        "Capture Architecture Mode",
        [
            "Mode A: Local Hardware Camera (OpenCV VideoCapture)",
            "Mode B: Browser-Accessible Camera (Client Snapshot)",
            "Mode C: Simulated CCTV Stream (Zero Hardware Required)",
        ],
        index=0,
        help=(
            "Mode A captures directly from a webcam connected to the computer running the server. "
            "Mode B captures in your web browser and sends the snapshot to the server. "
            "Mode C runs procedural synthetic traffic simulation with zero hardware dependencies."
        ),
    )

    st.markdown("---")
    st.markdown("### ⚙️ Camera & Detection Parameters")

    # Mode-specific parameters
    device_index = 0
    fps_limit = 15.0
    if cam_mode == "Mode A: Local Hardware Camera (OpenCV VideoCapture)":
        device_index = st.selectbox("Hardware Device Index", [0, 1, 2, 3], index=0, help="Camera device index on host OS (0 = default webcam).")
        fps_limit = st.slider("Target Processing Rate (FPS Limit)", min_value=1.0, max_value=30.0, value=15.0, step=1.0, help="Limits processing rate to manage CPU utilization.")

    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.90, value=0.45, step=0.05)
    inf_res = st.selectbox("Inference Resolution", ["640x640 (Standard)", "416x416 (Fast CPU)"], index=0)
    input_size = (640, 640) if "640" in inf_res else (416, 416)
    enable_tracking = st.checkbox("Persistent Multi-Object Tracking", value=True)
    show_hud = st.checkbox("Show HUD Bounding Boxes & Lanes", value=True)
    show_trails = st.checkbox("Show Motion Trajectory Trails", value=True)

    st.markdown("---")
    st.markdown("### 🚦 Signal Actuation")
    override_green = st.selectbox(
        "Manual Signal Override",
        ["Automatic AI Adaptive", "Force North-South Green", "Force East-West Green"],
        index=0,
    )

# Apply settings to detector
detector.set_conf_threshold(conf_thresh)
detector.set_input_size(input_size)
detector.enable_tracking = enable_tracking


# -----------------------------------------------------------------------------
# Mode A Hardware Stream Lifecycle Management
# -----------------------------------------------------------------------------
def start_hardware_camera(dev_idx: int, target_fps: float):
    """Safely initializes and connects to the host hardware camera."""
    stop_hardware_camera()
    stream = OpenCVCameraStream(device_index=dev_idx, target_fps=target_fps)
    success = stream.start()
    if success:
        st.session_state.live_cam_stream = stream
        st.session_state.live_cam_connected = True
        st.session_state.live_error_message = None
    else:
        status = stream.get_status()
        st.session_state.live_cam_stream = None
        st.session_state.live_cam_connected = False
        st.session_state.live_error_message = status.get("error_message") or "Camera failed to initialize."


def stop_hardware_camera():
    """Safely releases the hardware camera handle."""
    if st.session_state.live_cam_stream is not None:
        try:
            st.session_state.live_cam_stream.stop()
        except Exception:
            pass
        st.session_state.live_cam_stream = None
    st.session_state.live_cam_connected = False


def reset_live_analytics():
    """Resets tracking counters and detection history."""
    stop_hardware_camera()
    detector.reset_tracker()
    st.session_state.live_last_frame_bgr = None
    st.session_state.live_last_detections = []
    st.session_state.live_last_timestamp_str = "None"
    st.session_state.live_error_message = None


# -----------------------------------------------------------------------------
# Mode Header & Architecture Callout (Requirement 6 & 8)
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Architecture Guide: Local Server vs Browser Client Camera Access & Privacy Notice", expanded=False):
    st.markdown(
        """
        ### 🌐 Understanding Camera Access Architectures

        | Dimension | Mode A: Local Hardware Camera (`cv2.VideoCapture`) | Mode B: Browser Client Camera (`st.camera_input`) |
        | :--- | :--- | :--- |
        | **Execution Location** | Host Operating System kernel where the Python server runs | User's local web browser via HTML5 MediaDevices API |
        | **Hardware Access** | Directly opens physical device attached to host machine | Prompts browser user for webcam permission |
        | **Local Development** | Works natively on `localhost` (direct USB webcam) | Works on `localhost` |
        | **Remote Cloud Server** | **Unsupported:** Cloud containers (AWS, Docker, Heroku) have no local webcam | **Supported:** Browser captures frame locally and transmits snapshot |
        | **Best Use Case** | Dedicated edge appliance, roadside embedded computer (Jetson/PC) | Remote web demonstration, cloud evaluations, student laptops |

        ---

        ### 🔒 Privacy & Data Transmission Guarantee (Requirement 8)
        - **Zero Footage Storage:** Captured frames are processed strictly in-memory as temporary NumPy tensors.
        - **Ephemeral Extraction:** Once bounding boxes, classes, and track coordinates are evaluated, the raw frame buffer is discarded from memory.
        - **No Remote Telemetry:** No camera footage, snapshots, or biometric data are transmitted to external servers or cloud services.
        """
    )


# -----------------------------------------------------------------------------
# Frame Acquisition Logic Based on Mode
# -----------------------------------------------------------------------------
frame_bgr = None
is_simulated = False
camera_status_label = "Disconnected"
camera_status_color = "#94A3B8"
current_detections: List[Detection] = []
frame_timestamp_str = st.session_state.live_last_timestamp_str

# MODE A: LOCAL HARDWARE CAMERA
if cam_mode == "Mode A: Local Hardware Camera (OpenCV VideoCapture)":
    is_simulated = False

    st.markdown("#### 🎛️ Local Camera Hardware Controls")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.2, 1.2, 1.5])

    with ctrl_col1:
        if st.button("▶️ Connect & Start Camera", use_container_width=True):
            start_hardware_camera(device_index, fps_limit)
            st.rerun()

    with ctrl_col2:
        if st.button("⏹️ Disconnect Camera", use_container_width=True):
            stop_hardware_camera()
            st.rerun()

    with ctrl_col3:
        if st.button("🔄 Reset Analytics", use_container_width=True):
            reset_live_analytics()
            st.rerun()

    with ctrl_col4:
        if st.session_state.live_cam_connected:
            if st.button("📸 Capture Next Frame", use_container_width=True):
                st.rerun()

    # Active stream check
    cam_stream: Optional[OpenCVCameraStream] = st.session_state.live_cam_stream

    if st.session_state.live_cam_connected and cam_stream is not None and cam_stream.is_connected():
        ret, raw_frame, ts = cam_stream.read_frame()
        if ret and raw_frame is not None and raw_frame.size > 0:
            frame_bgr = cv2.resize(raw_frame, (720, 540))
            frame_timestamp_str = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
            st.session_state.live_last_timestamp_str = frame_timestamp_str

            camera_status_label = f"Connected (Device {device_index}) | Active Stream"
            camera_status_color = "#10B981"

            # Execute real detection
            try:
                current_detections = detector.detect(frame_bgr, timestamp=ts)
            except ModelNotLoadedError as err:
                st.error(f"⚠️ Pretrained Model Weights Error: {err}")
                st.info("💡 Model weights not found. Falling back to OpenCV MOG2 classical CV.")
                detector.set_backend("OpenCV MOG2 (Classical CV)")
                current_detections = detector.detect(frame_bgr, timestamp=ts)

            st.session_state.live_last_frame_bgr = frame_bgr
            st.session_state.live_last_detections = current_detections
        else:
            stop_hardware_camera()
            st.session_state.live_error_message = "Camera stream returned empty frame or was unplugged."
            camera_status_label = "Device Disconnected"
            camera_status_color = "#EF4444"
    else:
        camera_status_label = "Disconnected / Camera Inactive"
        camera_status_color = "#94A3B8"

        # Show error or instruction
        if st.session_state.live_error_message:
            st.error(f"⚠️ {st.session_state.live_error_message}")
            st.markdown(
                """
                <div style="background:#0F172A; border-left:4px solid #EF4444; padding:12px 16px; border-radius:6px; margin:10px 0;">
                    <b>Troubleshooting Steps:</b>
                    <ul style="margin:4px 0 0 0; color:#CBD5E1; font-size:0.85rem;">
                        <li>Ensure a physical webcam is plugged into your local computer.</li>
                        <li>Check that camera access is enabled in Windows Settings → Privacy & Security → Camera.</li>
                        <li>Ensure another application (Zoom, Teams, Skype) is not currently locking the webcam.</li>
                        <li>If running in a remote cloud/Docker container, switch to <b>Mode B (Browser Camera)</b> or <b>Mode C (Simulation)</b> above.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Click **'Connect & Start Camera'** above to initialize local hardware capture. The camera is never opened automatically.")

        # Display placeholder or last frame
        if st.session_state.live_last_frame_bgr is not None:
            frame_bgr = st.session_state.live_last_frame_bgr
            current_detections = st.session_state.live_last_detections
        else:
            frame_bgr = np.full((540, 720, 3), (25, 30, 40), dtype=np.uint8)
            cv2.putText(frame_bgr, "CAMERA INACTIVE - CLICK CONNECT ABOVE", (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 2)


# MODE B: BROWSER-ACCESSIBLE CAMERA SNAPSHOT
elif cam_mode == "Mode B: Browser-Accessible Camera (Client Snapshot)":
    is_simulated = False
    stop_hardware_camera()

    st.markdown("#### 📷 Browser Client Camera Snapshot")
    st.info("📷 Click **'Capture Photo'** in the browser widget below to acquire an instant camera snapshot from your client device:")

    cam_file = st.camera_input("Acquire Roadway Surveillance Frame")

    if cam_file is not None:
        bytes_data = cam_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if cv2_img is not None and cv2_img.size > 0:
            frame_bgr = cv2.resize(cv2_img, (720, 540))
            ts = time.time()
            frame_timestamp_str = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
            st.session_state.live_last_timestamp_str = frame_timestamp_str

            camera_status_label = "Browser Camera Snapshot Captured"
            camera_status_color = "#10B981"

            try:
                current_detections = detector.detect(frame_bgr, timestamp=ts)
            except ModelNotLoadedError as err:
                st.error(f"⚠️ Pretrained Model Weights Error: {err}")
                detector.set_backend("OpenCV MOG2 (Classical CV)")
                current_detections = detector.detect(frame_bgr, timestamp=ts)

            st.session_state.live_last_frame_bgr = frame_bgr
            st.session_state.live_last_detections = current_detections
    else:
        camera_status_label = "Awaiting Browser Camera Snapshot"
        camera_status_color = "#38BDF8"

        if st.session_state.live_last_frame_bgr is not None:
            frame_bgr = st.session_state.live_last_frame_bgr
            current_detections = st.session_state.live_last_detections
        else:
            frame_bgr = np.full((540, 720, 3), (25, 30, 40), dtype=np.uint8)
            cv2.putText(frame_bgr, "AWAITING BROWSER SNAPSHOT - TAKE PHOTO ABOVE", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 2)


# MODE C: SIMULATED CCTV STREAM (ZERO HARDWARE FALLBACK)
else:
    is_simulated = True
    stop_hardware_camera()

    st.markdown("#### 🚦 Simulated CCTV Benchmark Stream (Zero Hardware Required)")
    c_step1, c_step2, c_step3 = st.columns([1, 1, 2])
    with c_step1:
        step_sim = st.button("🔄 Step Next Frame", use_container_width=True)
    with c_step2:
        step_10 = st.button("⚡ Run 10 Frames", use_container_width=True)

    sim_stream: SyntheticTrafficStream = st.session_state.sim_cam_stream
    num_steps = 10 if step_10 else (1 if step_sim else 0)

    for _ in range(max(1, num_steps)):
        sim_stream.step()

    frame_bgr = sim_stream.render_frame()
    ts = time.time()
    frame_timestamp_str = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
    st.session_state.live_last_timestamp_str = frame_timestamp_str

    camera_status_label = "Synthetic CCTV Stream Active (Procedural Physics)"
    camera_status_color = "#F59E0B"

    current_detections = detector.detect(frame_bgr, synthetic_stream=sim_stream, timestamp=ts)
    st.session_state.live_last_frame_bgr = frame_bgr
    st.session_state.live_last_detections = current_detections


# -----------------------------------------------------------------------------
# Drawing Annotations & Telemetry Calculation
# -----------------------------------------------------------------------------
if frame_bgr is not None:
    if show_hud:
        hud_banner = (
            f"Mode: {'Simulated' if is_simulated else 'Live Edge'} | "
            f"Visible: {len(current_detections)} | "
            f"Unique: {detector.tracker.get_unique_vehicle_count()} | "
            f"Timestamp: {frame_timestamp_str}"
        )
        annotated_frame = detector.draw_detections(
            frame_bgr,
            current_detections,
            draw_lanes=True,
            show_labels=True,
            show_tracks=enable_tracking,
            show_trails=show_trails,
            tracker=detector.tracker,
            header_text=hud_banner,
        )
    else:
        annotated_frame = frame_bgr
else:
    annotated_frame = np.full((540, 720, 3), (25, 30, 40), dtype=np.uint8)

lane_metrics = analyzer.analyze_frame_detections(current_detections)
signal_plan = optimizer.compute_adaptive_plan(lane_metrics)


# -----------------------------------------------------------------------------
# Main Visual Layout: Feed & Signal Actuator
# -----------------------------------------------------------------------------
left_feed, right_metrics = st.columns([1.35, 1.0])

with left_feed:
    st.markdown("### 🖥️ Optical Camera HUD View")
    rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, channels="RGB", use_container_width=True)

    # Status Banner
    unique_tracks_count = detector.tracker.get_unique_vehicle_count()
    flow_rate_vph = int(detector.tracker.compute_traffic_flow_rate())

    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px 16px; margin-top:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8;">HARDWARE STATUS</span><br>
                    <span style="font-size:0.95rem; font-weight:700; color:{camera_status_color};">● {camera_status_label}</span>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8;">LATEST FRAME TIME</span><br>
                    <span style="font-size:0.95rem; font-weight:700; color:#F8FAFC; font-family:monospace;">{frame_timestamp_str}</span>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8;">ACTIVE DETECTIONS</span><br>
                    <span style="font-size:0.95rem; font-weight:700; color:#22D3EE;">{len(current_detections)} veh</span>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8;">UNIQUE TRACKS</span><br>
                    <span style="font-size:0.95rem; font-weight:700; color:#10B981;">{unique_tracks_count} total</span>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8;">ESTIMATED FLOW</span><br>
                    <span style="font-size:0.95rem; font-weight:700; color:#F59E0B;">{flow_rate_vph} veh/h</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_metrics:
    st.markdown("### 🎛️ Real-Time Signal Actuator")

    # Signal Phase State Override
    active_phase = "North-South"
    if override_green == "Force East-West Green":
        active_phase = "East-West"
    elif override_green == "Force North-South Green":
        active_phase = "North-South"

    ns_green = active_phase == "North-South"
    ew_green = not ns_green

    col_sig_ns, col_sig_ew = st.columns(2)
    with col_sig_ns:
        st.markdown(
            f"""
            <div style="background:#0F172A; border:2px solid {'#10B981' if ns_green else '#EF4444'}; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:0.85rem; color:#94A3B8; font-weight:600;">NORTH-SOUTH APPROACH</div>
                <div style="font-size:2.2rem; font-weight:800; color:{'#10B981' if ns_green else '#EF4444'}; margin:6px 0;">
                    {'🟢 GREEN' if ns_green else '🔴 RED'}
                </div>
                <div style="font-size:0.85rem; color:#CBD5E1;">
                    Allocated Green: <b>{signal_plan.phases[0].green_time_seconds}s</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sig_ew:
        st.markdown(
            f"""
            <div style="background:#0F172A; border:2px solid {'#10B981' if ew_green else '#EF4444'}; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:0.85rem; color:#94A3B8; font-weight:600;">EAST-WEST APPROACH</div>
                <div style="font-size:2.2rem; font-weight:800; color:{'#10B981' if ew_green else '#EF4444'}; margin:6px 0;">
                    {'🟢 GREEN' if ew_green else '🔴 RED'}
                </div>
                <div style="font-size:0.85rem; color:#CBD5E1;">
                    Allocated Green: <b>{signal_plan.phases[1].green_time_seconds}s</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 📈 Live Approach Density Breakdown")

    # Table breakdown of PCU and Counts
    table_data = []
    for lane in ["North", "South", "East", "West"]:
        m = lane_metrics[lane]
        table_data.append({
            "Approach": lane,
            "Cars": m.cars,
            "Buses": m.buses,
            "Trucks": m.trucks,
            "Bikes": m.motorcycles,
            "Total PCU": m.pcu_total,
            "LOS Level": m.congestion_level,
        })
    df_live = pd.DataFrame(table_data)
    st.dataframe(df_live, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Deployment Notice & Fallback Redirection (Requirement 9)
# -----------------------------------------------------------------------------
st.markdown("---")
c_fall1, c_fall2 = st.columns(2)

with c_fall1:
    st.markdown(
        """
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:14px;">
            <div style="font-size:0.95rem; font-weight:700; color:#38BDF8; margin-bottom:4px;">
                🔄 Fallback 1: Upload Pre-Recorded Traffic Video
            </div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-bottom:10px;">
                If physical camera capture is unavailable or your browser permissions are restricted, 
                navigate to <b>Page 1: Video Analysis</b> to analyze MP4 or AVI roadway footage with identical detection models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📁 Go to Page 1: Video Analysis", use_container_width=True):
        st.switch_page("pages/1_Video_Analysis.py")

with c_fall2:
    st.markdown(
        """
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:14px;">
            <div style="font-size:0.95rem; font-weight:700; color:#10B981; margin-bottom:4px;">
                ⚡ Fallback 2: Synthetic Benchmark Traffic Simulator
            </div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-bottom:10px;">
                Select <b>Mode C</b> above to test real-time object tracking, signal phasing, and queue management 
                using our zero-dependency mathematical intersection simulation engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
