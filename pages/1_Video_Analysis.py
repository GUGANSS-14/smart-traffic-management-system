"""
Page 1: Video Analysis & Vehicle Analytics
Comprehensive computer-vision pipeline for traffic surveillance streams and uploaded video files.

Key Features:
- Video ingestion (MP4, AVI, MOV, MKV) with metadata extraction (duration, resolution, FPS).
- Object detection (YOLOv8 Nano ONNX, Classical MOG2, Procedural Simulation) with HUD annotations.
- Multi-Object Tracking (MOT) with persistent track IDs and non-repeated counting.
- Configurable Region of Interest (ROI) and Virtual Counting Line (Tripwire).
- Congestion index calculation and Level-of-Service (LOS) classification.
- Measured vs Estimated telemetry separation for academic transparency.
- Time-series vehicle class trends, approach lane demand splits, and queue estimations.
- Interactive analysis log table with 1-click CSV export.
- Safe OpenCV resource release and non-persistent video disk cleanup.
"""

import os
import time
import tempfile
from typing import Dict, List, Tuple, Any, Optional, Set

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
import plotly.express as px

from modules.vehicle_detector import VehicleDetector, Detection, ModelNotLoadedError
from modules.video_processor import (
    VideoProcessor,
    SyntheticTrafficStream,
    UnsupportedVideoFormatError,
    CorruptedVideoError,
)
from modules.traffic_analyzer import TrafficAnalyzer, LaneMetrics
from modules.signal_optimizer import SignalOptimizer
from modules.report_generator import ReportGenerator, TrafficReportData
from visualizations.charts import (
    get_custom_css,
    create_lane_distribution_donut,
    create_queue_length_bar,
    DARK_LAYOUT_TEMPLATE,
)

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Video Analysis - Smart Traffic Management",
    page_icon="🎥",
    layout="wide",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "synthetic_stream" not in st.session_state:
    st.session_state.synthetic_stream = SyntheticTrafficStream(width=720, height=540)
if "detector" not in st.session_state:
    st.session_state.detector = VehicleDetector(backend="Pretrained DNN / YOLO", conf_threshold=0.45)
if "analyzer" not in st.session_state:
    st.session_state.analyzer = TrafficAnalyzer()
if "optimizer" not in st.session_state:
    st.session_state.optimizer = SignalOptimizer()

# Telemetry History & State
if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = []  # Frame-level summaries
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []      # Vehicle-level detection records
if "tripwire_crossed_ids" not in st.session_state:
    st.session_state.tripwire_crossed_ids = set()
if "crossed_events" not in st.session_state:
    st.session_state.crossed_events = {}
if "current_video_frame_idx" not in st.session_state:
    st.session_state.current_video_frame_idx = 0
if "last_annotated_frame" not in st.session_state:
    st.session_state.last_annotated_frame = None
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

SOURCE_UPLOAD = "📁 Mode A: Manual Video Upload Working"
SOURCE_SYNTHETIC = "⚡ Mode B: Sample Benchmark Demo Working"

if "va_stream_source" not in st.session_state or st.session_state.va_stream_source not in [SOURCE_UPLOAD, SOURCE_SYNTHETIC]:
    if st.session_state.get("global_data_source") == "Custom Video File Upload" or st.session_state.get("temp_video_path"):
        st.session_state.va_stream_source = SOURCE_UPLOAD
    else:
        st.session_state.va_stream_source = SOURCE_UPLOAD

detector: VehicleDetector = st.session_state.detector
analyzer: TrafficAnalyzer = st.session_state.analyzer
optimizer: SignalOptimizer = st.session_state.optimizer


# -----------------------------------------------------------------------------
# Helper Functions: Video Inspection & Resource Management
# -----------------------------------------------------------------------------
# Reusable helper methods delegated to VideoProcessor to eliminate duplicate logic
get_video_metadata = VideoProcessor.get_video_metadata
check_tripwire_crossing = VideoProcessor.check_tripwire_crossing
is_inside_roi = VideoProcessor.is_inside_roi


def cleanup_temp_video():
    """Removes temporary uploaded video files from disk to prevent storage accumulation."""
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        try:
            os.remove(st.session_state.temp_video_path)
        except Exception:
            pass
        st.session_state.temp_video_path = None


# -----------------------------------------------------------------------------
# Header with Dynamic Live Analysis Indicator
# -----------------------------------------------------------------------------
col_head_title, col_head_badge = st.columns([2.8, 1.2])

with col_head_title:
    st.title("🎥 Traffic Video Analytics & Vehicle Detection")
    st.markdown(
        """
        **Autonomous Computer Vision Pipeline, Multi-Object Tracking & Traffic Engineering Analytics**  
        *Processes video streams using pretrained convolutional networks (YOLOv8 ONNX), assigns persistent track IDs, 
        evaluates spatial density across regions of interest, and generates adaptive signal plans.*
        """
    )

with col_head_badge:
    if st.session_state.get("is_analyzing_video", False):
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-emerald">🟢 Active Ingestion</span><br>
                <span style="font-size: 0.75rem; color: #34D399; font-weight:600;">Processing Video Frame</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-amber">⚪ Ready / Standby</span><br>
                <span style="font-size: 0.75rem; color: #94A3B8;">Awaiting Next Step / Stream</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# -----------------------------------------------------------------------------
# Sidebar: Analytics & Model Controls
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

    st.markdown("### 🎛️ Active Pipeline Mode")
    sidebar_source = st.radio(
        "Select Pipeline Mode",
        [SOURCE_UPLOAD, SOURCE_SYNTHETIC],
        index=0 if st.session_state.va_stream_source == SOURCE_UPLOAD else 1,
        key="va_sidebar_source_radio",
    )
    if sidebar_source != st.session_state.va_stream_source:
        st.session_state.va_stream_source = sidebar_source
        st.rerun()

    video_source_type = st.session_state.va_stream_source

    detection_engine = st.selectbox(
        "Detection Engine Backend",
        [
            "Pretrained DNN / YOLO (Real Deep Learning)",
            "OpenCV MOG2 (Classical Background Subtraction)",
            "Procedural Simulation (Ground Truth Engine)",
        ],
        index=0 if "Upload" in video_source_type else 2,
    )

    if "Pretrained DNN" in detection_engine and not detector.get_status()["is_dnn_model_loaded"]:
        st.warning("⚠️ YOLOv8 ONNX weights not found in `models/`.")
        st.caption("Requires outbound HTTPS to `github.com` (~12.8 MB).")
        if st.button("📥 Download YOLOv8 Weights (~12.8 MB)", key="va_dl_model_btn"):
            with st.spinner("Downloading YOLOv8 Nano weights..."):
                success, dl_msg = detector.download_model_weights(timeout=60)
                if success:
                    st.success("✔ Model downloaded!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Download failed: {dl_msg}")

        uploaded_onnx = st.file_uploader("Or Upload yolov8n.onnx Directly:", type=["onnx"], key="va_onnx_upload")
        if uploaded_onnx is not None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
            os.makedirs(models_dir, exist_ok=True)
            target_path = os.path.join(models_dir, "yolov8n.onnx")
            with open(target_path, "wb") as f_out:
                f_out.write(uploaded_onnx.getbuffer())
            detector.reload_dnn()
            st.success("✔ Model weights uploaded and loaded successfully!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Inference Parameters")
    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.95, value=0.45, step=0.05)
    inf_res = st.selectbox("Inference Resolution", ["640x640 (Standard High-Accuracy)", "416x416 (Fast CPU)"], index=0)
    input_size = (640, 640) if "640" in inf_res else (416, 416)
    frame_sampling_interval = st.slider("Frame Sampling Interval (CPU Saver)", min_value=1, max_value=5, value=1, help="Process every k-th frame to reduce CPU load on high-FPS video.")

    st.markdown("---")
    st.markdown("### 🎯 Spatial Zones & Counting Lines")
    spatial_mode = st.radio(
        "Spatial Counting Mode",
        [
            "Full Intersection (All Approaches)",
            "Virtual Counting Line (Tripwire)",
            "Region of Interest (ROI Box)",
        ],
        index=0,
    )

    tripwire_lane = "North Approach Stopline"
    tripwire_pct = 50
    roi_preset = "North Approach Corridor"

    if spatial_mode == "Virtual Counting Line (Tripwire)":
        tripwire_lane = st.selectbox(
            "Tripwire Target Corridor",
            [
                "North Approach Stopline",
                "South Approach Stopline",
                "East Approach Stopline",
                "West Approach Stopline",
                "Mid-Intersection Horizontal",
            ],
            index=0,
        )
        tripwire_pct = st.slider("Tripwire Position (% of Frame)", min_value=15, max_value=85, value=50, step=5)
    elif spatial_mode == "Region of Interest (ROI Box)":
        roi_preset = st.selectbox(
            "ROI Target Zone",
            [
                "North Approach Corridor",
                "South Approach Corridor",
                "East Approach Corridor",
                "West Approach Corridor",
                "Central Conflict Box",
            ],
            index=0,
        )

    st.markdown("---")
    st.markdown("### 📐 Congestion Metric Settings")
    capacity_threshold_pcu = st.slider(
        "Approach Capacity Threshold (PCU)",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=5.0,
        help="Civil engineering saturation flow threshold used to calculate Congestion Index: (PCU / Capacity) * 100%.",
    )

    st.markdown("---")
    st.markdown("### 👁️ Overlays & Trajectories")
    enable_tracking = st.checkbox("Persistent Multi-Object Tracking", value=True)
    show_lanes = st.checkbox("Overlay Lane Guidelines & Approaches", value=True)
    show_labels = st.checkbox("Display Bounding Box Labels & IDs", value=True)
    show_trails = st.checkbox("Display Motion Trajectory Trails", value=True)

    st.markdown("---")
    st.markdown("### 🚨 Emergency Simulation")
    emergency_lane = st.selectbox("Simulate Emergency Vehicle In", ["None", "North", "South", "East", "West"], index=0)
    emergency_param = None if emergency_lane == "None" else emergency_lane


# Apply configurations to detector
detector.set_backend(detection_engine)
detector.set_conf_threshold(conf_thresh)
detector.set_input_size(input_size)
detector.enable_tracking = enable_tracking
analyzer.lane_capacity_pcu = capacity_threshold_pcu


# -----------------------------------------------------------------------------
# Video Upload & Metadata Inspection
# -----------------------------------------------------------------------------
uploaded_file = None
video_metadata: Optional[Dict[str, Any]] = None

st.markdown("### 🎛️ Dual Operating Ingestion Pipelines")
col_src_view1, col_src_view2 = st.columns([2.5, 1.5])
with col_src_view1:
    main_source = st.radio(
        "Select Pipeline Operating Mode:",
        [SOURCE_UPLOAD, SOURCE_SYNTHETIC],
        horizontal=True,
        index=0 if st.session_state.va_stream_source == SOURCE_UPLOAD else 1,
        key="va_main_source_radio",
    )
    if main_source != st.session_state.va_stream_source:
        st.session_state.va_stream_source = main_source
        st.rerun()
with col_src_view2:
    if st.session_state.va_stream_source == SOURCE_UPLOAD:
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; color: #38BDF8;">
                <b>📁 Manual Upload Active:</b> Analyze custom MP4/AVI surveillance footage with YOLOv8/MOG2.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; color: #34D399;">
                <b>⚡ Sample Demo Active:</b> Continuous ground-truth 4-way intersection simulation. Zero files needed.
            </div>
            """,
            unsafe_allow_html=True,
        )

if st.session_state.va_stream_source == SOURCE_UPLOAD:
    st.markdown("#### 📁 Manual Video Upload Workspace")
    uploaded_file = st.file_uploader(
        "Choose an MP4, AVI, MOV, MKV video file to analyze:",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload pre-recorded roadway footage. Supported containers: MP4, AVI, MOV, MKV (up to 200 MB).",
        key="traffic_video_file_uploader",
    )

    if uploaded_file is not None:
        # Check if new file uploaded; clean previous temp file if necessary
        if st.session_state.uploaded_filename != uploaded_file.name:
            cleanup_temp_video()
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.current_video_frame_idx = 0
            st.session_state.telemetry_history.clear()
            st.session_state.detection_log.clear()
            st.session_state.tripwire_crossed_ids.clear()
            st.session_state.last_annotated_frame = None
            detector.reset_tracker()

            # Save uploaded bytes to a local temporary file for OpenCV streaming
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            st.session_state.temp_video_path = temp_file.name

    has_valid_video = bool(st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path))
    if has_valid_video:
        video_metadata = get_video_metadata(st.session_state.temp_video_path)
        if video_metadata:
            # Metadata Banner Card
            st.markdown(
                f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px 20px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                        <div>
                            <span style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Uploaded Video Source</span>
                            <div style="font-size:1.15rem; font-weight:700; color:#38BDF8;">{st.session_state.uploaded_filename or 'Uploaded Video'}</div>
                        </div>
                        <div style="display:flex; gap:20px; flex-wrap:wrap;">
                            <div>
                                <span style="font-size:0.75rem; color:#94A3B8;">Resolution</span>
                                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">{video_metadata['width']} × {video_metadata['height']} px</div>
                            </div>
                            <div>
                                <span style="font-size:0.75rem; color:#94A3B8;">Duration</span>
                                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">{video_metadata['duration_formatted']}</div>
                            </div>
                            <div>
                                <span style="font-size:0.75rem; color:#94A3B8;">Frame Rate</span>
                                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">{video_metadata['fps']} FPS ({video_metadata['total_frames']} frames)</div>
                            </div>
                            <div>
                                <span style="font-size:0.75rem; color:#94A3B8;">File Size</span>
                                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">{video_metadata['file_size_mb']} MB</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("⚠️ Failed to parse uploaded video file. The file may be corrupt or lacks compatible OpenCV codecs.")
    else:
        st.markdown(
            """
            <div style="background: rgba(6, 182, 212, 0.05); border: 2px dashed rgba(6, 182, 212, 0.35); border-radius: 12px; padding: 22px; text-align: center; margin-bottom: 16px;">
                <span style="font-size: 2.2rem;">🎬</span>
                <div style="font-size: 1.05rem; font-weight: 600; color: #38BDF8; margin-top: 6px;">
                    Awaiting Video File Upload
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px; max-width: 520px; margin-left: auto; margin-right: auto;">
                    Select or drag an MP4, AVI, MOV, or MKV file into the uploader above to begin deep computer-vision vehicle detection and queue analysis.
                </div>
                <div style="margin-top: 10px;">
                    <span style="font-size: 0.8rem; color: #64748B;">
                        💡 No local video file? Toggle to <b>Synthetic Benchmark Stream</b> above for an instant interactive simulation!
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    # Synthetic Stream Metadata Banner
    st.markdown("#### ⚡ Sample Benchmark Intersection Workspace")
    st.markdown(
        """
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px 20px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <span style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Active Video Stream Source</span>
                    <div style="font-size:1.15rem; font-weight:700; color:#10B981;">Synthetic Benchmark Intersection Stream (Ground Truth Engine)</div>
                </div>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div>
                        <span style="font-size:0.75rem; color:#94A3B8;">Resolution</span>
                        <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">720 × 540 px</div>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:#94A3B8;">Frame Rate</span>
                        <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">30.0 FPS (Continuous Physics)</div>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:#94A3B8;">Kinematics</span>
                        <div style="font-size:0.95rem; font-weight:600; color:#38BDF8;">4 Approaches | Signal-Compliant</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Spatial Overlay Geometries (Tripwire & ROI Box Calculation)
# -----------------------------------------------------------------------------
# Coordinate mapping for standard 720x540 canvas
CANVAS_W, CANVAS_H = 720, 540

active_tripwire_line = None
active_tripwire_text = None
active_roi_box = None

if spatial_mode == "Virtual Counting Line (Tripwire)":
    y_coord = int((tripwire_pct / 100.0) * CANVAS_H)
    if tripwire_lane == "North Approach Stopline":
        active_tripwire_line = ((CANVAS_W // 2 - 120, y_coord), (CANVAS_W // 2, y_coord))
    elif tripwire_lane == "South Approach Stopline":
        active_tripwire_line = ((CANVAS_W // 2, y_coord), (CANVAS_W // 2 + 120, y_coord))
    elif tripwire_lane == "East Approach Stopline":
        active_tripwire_line = ((CANVAS_W // 2 + 60, y_coord), (CANVAS_W - 40, y_coord))
    elif tripwire_lane == "West Approach Stopline":
        active_tripwire_line = ((40, y_coord), (CANVAS_W // 2 - 60, y_coord))
    else:  # Mid-Intersection Horizontal
        active_tripwire_line = ((60, y_coord), (CANVAS_W - 60, y_coord))

    cross_count = len(st.session_state.tripwire_crossed_ids)
    active_tripwire_text = f"TRIPWIRE: {tripwire_lane} [{cross_count} crossed]"

elif spatial_mode == "Region of Interest (ROI Box)":
    if roi_preset == "North Approach Corridor":
        active_roi_box = (CANVAS_W // 2 - 90, 0, CANVAS_W // 2, CANVAS_H // 2 - 40)
    elif roi_preset == "South Approach Corridor":
        active_roi_box = (CANVAS_W // 2, CANVAS_H // 2 + 40, CANVAS_W // 2 + 90, CANVAS_H)
    elif roi_preset == "East Approach Corridor":
        active_roi_box = (CANVAS_W // 2 + 40, CANVAS_H // 2 - 90, CANVAS_W, CANVAS_H // 2)
    elif roi_preset == "West Approach Corridor":
        active_roi_box = (0, CANVAS_H // 2, CANVAS_W // 2 - 40, CANVAS_H // 2 + 90)
    else:  # Central Conflict Box
        r = min(CANVAS_W, CANVAS_H) // 7
        active_roi_box = (CANVAS_W // 2 - r, CANVAS_H // 2 - r, CANVAS_W // 2 + r, CANVAS_H // 2 + r)

else:  # Full Intersection (All Approaches)
    # Default mid-intersection crossing cordon line at 50% height
    active_tripwire_line = ((40, CANVAS_H // 2), (CANVAS_W - 40, CANVAS_H // 2))
    cross_count = len(st.session_state.tripwire_crossed_ids)
    active_tripwire_text = f"INTERSECTION CORDON [{cross_count} crossed]"


# -----------------------------------------------------------------------------
# Execution Controls: Process, Batch & Reset
# -----------------------------------------------------------------------------
is_synth = (st.session_state.va_stream_source == SOURCE_SYNTHETIC)
lbl_1 = "▶️ Step 1 Simulation Step" if is_synth else "▶️ Step 1 Video Frame"
lbl_10 = "⚡ Run 10 Simulation Steps" if is_synth else "⚡ Run 10 Video Frames"
lbl_30 = "⏩ Run 30 Simulation Steps" if is_synth else "⏩ Run 30 Video Frames"
lbl_rst = "🔄 Reset Simulation" if is_synth else "🔄 Reset Video & Clear Logs"

c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1, 1, 1.2, 1.2])

with c_btn1:
    btn_step_1 = st.button(lbl_1, use_container_width=True)
with c_btn2:
    btn_run_10 = st.button(lbl_10, use_container_width=True)
with c_btn3:
    btn_run_30 = st.button(lbl_30, use_container_width=True)
with c_btn4:
    btn_reset = st.button(lbl_rst, use_container_width=True)

if btn_reset:
    detector.reset_tracker()
    st.session_state.telemetry_history.clear()
    st.session_state.detection_log.clear()
    st.session_state.tripwire_crossed_ids.clear()
    st.session_state.crossed_events.clear()
    st.session_state.current_video_frame_idx = 0
    st.session_state.last_annotated_frame = None
    st.session_state.last_detections = []
    st.rerun()


# -----------------------------------------------------------------------------
# Frame Processing Engine
# -----------------------------------------------------------------------------
is_synthetic_mode = (st.session_state.va_stream_source == SOURCE_SYNTHETIC)
has_valid_video = bool(st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path))

frames_to_advance = 0
if btn_step_1:
    frames_to_advance = 1
elif btn_run_10:
    frames_to_advance = 10
elif btn_run_30:
    frames_to_advance = 30
elif st.session_state.last_annotated_frame is None:
    # Initial cold start frame only if stream source is ready
    if is_synthetic_mode or has_valid_video:
        frames_to_advance = 1

if not is_synthetic_mode and not has_valid_video:
    if st.session_state.last_annotated_frame is None:
        placeholder_frame = np.full((CANVAS_H, CANVAS_W, 3), (15, 23, 42), dtype=np.uint8)
        cv2.putText(
            placeholder_frame,
            "AWAITING VIDEO UPLOAD",
            (CANVAS_W // 2 - 190, CANVAS_H // 2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (6, 182, 212),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder_frame,
            "Choose an MP4/AVI file above or switch to Synthetic Stream",
            (CANVAS_W // 2 - 270, CANVAS_H // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
        st.session_state.last_annotated_frame = placeholder_frame

progress_bar = st.empty()
status_message = st.empty()

if frames_to_advance > 0:
    if not is_synthetic_mode and not has_valid_video:
        status_message.warning("⚠️ Please select or upload a video file above before advancing frames.")
    elif is_synthetic_mode:
        stream: SyntheticTrafficStream = st.session_state.synthetic_stream
        for step_idx in range(frames_to_advance):
            stream.step()
            st.session_state.current_video_frame_idx += 1
            cur_idx = st.session_state.current_video_frame_idx
            cur_ts = round(cur_idx / 30.0, 2)

            raw_frame = stream.render_frame()

            # Execute detection
            try:
                detections = detector.detect(raw_frame, synthetic_stream=stream, timestamp=cur_ts)
            except ModelNotLoadedError as err:
                st.error(f"⚠️ Pretrained Model Weights Error: {err}")
                st.info("💡 Model weights file not found. Please select 'Procedural Simulation' or 'OpenCV MOG2' in the sidebar.")
                detections = []

            # Check spatial filters: Tripwire and ROI
            vehicles_in_roi = 0
            for det in detections:
                in_roi = is_inside_roi(det.center, active_roi_box)
                det_crossed = False
                if active_tripwire_line is not None:
                    det_crossed = check_tripwire_crossing(
                        det.center,
                        active_tripwire_line[0],
                        active_tripwire_line[1],
                        det.track_id,
                        st.session_state.tripwire_crossed_ids,
                    )
                if in_roi:
                    vehicles_in_roi += 1

                is_verified_crossed = (det_crossed or (det.track_id is not None and det.track_id in st.session_state.tripwire_crossed_ids))
                if is_verified_crossed and det.track_id is not None:
                    if det.track_id not in st.session_state.crossed_events:
                        st.session_state.crossed_events[det.track_id] = {
                            "order_idx": len(st.session_state.crossed_events) + 1,
                            "track_id": det.track_id,
                            "class_name": det.class_name.capitalize(),
                            "timestamp_s": cur_ts,
                            "frame_index": cur_idx,
                            "approach": det.lane_id,
                            "pcu_weight": det.pcu_weight,
                            "confidence_pct": round(det.confidence * 100, 1),
                            "tripwire_name": active_tripwire_text or "Intersection Cordon",
                        }

                # Append detailed vehicle record (keep last 500 events)
                if det.track_id is not None:
                    st.session_state.detection_log.append({
                        "Timestamp (s)": cur_ts,
                        "Frame Index": cur_idx,
                        "Track ID": f"#{det.track_id}",
                        "Class": det.class_name.capitalize(),
                        "Confidence": f"{int(det.confidence * 100)}%",
                        "Approach": det.lane_id,
                        "PCU Weight": det.pcu_weight,
                        "In ROI": "Yes" if in_roi else "No",
                        "Crossed Line": "Yes" if is_verified_crossed else "No",
                    })

            # Tabulate class counts
            class_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
            for d in detections:
                c_key = d.class_name.lower()
                if c_key in class_counts:
                    class_counts[c_key] += 1
                else:
                    class_counts["car"] += 1

            pcu_frame = sum(d.pcu_weight for d in detections)
            unique_tracks = detector.tracker.get_unique_vehicle_count()
            flow_vph = detector.tracker.compute_traffic_flow_rate(elapsed_seconds=cur_ts)

            # Record frame summary in telemetry history
            st.session_state.telemetry_history.append({
                "timestamp": cur_ts,
                "frame": cur_idx,
                "car": class_counts["car"],
                "bus": class_counts["bus"],
                "truck": class_counts["truck"],
                "motorcycle": class_counts["motorcycle"],
                "total_visible": len(detections),
                "unique_tracks": unique_tracks,
                "flow_vph": flow_vph,
                "pcu_total": round(pcu_frame, 1),
                "roi_vehicles": vehicles_in_roi,
                "tripwire_crossings": len(st.session_state.tripwire_crossed_ids),
            })

            # Annotate HUD frame
            banner_text = (
                f"Visible: {len(detections)} | Unique Tracks: {unique_tracks} | "
                f"Flow: {int(flow_vph)} veh/h | Backend: {detector.backend}"
            )
            annotated_frame = detector.draw_detections(
                raw_frame,
                detections,
                draw_lanes=show_lanes,
                show_labels=show_labels,
                show_tracks=enable_tracking,
                show_trails=show_trails,
                tracker=detector.tracker,
                header_text=banner_text,
                roi_box=active_roi_box,
                tripwire_line=active_tripwire_line,
                tripwire_text=active_tripwire_text,
            )

            st.session_state.last_annotated_frame = annotated_frame
            st.session_state.last_detections = detections

    else:
        # Uploaded Video Processing
        video_path = st.session_state.temp_video_path
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1000)

            # Seek to current frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_video_frame_idx)

            try:
                for step_idx in range(frames_to_advance):
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        status_message.success("✅ End of video reached. All frames processed successfully.")
                        break

                    st.session_state.current_video_frame_idx += 1
                    cur_idx = st.session_state.current_video_frame_idx

                    # Frame sampling interval check
                    if cur_idx % frame_sampling_interval != 0:
                        continue

                    cur_ts = round(cur_idx / fps, 2) if fps > 0 else 0.0
                    frame_resized = cv2.resize(raw_frame, (CANVAS_W, CANVAS_H))

                    try:
                        detections = detector.detect(frame_resized, timestamp=cur_ts)
                    except ModelNotLoadedError as err:
                        st.error(f"⚠️ Pretrained Model Weights Error: {err}")
                        st.info("💡 To run real vehicle inference, ensure 'models/yolov8n.onnx' exists, or select 'Procedural Simulation' demo mode.")
                        detections = []

                    # Spatial filter checks
                    vehicles_in_roi = 0
                    for det in detections:
                        in_roi = is_inside_roi(det.center, active_roi_box)
                        det_crossed = False
                        if active_tripwire_line is not None:
                            det_crossed = check_tripwire_crossing(
                                det.center,
                                active_tripwire_line[0],
                                active_tripwire_line[1],
                                det.track_id,
                                st.session_state.tripwire_crossed_ids,
                            )
                        if in_roi:
                            vehicles_in_roi += 1

                        is_verified_crossed = (det_crossed or (det.track_id is not None and det.track_id in st.session_state.tripwire_crossed_ids))
                        if is_verified_crossed and det.track_id is not None:
                            if det.track_id not in st.session_state.crossed_events:
                                st.session_state.crossed_events[det.track_id] = {
                                    "order_idx": len(st.session_state.crossed_events) + 1,
                                    "track_id": det.track_id,
                                    "class_name": det.class_name.capitalize(),
                                    "timestamp_s": cur_ts,
                                    "frame_index": cur_idx,
                                    "approach": det.lane_id,
                                    "pcu_weight": det.pcu_weight,
                                    "confidence_pct": round(det.confidence * 100, 1),
                                    "tripwire_name": active_tripwire_text or "Intersection Cordon",
                                }

                        if det.track_id is not None:
                            st.session_state.detection_log.append({
                                "Timestamp (s)": cur_ts,
                                "Frame Index": cur_idx,
                                "Track ID": f"#{det.track_id}",
                                "Class": det.class_name.capitalize(),
                                "Confidence": f"{int(det.confidence * 100)}%",
                                "Approach": det.lane_id,
                                "PCU Weight": det.pcu_weight,
                                "In ROI": "Yes" if in_roi else "No",
                                "Crossed Line": "Yes" if is_verified_crossed else "No",
                            })

                    class_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
                    for d in detections:
                        c_key = d.class_name.lower()
                        if c_key in class_counts:
                            class_counts[c_key] += 1
                        else:
                            class_counts["car"] += 1

                    pcu_frame = sum(d.pcu_weight for d in detections)
                    unique_tracks = detector.tracker.get_unique_vehicle_count()
                    flow_vph = detector.tracker.compute_traffic_flow_rate(elapsed_seconds=cur_ts)

                    st.session_state.telemetry_history.append({
                        "timestamp": cur_ts,
                        "frame": cur_idx,
                        "car": class_counts["car"],
                        "bus": class_counts["bus"],
                        "truck": class_counts["truck"],
                        "motorcycle": class_counts["motorcycle"],
                        "total_visible": len(detections),
                        "unique_tracks": unique_tracks,
                        "flow_vph": flow_vph,
                        "pcu_total": round(pcu_frame, 1),
                        "roi_vehicles": vehicles_in_roi,
                        "tripwire_crossings": len(st.session_state.tripwire_crossed_ids),
                    })

                    banner_text = (
                        f"Visible: {len(detections)} | Unique Tracks: {unique_tracks} | "
                        f"Flow: {int(flow_vph)} veh/h | Backend: {detector.backend}"
                    )
                    annotated_frame = detector.draw_detections(
                        frame_resized,
                        detections,
                        draw_lanes=show_lanes,
                        show_labels=show_labels,
                        show_tracks=enable_tracking,
                        show_trails=show_trails,
                        tracker=detector.tracker,
                        header_text=banner_text,
                        roi_box=active_roi_box,
                        tripwire_line=active_tripwire_line,
                        tripwire_text=active_tripwire_text,
                    )

                    st.session_state.last_annotated_frame = annotated_frame
                    st.session_state.last_detections = detections

                    # Update progress bar
                    if total_frames > 0:
                        pct = min(1.0, cur_idx / total_frames)
                        progress_bar.progress(pct, text=f"Processing Video: Frame {cur_idx} of {total_frames} ({int(pct*100)}%)")
            finally:
                cap.release()  # Mandatory OpenCV resource release


# Ensure fallback blank frame if uninitialized
if st.session_state.last_annotated_frame is None:
    st.session_state.last_annotated_frame = np.full((CANVAS_H, CANVAS_W, 3), (25, 30, 40), dtype=np.uint8)

current_frame_detections = st.session_state.last_detections
current_lane_metrics = analyzer.analyze_frame_detections(current_frame_detections)
current_signal_plan = optimizer.compute_adaptive_plan(current_lane_metrics, emergency_lane=emergency_param)


# -----------------------------------------------------------------------------
# Main Visual Layout: Video Player & Real-Time Telemetry
# -----------------------------------------------------------------------------
col_video, col_telemetry = st.columns([1.35, 1.0])

with col_video:
    st.markdown("### 🖥️ Optical Surveillance HUD")
    frame_rgb = cv2.cvtColor(st.session_state.last_annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", use_container_width=True)

    # Frame Metadata Footer Badge
    vis_count = len(current_frame_detections)
    unique_cnt = detector.tracker.get_unique_vehicle_count()
    flow_val = int(detector.tracker.compute_traffic_flow_rate())
    tw_cross_cnt = len(st.session_state.tripwire_crossed_ids)

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#0F172A; padding:10px 16px; border-radius:10px; border:1px solid #1E293B; flex-wrap:wrap; gap:10px; margin-top:8px;">
            <div><span style="color:#94A3B8; font-size:0.75rem;">BACKEND</span><br><b style="color:#38BDF8; font-size:0.85rem;">{detector.backend}</b></div>
            <div><span style="color:#94A3B8; font-size:0.75rem;">VISIBLE NOW</span><br><b style="color:#22D3EE; font-size:0.95rem;">{vis_count} veh</b></div>
            <div><span style="color:#94A3B8; font-size:0.75rem;">UNIQUE TRACKS</span><br><b style="color:#10B981; font-size:0.95rem;">{unique_cnt} total</b></div>
            <div><span style="color:#94A3B8; font-size:0.75rem;">TRIPWIRE CROSSED</span><br><b style="color:#F59E0B; font-size:0.95rem;">{tw_cross_cnt} veh</b></div>
            <div><span style="color:#94A3B8; font-size:0.75rem;">ESTIMATED FLOW</span><br><b style="color:#E2E8F0; font-size:0.95rem;">{flow_val} veh/h</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_telemetry:
    st.markdown("### 📊 Approach Lane Telemetry & LOS")

    # 4-Approach Lane Telemetry Cards
    kpi_cols = st.columns(4)
    for idx, lane in enumerate(["North", "South", "East", "West"]):
        m = current_lane_metrics[lane]
        with kpi_cols[idx]:
            status_color = "#10B981" if m.congestion_level in ["Free Flow", "Low"] else ("#F59E0B" if m.congestion_level in ["Moderate", "Medium"] else "#EF4444")
            st.markdown(
                f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:10px 6px; text-align:center;">
                    <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">{lane}</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#38BDF8; margin:2px 0;">{m.total_vehicles}</div>
                    <div style="font-size:0.75rem; color:#CBD5E1;">{m.pcu_total} PCU</div>
                    <div style="font-size:0.7rem; color:{status_color}; font-weight:700; margin-top:2px;">
                        {m.congestion_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Adaptive Signal Recommendation Display
    st.markdown("### 🚦 Adaptive AI Signal Recommendation")
    p1 = current_signal_plan.phases[0]
    p2 = current_signal_plan.phases[1]

    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        st.markdown(
            f"""
            <div style="background:#0F172A; border:1px solid {'#EF4444' if p1.priority_level=='Emergency Override' else '#06B6D4'}; border-radius:10px; padding:12px;">
                <div style="font-size:0.75rem; color:#38BDF8; font-weight:600;">{p1.phase_id}</div>
                <div style="font-size:1.85rem; font-weight:800; color:#F8FAFC; margin:2px 0;">
                    {p1.green_time_seconds}s <span style="font-size:0.85rem; color:#10B981;">GREEN</span>
                </div>
                <div style="font-size:0.72rem; color:#94A3B8;">Controlled: {', '.join(p1.controlled_lanes)}</div>
                <div style="font-size:0.72rem; color:#E2E8F0; margin-top:4px;">Priority: <b>{p1.priority_level}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sp_col2:
        st.markdown(
            f"""
            <div style="background:#0F172A; border:1px solid {'#EF4444' if p2.priority_level=='Emergency Override' else '#1E293B'}; border-radius:10px; padding:12px;">
                <div style="font-size:0.75rem; color:#38BDF8; font-weight:600;">{p2.phase_id}</div>
                <div style="font-size:1.85rem; font-weight:800; color:#F8FAFC; margin:2px 0;">
                    {p2.green_time_seconds}s <span style="font-size:0.85rem; color:#10B981;">GREEN</span>
                </div>
                <div style="font-size:0.72rem; color:#94A3B8;">Controlled: {', '.join(p2.controlled_lanes)}</div>
                <div style="font-size:0.72rem; color:#E2E8F0; margin-top:4px;">Priority: <b>{p2.priority_level}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="margin-top:10px; font-size:0.85rem; color:#94A3B8; background:#0B1120; padding:8px 12px; border-radius:8px; border:1px solid #1E293B;">
            Predicted Delay Reduction vs Fixed Baseline: 
            <b style="color:#10B981; font-size:1.05rem;">+{current_signal_plan.delay_reduction_pct}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Measured vs Estimated Values Separation (Requirement)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔬 Academic Verification: Measured Empirical Data vs Engineered Estimations")

c_meas, c_est = st.columns(2)

with c_meas:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid rgba(6, 182, 212, 0.4); border-radius:12px; padding:16px;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:1.2rem;">🔬</span>
                <span style="font-size:1rem; font-weight:700; color:#22D3EE;">MEASURED EMPIRICAL VALUES (Direct CV Observation)</span>
            </div>
            <ul style="color:#CBD5E1; font-size:0.88rem; line-height:1.7; margin-bottom:0;">
                <li><b>Instantaneous Vehicle Count:</b> <span style="color:#38BDF8;">{len(current_frame_detections)}</span> vehicles visible in active frame.</li>
                <li><b>Cumulative Unique Track IDs:</b> <span style="color:#10B981;">{detector.tracker.get_unique_vehicle_count()}</span> distinct physical entities observed.</li>
                <li><b>Virtual Tripwire Crossings:</b> <span style="color:#F59E0B;">{len(st.session_state.tripwire_crossed_ids)}</span> verified line intersections.</li>
                <li><b>Observed Class Composition:</b> Cars, Buses, Trucks, and Motorcycles localized by bounding boxes.</li>
                <li><b>Tracking Trajectory Breadcrumbs:</b> 2D spatial motion vectors tracked across consecutive frames.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_est:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid rgba(245, 158, 11, 0.4); border-radius:12px; padding:16px;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:1.2rem;">📐</span>
                <span style="font-size:1rem; font-weight:700; color:#FBBF24;">ENGINEERED ESTIMATED VALUES (Civil Mathematical Models)</span>
            </div>
            <ul style="color:#CBD5E1; font-size:0.88rem; line-height:1.7; margin-bottom:0;">
                <li><b>Passenger Car Units (PCU):</b> Standard IRC weighting (Car: 1.0, Bus: 3.0, Truck: 3.0, Bike: 0.5).</li>
                <li><b>Estimated Traffic Throughput:</b> <span style="color:#F59E0B;">{int(detector.tracker.compute_traffic_flow_rate())} veh/h</span> derived via <code>q = (ΔN / Δt) * 3600</code>.</li>
                <li><b>Congestion Index (V/C Ratio):</b> Evaluated against approach capacity threshold ({capacity_threshold_pcu} PCU).</li>
                <li><b>Queue Length Projections:</b> Physical vehicle length heuristics + 1.5m safe stopping gap.</li>
                <li><b>Adaptive Green Splits:</b> Proportional Webster phase allocation algorithm.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Temporal Analytics: Time-Series Trends & Vehicle Class Comparison
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Time-Series Trends & Vehicle Fleet Comparison")

if len(st.session_state.telemetry_history) > 1:
    hist_df = pd.DataFrame(st.session_state.telemetry_history)

    c_plot1, c_plot2 = st.columns(2)

    with c_plot1:
        # Time-Series Line Chart
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["car"], mode="lines+markers", name="Cars", line=dict(color="#22D3EE", width=2)))
        fig_ts.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["bus"], mode="lines+markers", name="Buses", line=dict(color="#F59E0B", width=2)))
        fig_ts.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["truck"], mode="lines+markers", name="Trucks", line=dict(color="#EF4444", width=2)))
        fig_ts.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["motorcycle"], mode="lines+markers", name="Motorcycles", line=dict(color="#10B981", width=2)))

        fig_ts.update_layout(
            **DARK_LAYOUT_TEMPLATE,
            title="<b>Vehicle Class Counts Over Elapsed Time</b>",
            xaxis_title="Elapsed Video Time (seconds)",
            yaxis_title="Vehicles Detected",
            height=340,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with c_plot2:
        # Cumulative Class Breakdown Donut Chart
        class_totals = {
            "Cars": hist_df["car"].iloc[-1] if not hist_df.empty else 0,
            "Buses": hist_df["bus"].iloc[-1] if not hist_df.empty else 0,
            "Trucks": hist_df["truck"].iloc[-1] if not hist_df.empty else 0,
            "Motorcycles": hist_df["motorcycle"].iloc[-1] if not hist_df.empty else 0,
        }
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=list(class_totals.keys()),
                    values=list(class_totals.values()),
                    hole=0.55,
                    marker=dict(colors=["#22D3EE", "#F59E0B", "#EF4444", "#10B981"], line=dict(color="#0B1120", width=2)),
                    textinfo="label+percent",
                )
            ]
        )
        fig_donut.update_layout(
            **DARK_LAYOUT_TEMPLATE,
            title="<b>Observed Vehicle Fleet Composition (%)</b>",
            height=340,
        )
        st.plotly_chart(fig_donut, use_container_width=True)
else:
    st.info("ℹ️ Run at least 2 frames using the controls above to plot time-series trend curves.")


# -----------------------------------------------------------------------------
# Video Traffic Intelligence Report & Multi-Sheet Excel Export
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Downloadable Video Traffic Intelligence Report & Excel Workbook")
st.markdown(
    """
    *Generate and download an official multi-sheet Microsoft Excel (`.xlsx`) inspection workbook 
    or Executive Traffic Engineering Markdown summary (`.md`) based on this video's telemetry.*
    """
)

fps_val = float(st.session_state.get("uploaded_fps", 30.0)) if st.session_state.get("uploaded_fps") else 30.0
video_title = st.session_state.uploaded_filename or ("synthetic_benchmark_stream.mp4" if is_synth else "uploaded_traffic_video.mp4")
dur_s = round(st.session_state.current_video_frame_idx / fps_val, 2) if fps_val > 0 else 0.0

rep_data = TrafficReportData(
    video_name=video_title,
    total_frames_processed=st.session_state.current_video_frame_idx,
    video_duration_s=dur_s,
    fps=fps_val,
    resolution="720x540",
    source_mode=st.session_state.va_stream_source,
    backend_model=detector.backend,
    total_unique_vehicles=detector.tracker.get_unique_vehicle_count(),
    total_vehicles_crossed=len(st.session_state.crossed_events),
    crossed_events=list(st.session_state.crossed_events.values()),
    detection_log=st.session_state.detection_log,
    telemetry_history=st.session_state.telemetry_history,
    lane_metrics=current_lane_metrics,
    signal_plan_data={
        "cycle_time_seconds": current_signal_plan.cycle_time_seconds,
        "delay_reduction_pct": current_signal_plan.delay_reduction_pct,
        "phases": [
            {
                "phase_id": p.phase_id,
                "green_time_seconds": p.green_time_seconds,
                "controlled_lanes": p.controlled_lanes,
                "priority_level": p.priority_level,
            }
            for p in current_signal_plan.phases
        ],
    },
)

crossed_summary = ReportGenerator.extract_crossed_summary(rep_data.crossed_events)
crossed_df = ReportGenerator.build_crossed_vehicles_dataframe(rep_data.crossed_events)
crossed_ids = {ev.get("track_id") for ev in rep_data.crossed_events if ev.get("track_id") is not None}
crossed_map = {ev.get("track_id"): ev for ev in rep_data.crossed_events if ev.get("track_id") is not None}
inventory_df = ReportGenerator.build_vehicle_inventory_dataframe(st.session_state.detection_log, crossed_ids, crossed_map)

# KPI Cards for Crossing & Fleet
rep_kpi1, rep_kpi2, rep_kpi3, rep_kpi4 = st.columns(4)
with rep_kpi1:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid #065F46; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#34D399; font-weight:700;">TOTAL VEHICLES CROSSED</div>
            <div style="font-size:1.75rem; font-weight:800; color:#10B981; margin:2px 0;">{crossed_summary['total_crossed']}</div>
            <div style="font-size:0.72rem; color:#94A3B8;">Verified physical crossings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rep_kpi2:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#38BDF8; font-weight:700;">UNIQUE VEHICLES TRACKED</div>
            <div style="font-size:1.75rem; font-weight:800; color:#F8FAFC; margin:2px 0;">{detector.tracker.get_unique_vehicle_count()}</div>
            <div style="font-size:0.72rem; color:#94A3B8;">Hungarian persistent tracks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rep_kpi3:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#F59E0B; font-weight:700;">TOTAL CROSSED PCU LOAD</div>
            <div style="font-size:1.75rem; font-weight:800; color:#FBBF24; margin:2px 0;">{crossed_summary['total_pcu_crossed']}</div>
            <div style="font-size:0.72rem; color:#94A3B8;">IRC standard passenger units</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rep_kpi4:
    st.markdown(
        f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#A78BFA; font-weight:700;">CROSSED FLEET BREAKDOWN</div>
            <div style="font-size:0.85rem; font-weight:700; color:#E2E8F0; margin-top:6px;">
                🚗 {crossed_summary['by_class']['Car']} | 🚌 {crossed_summary['by_class']['Bus']}<br>
                🚚 {crossed_summary['by_class']['Truck']} | 🏍️ {crossed_summary['by_class']['Motorcycle']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Action Buttons: Excel (.xlsx), Markdown (.md), and CSV
col_dl_excel, col_dl_md, col_dl_csv = st.columns([1.4, 1.3, 1.1])

with col_dl_excel:
    excel_bytes = ReportGenerator.generate_excel_traffic_report(rep_data)
    st.download_button(
        label="📗 Download Complete Excel Report (.xlsx)",
        data=excel_bytes,
        file_name=f"traffic_report_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("Includes 5 sheets: Executive Summary, Crossed Vehicles Log, Vehicle Inventory, Telemetry, and Signal Plan.")

with col_dl_md:
    md_report = ReportGenerator.generate_markdown_traffic_report(rep_data)
    st.download_button(
        label="📄 Download Executive Summary (.md)",
        data=md_report,
        file_name=f"executive_traffic_report_{int(time.time())}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.caption("Executive engineering summary with metric tables and recommendations.")

with col_dl_csv:
    if st.session_state.detection_log:
        csv_data = pd.DataFrame(st.session_state.detection_log).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Raw CSV Log (.csv)",
            data=csv_data,
            file_name=f"traffic_raw_log_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Flat frame-by-frame raw event log.")
    else:
        st.caption("Raw CSV log will appear after stepping frames.")

# Interactive Tabs Previewing Data in Streamlit
st.write("")
tab_crossed, tab_inventory, tab_raw_log = st.tabs([
    f"🏁 Crossed Vehicles Log ({len(crossed_df)} Crossed)",
    f"🚘 Complete Vehicle Inventory ({len(inventory_df)} Unique)",
    f"📋 Raw Detection Event Stream ({len(st.session_state.detection_log)} Records)",
])

with tab_crossed:
    if not crossed_df.empty:
        st.dataframe(crossed_df, use_container_width=True, height=280)
    else:
        st.info("💡 No vehicles have crossed the cordon line yet. Step frames or run simulation above to register crossings!")

with tab_inventory:
    if not inventory_df.empty:
        st.dataframe(inventory_df, use_container_width=True, height=280)
    else:
        st.info("💡 No vehicle tracking data recorded yet. Advance frames above to populate.")

with tab_raw_log:
    if st.session_state.detection_log:
        st.dataframe(pd.DataFrame(st.session_state.detection_log).tail(200), use_container_width=True, height=280)
    else:
        st.info("💡 Advance frames above to generate raw event records.")


# -----------------------------------------------------------------------------
# Academic Theory & Metric Definitions Expander
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📚 Educational & Viva Guide: Mathematical Formulations & Metric Distinctions", expanded=False):
    explanations = detector.explain_traffic_metrics()
    st.markdown(
        f"""
        ### 1. Mathematical Distinction of Core Flow Metrics
        - **Unique Tracks ($|T_{{\\text{{unique}}}}|$):**  
          {explanations['unique_tracks']}
          
        - **Instantaneous Spatial Density ($N(t)$):**  
          {explanations['vehicles_visible_in_frame']}
          
        - **Estimated Traffic Flow Rate ($q$):**  
          {explanations['estimated_traffic_flow']}

        ---

        ### 2. Congestion Index Formulation (Volume-to-Capacity $V/C$)
        Civil and municipal transportation authorities evaluate roadway Level of Service (LOS) using the **Bureau of Public Roads (BPR)** and **Highway Capacity Manual (HCM)** standard:

        $$\\text{{Congestion Index (\\%)}} = \\min\\left(100\\%, \\frac{{\\sum_{{i}} \\text{{PCU}}_i}}{{\\text{{Capacity}}_{{\\text{{approach}}}}}} \\times 100\\%\\right)$$

        Where:
        - $\\text{{Car}} = 1.0\\text{{ PCU}}$
        - $\\text{{Bus}} = 3.0\\text{{ PCU}}$
        - $\\text{{Truck}} = 3.0\\text{{ PCU}}$
        - $\\text{{Motorcycle}} = 0.5\\text{{ PCU}}$
        - Default Lane Saturation Capacity = **{capacity_threshold_pcu} PCU**
        """
    )
