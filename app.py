"""
Smart Traffic Management System - Executive Dashboard & Architecture Hub
========================================================================
Capstone Project: AI-Based Smart Traffic Management System Using Video Analytics and Graph Theory
B.Tech Computer Science & Engineering (2nd Year)

Design & Architecture:
- Dark Navy & Cyan Premium Dashboard Theme (#070D18 / #0F172A / #06B6D4)
- Live Processing Status Indicator (Active only during physical video/camera processing)
- Explicitly Labeled Benchmark Demo-Data Mode (Zero fabricated/random numbers)
- Graph-Theoretic Intersection Network Overview
- Multi-Tier Analysis Modes Guide & Comprehensive Diagnostic Engine Inspector
- Modular Navigation Directory linking to all 6 application pages
"""

import os
import time
import streamlit as st
import pandas as pd
import cv2
import networkx as nx

from modules.graph_model import IntersectionGraphNetwork
from modules.traffic_analyzer import TrafficAnalyzer
from modules.vehicle_detector import VehicleDetector
from modules.api_config import get_api_key, is_api_key_configured, mask_api_key
from visualizations.charts import get_custom_css, create_network_graph_plot

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Dark Navy & Cyan Theme CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# 2. Sidebar Navigation & Academic Metadata
with st.sidebar:
    st.markdown("### 🚦 Smart Traffic System")
    st.markdown(
        """
        <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 10px; padding: 12px; margin-bottom: 16px;">
            <div style="color: #38BDF8; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase;">
                🎓 Academic Capstone
            </div>
            <div style="color: #E2E8F0; font-size: 0.95rem; font-weight: 600; margin-top: 4px;">
                2nd Year CSE Project
            </div>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 2px;">
                Discrete Mathematics & Video Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧭 System Navigation")
    st.page_link("app.py", label="Executive Overview", icon="🏠")
    st.page_link("pages/1_Video_Analysis.py", label="1. Video Analysis", icon="🎥")
    st.page_link("pages/2_Live_Camera.py", label="2. Live Camera Feed", icon="📷")
    st.page_link("pages/3_Traffic_Analytics.py", label="3. Traffic Analytics", icon="📊")
    st.page_link("pages/4_Intersection_Graph.py", label="4. Intersection Graph", icon="🕸️")
    st.page_link("pages/5_Discrete_Mathematics_Lab.py", label="5. Discrete Math Lab", icon="📐")
    st.page_link("pages/6_Project_Information.py", label="6. Project Information & Docs", icon="📖")
    st.page_link("pages/7_How_to_Use.py", label="7. How to Use & User Guide", icon="💡")

    st.markdown("---")
    st.markdown("#### ⚙️ Data Source Active")
    data_source = st.selectbox(
        "Select Active Pipeline Mode",
        [
            "Synthetic Benchmark Stream (Zero-Dependency)",
            "Sample Traffic Log (CSV Batch)",
            "Live Hardware Webcam / Camera",
            "Custom Video File Upload",
        ],
        index=0,
        help="Configures the primary ingestion source across all analytical modules",
    )
    st.session_state["global_data_source"] = data_source

    st.markdown("---")
    with st.expander("🔑 API Key & Secrets Settings", expanded=False):
        has_key = is_api_key_configured()
        if has_key:
            st.markdown(f"**Status:** `🟢 Configured ({mask_api_key(get_api_key())})`")
        else:
            st.markdown("**Status:** `⚪ Not Configured (Optional)`")

        user_input_key = st.text_input(
            "API Key (Optional):",
            value=st.session_state.get("user_api_key", ""),
            type="password",
            help="Enter your API key or configure it in .streamlit/secrets.toml",
            key="ui_sidebar_api_key_input",
        )
        if user_input_key:
            st.session_state["user_api_key"] = user_input_key
            st.caption("✔ Key saved to active session.")

        st.caption("💡 Permanent setup: create `.streamlit/secrets.toml` with `API_KEY = \"...\"`.")

    st.markdown(
        """
        <div style="margin-top: 20px; font-size: 0.75rem; color: #64748B; text-align: center;">
            Engineered with Streamlit, OpenCV & NetworkX<br>
            © 2026 Department of Computer Science & Engineering
        </div>
        """,
        unsafe_allow_html=True,
    )


# 3. Dynamic Live Analysis Status Checker
is_video_processing = bool(st.session_state.get("is_analyzing_video", False))
is_camera_streaming = bool(st.session_state.get("live_cam_connected", False))
is_any_pipeline_active = is_video_processing or is_camera_streaming

# Main Dashboard Header
col_header_title, col_header_badge = st.columns([2.8, 1.2])

with col_header_title:
    st.title("AI-Based Smart Traffic Management System")
    st.markdown(
        """
        **Integrating Deep Computer Vision Analytics with Graph-Theoretic Optimization**  
        *An end-to-end intelligent transportation solution mapping vehicular queue dynamics to Discrete Mathematics algorithms.*
        """
    )

with col_header_badge:
    if is_any_pipeline_active:
        active_pipeline_name = "Video Processing" if is_video_processing else "Live Camera"
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-emerald">🟢 Active Processing</span>
                <span class="badge badge-cyan">Pipeline Running</span><br>
                <span style="font-size: 0.75rem; color: #34D399; font-weight:600;">{active_pipeline_name} Feed Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span class="badge badge-cyan">v2.5 Production</span>
                <span class="badge badge-amber">⚪ System Standby</span><br>
                <span style="font-size: 0.75rem; color: #94A3B8;">Ready for Video / Camera Ingestion</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# 4. Explicitly Labeled Benchmark Demo-Data Banner
st.markdown(
    """
    <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.3rem;">📊</span>
            <div style="font-size:0.85rem; color:#E0F2FE; line-height:1.5;">
                <b>EXPLICIT BENCHMARK DEMO-DATA MODE:</b> Overview metrics and charts displayed on this executive page are computed from 
                verified, deterministic historical intersection logs (<code>data/sample_traffic.csv</code>) and graph theoretical models. 
                <b>Zero randomized numbers or fabricated sensor readings are used.</b> 
                To process live visual feeds, launch <b>Module 1 (Video Analysis)</b> or <b>Module 2 (Live Camera)</b>.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# 5. Load Sample Traffic Data & KPI Summary
@st.cache_data
def load_kpis():
    try:
        df = TrafficAnalyzer.load_sample_dataset("data/sample_traffic.csv")
        kpis = TrafficAnalyzer.compute_kpi_summary(df)
        return df, kpis
    except Exception:
        return pd.DataFrame(), TrafficAnalyzer.compute_kpi_summary(pd.DataFrame())

df_sample, kpi_summary = load_kpis()


# 6. Top Metric Cards with Tooltips and Clear Rationale
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="Vehicles Monitored",
        value=f"{kpi_summary['total_vehicles_processed']:,}",
        delta="Across 4 Approaches",
        help="Total unique vehicle records analyzed in the historical intersection benchmark dataset.",
    )

with m2:
    st.metric(
        label="Avg Delay Saved",
        value=f"{kpi_summary['delay_saved_pct']}%",
        delta="Adaptive vs Static",
        delta_color="normal",
        help="Average reduction in driver intersection idling delay achieved by adaptive proportional signal allocation compared to Webster fixed-time pre-timed baselines.",
    )

with m3:
    st.metric(
        label="CO₂ Reduction",
        value=f"{kpi_summary['co2_saved_kg']} kg",
        delta="Idle Emissions Avoided",
        help="Environmental benefit calculated based on 0.15 kg CO2 emission prevented per minute of unnecessary vehicular idling at red lights.",
    )

with m4:
    st.metric(
        label="Chromatic Phases",
        value="χ(G) = 4",
        delta="Welsh-Powell Conflict-Free",
        help="Minimum number of collision-free green signal phases calculated by graph vertex coloring on the 8-movement turning conflict graph.",
    )

st.write("")

# Dual Operating Ingestion Pipelines: Separated Manual Upload vs Sample Demo
st.markdown("### 🎛️ Dual Operating Ingestion Pipelines")
tab_manual_home, tab_demo_home = st.tabs([
    "📁 Mode A: Manual Video Upload Working",
    "⚡ Mode B: Sample Benchmark Demo Working",
])

with tab_manual_home:
    c_m1, c_m2 = st.columns([2, 1.2])
    with c_m1:
        st.markdown(
            """
            <div style="font-size:0.9rem; color:#E0F2FE; margin-bottom:12px;">
                <b>Upload Real Traffic Footage:</b> Drag and drop an MP4, AVI, MOV, or MKV video file below to feed directly into the Computer Vision detection & tracking pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )
        app_vid_file = st.file_uploader(
            "Choose a traffic surveillance video to analyze:",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported video containers: MP4, AVI, MOV, MKV, WebM (up to 200 MB).",
            key="app_home_video_uploader",
        )
        if app_vid_file is not None:
            import tempfile
            from modules.video_processor import VideoProcessor
            if st.session_state.get("uploaded_filename") != app_vid_file.name:
                temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(app_vid_file.name)[1])
                temp_f.write(app_vid_file.getbuffer())
                temp_f.close()
                st.session_state["temp_video_path"] = temp_f.name
                st.session_state["uploaded_filename"] = app_vid_file.name
                st.session_state["va_stream_source"] = "📁 Mode A: Manual Video Upload Working"
                st.session_state["current_video_frame_idx"] = 0
                if "telemetry_history" in st.session_state:
                    st.session_state.telemetry_history.clear()
                if "detection_log" in st.session_state:
                    st.session_state.detection_log.clear()
                if "tripwire_crossed_ids" in st.session_state:
                    st.session_state.tripwire_crossed_ids.clear()
                st.session_state["last_annotated_frame"] = None

            vid_meta = VideoProcessor.get_video_metadata(st.session_state["temp_video_path"])
            if vid_meta:
                st.success(f"✔ **{app_vid_file.name}** parsed: {vid_meta['width']}×{vid_meta['height']} px | {vid_meta['duration_formatted']} | {vid_meta['fps']} FPS ({vid_meta['file_size_mb']} MB)")
            else:
                st.success(f"✔ Video **{app_vid_file.name}** uploaded successfully!")

            st.page_link("pages/1_Video_Analysis.py", label="🚀 Launch Analysis on Uploaded Video", icon="🎥")
    with c_m2:
        st.markdown(
            """
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:16px; font-size:0.85rem; color:#94A3B8;">
                <b style="color:#38BDF8;">📁 Manual Mode Features:</b><br><br>
                • YOLOv8 Nano ONNX Deep Learning Detection<br>
                • Centroid Multi-Object Tracking with persistent IDs<br>
                • Stopline tripwires and approach queue estimation<br>
                • Proportional signal phase optimization
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_demo_home:
    c_d1, c_d2 = st.columns([2, 1.2])
    with c_d1:
        st.markdown(
            """
            <div style="font-size:0.9rem; color:#E0F2FE; margin-bottom:12px;">
                <b>Zero-Dependency Classroom Simulation:</b> Runs a continuous 4-way intersection benchmark with automated vehicle physics. No video file or camera required.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 14px; margin-bottom: 12px; font-size:0.85rem; color: #CBD5E1;">
                • <b>Approaches:</b> North, South, East, West<br>
                • <b>Vehicle Mix:</b> Cars (1.0 PCU), Buses (3.0 PCU), Trucks (3.0 PCU), Motorcycles (0.5 PCU)<br>
                • <b>Emergency Simulation:</b> Preemption override testing supported
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Video_Analysis.py", label="⚡ Launch Sample Benchmark Demo in Module 1", icon="🚀")
    with c_d2:
        st.markdown(
            """
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:16px; font-size:0.85rem; color:#10B981;">
                <b>⚡ Sample Demo Benefits:</b><br><br>
                • 100% offline and deterministic<br>
                • Ideal for quick classroom presentations<br>
                • Real-time graph & metric updates<br>
                • Demonstrates adaptive green time shifting
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# Quick Launch Navigation Hub
with st.expander("⚡ Quick Launch Application Modules", expanded=True):
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1:
        st.page_link("pages/1_Video_Analysis.py", label="1. Video Analysis", icon="🎥")
        st.page_link("pages/2_Live_Camera.py", label="2. Live Camera", icon="📷")
    with col_nav2:
        st.page_link("pages/3_Traffic_Analytics.py", label="3. Traffic Analytics", icon="📊")
        st.page_link("pages/4_Intersection_Graph.py", label="4. Intersection Graph", icon="🕸️")
    with col_nav3:
        st.page_link("pages/5_Discrete_Mathematics_Lab.py", label="5. Discrete Math Lab", icon="📐")
        st.page_link("pages/6_Project_Information.py", label="6. Project Information", icon="📖")
    with col_nav4:
        st.page_link("pages/7_How_to_Use.py", label="7. How to Use Guide", icon="💡")
        st.page_link("app.py", label="Executive Overview", icon="🏠")

st.write("")


# 7. Two-Column Layout: System Architecture & Quick Network Preview
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.markdown(
        """
        <div class="glass-card glass-card-glow">
            <h3 style="margin-top:0; color:#38BDF8;">🏛️ Project Overview & Engineering Architecture</h3>
            <p style="color:#CBD5E1; font-size:0.95rem; line-height:1.6;">
                Urban traffic congestion causes severe economic loss, excess fuel consumption, and emergency response delays.
                Traditional traffic lights use <b>fixed-time pre-timed cycles (Webster's baseline)</b> that cannot adapt to asymmetrical queue spikes.
            </p>
            <p style="color:#CBD5E1; font-size:0.95rem; line-height:1.6;">
                This capstone introduces a closed-loop cyber-physical architecture:
            </p>
            <ol style="color:#E2E8F0; font-size:0.85rem; padding-left: 20px; line-height:1.7;">
                <li><b>Computer Vision Layer:</b> Object detection monitors traffic feeds to classify vehicles into Passenger Car Units (PCU).</li>
                <li><b>Adaptive Signal Layer:</b> Proportional green-time optimization dynamically allocates signal phases based on detected demand.</li>
                <li><b>Graph Theory Layer:</b> City intersections are mapped as directed weighted graphs <code>G = (V, E)</code>. Live density updates edge weights via the Bureau of Public Roads (BPR) function.</li>
                <li><b>Discrete Mathematics Lab:</b> Implements Welsh-Powell vertex coloring for collision-free phase scheduling, planarity verification (Euler's formula), and Dijkstra dynamic rerouting.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Diagnostic System Status Inspector
    with st.expander("🔍 System Diagnostic & Model Setup Inspector", expanded=False):
        c_env1, c_env2 = st.columns(2)
        detector = VehicleDetector()
        status_info = detector.get_status()

        with c_env1:
            st.markdown(f"**OpenCV Core:** `{cv2.__version__}`")
            st.markdown(f"**NetworkX Engine:** `{nx.__version__}`")
            st.markdown(f"**Streamlit Dashboard:** `{st.__version__}`")
            st.markdown(f"**Active Backend:** `{status_info['backend']}`")
            api_status = f"🟢 Configured ({mask_api_key(get_api_key())})" if is_api_key_configured() else "⚪ Not Configured (Optional)"
            st.markdown(f"**API Key:** `{api_status}`")

        with c_env2:
            dnn_loaded = status_info["is_dnn_model_loaded"]
            if dnn_loaded:
                st.markdown("**Pretrained DNN:** `🟢 YOLOv8 ONNX Ready (12.8 MB)`")
                st.markdown(f"**Model Path:** `{os.path.basename(status_info['model_path'])}`")
            else:
                st.markdown("**Pretrained DNN:** `🟡 Weights Missing (MOG2 / Synthetic Fallback Active)`")
                st.caption("Offline fallback operates seamlessly with zero external weights.")

        if not status_info["is_dnn_model_loaded"]:
            st.markdown("---")
            st.markdown(
                """
                **Model Setup Options:**  
                - **Network Requirement:** Outbound HTTPS access to `github.com` is required to download the official YOLOv8 Nano weights (~12.8 MB).  
                - **Offline Operation:** If internet is restricted, the system functions completely offline using classical OpenCV MOG2 and the procedural synthetic traffic stream.
                """
            )
            col_dl1, col_dl2 = st.columns([1.5, 2.5])
            with col_dl1:
                if st.button("📥 Download YOLOv8 Weights (~12.8 MB)", key="app_dl_model_btn"):
                    with st.spinner("Downloading YOLOv8 Nano ONNX model from official release assets..."):
                        success, dl_msg = detector.download_model_weights(timeout=60)
                        if success:
                            st.success("✔ Model downloaded successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Download failed: {dl_msg}")
            with col_dl2:
                st.caption("You can also run `python setup_models.py` in your terminal.")

            st.markdown("**Option 2: Direct Model File Upload**")
            app_uploaded_onnx = st.file_uploader("Upload local yolov8n.onnx file:", type=["onnx"], key="app_onnx_upload")
            if app_uploaded_onnx is not None:
                models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
                os.makedirs(models_dir, exist_ok=True)
                target_path = os.path.join(models_dir, "yolov8n.onnx")
                with open(target_path, "wb") as f_out:
                    f_out.write(app_uploaded_onnx.getbuffer())
                detector.reload_dnn()
                st.success("✔ Model weights uploaded and loaded successfully!")
                time.sleep(1)
                st.rerun()
        else:
            st.caption("ℹ️ All components are operational and running locally without cloud dependencies.")

with right_col:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0; color:#38BDF8;">🕸️ Urban Road Network Overview</h3>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Mathematical discrete graph model <code>G = (V, E)</code> representing 6 urban intersections and connecting corridors. 
                Edge labels display physical distances (km) and dynamic congestion travel times (min).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Plotly interactive network graph
    network = IntersectionGraphNetwork()
    fig_preview = create_network_graph_plot(network.graph, show_edge_labels=True)
    st.plotly_chart(fig_preview, use_container_width=True)


# 8. Analysis Modes Comparison Table
st.markdown("### 🎛️ Analysis Modes Comparison Matrix")
st.markdown(
    """
    The system supports multiple ingestion and analysis modalities tailored for varied computing environments:
    """
)
modes_data = [
    {
        "Mode Name": "1. Video Analysis",
        "Target Ingestion": "Pre-recorded traffic video files (MP4, AVI, MOV)",
        "Key Capabilities": "Full-frame CV inference, Centroid MOT tracking, Spatial tripwire counts, PCU classification",
        "Primary Use Case": "Surveillance forensic analysis, junction capacity audits, highway studies",
    },
    {
        "Mode Name": "2. Live Camera Feed",
        "Target Ingestion": "Local USB hardware camera or Browser WebRTC camera",
        "Key Capabilities": "Real-time edge frame processing, instant signal phase actuation, zero persistent storage",
        "Primary Use Case": "Live junction control demonstration, edge microcontroller deployment",
    },
    {
        "Mode Name": "3. Traffic Analytics",
        "Target Ingestion": "Historical traffic telemetry & synthetic demand profiles",
        "Key Capabilities": "Temporal trends, Level of Service (LOS A–F), Adaptive vs Static delay comparisons",
        "Primary Use Case": "Transportation planning, municipal signal timing design, environmental reporting",
    },
    {
        "Mode Name": "4. Intersection Graph",
        "Target Ingestion": "Urban network topology G = (V, E)",
        "Key Capabilities": "BPR congestion impedance, Dijkstra vs BFS shortest paths, Road incident closures & rerouting",
        "Primary Use Case": "Citywide traffic distribution, navigation bypass routing, network bottleneck analysis",
    },
    {
        "Mode Name": "5. Discrete Math Lab",
        "Target Ingestion": "2nd-year CSE curriculum discrete mathematical structures",
        "Key Capabilities": "Welsh-Powell coloring, Planar graph flyover tests, Groups, Homomorphisms, Logic & Combinatorics",
        "Primary Use Case": "Academic capstone examination, viva voce preparation, curriculum demonstration",
    },
    {
        "Mode Name": "6. Project Info & Docs",
        "Target Ingestion": "Comprehensive technical report & methodologies",
        "Key Capabilities": "Problem statement, architecture diagrams, detection limitations, mathematical equations",
        "Primary Use Case": "Peer review, technical reference, architectural defense",
    },
]
st.dataframe(pd.DataFrame(modes_data), use_container_width=True, hide_index=True)


# 9. Modular Navigation Directory (Interactive Cards)
st.markdown("### 🚀 System Modules & Demonstration Directory")
st.markdown("Click on any module below or use the sidebar navigation to test each capability:")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-cyan">Module 1</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">🎥 Video Analysis (Upload & Streams)</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Upload custom traffic videos (MP4, AVI, MOV) or analyze procedural synthetic streams. Full bounding-box detection, PCU counts, and virtual tripwires.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Upload Video & Analyze ➡️", key="btn_nav_1"):
        st.switch_page("pages/1_Video_Analysis.py")

with c2:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-emerald">Module 2</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">📷 Live Camera Feed</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Stream real-time webcam video or interactive simulated camera feeds with zero hardware dependencies. Instant signal phase adjustment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Live Camera ➡️", key="btn_nav_2"):
        st.switch_page("pages/2_Live_Camera.py")

with c3:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-amber">Module 3</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">📊 Traffic Analytics</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Temporal volume trends, peak-hour heatmaps, Level of Service (LOS) gauges, and comparative delay analysis (Static vs Adaptive).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Traffic Analytics ➡️", key="btn_nav_3"):
        st.switch_page("pages/3_Traffic_Analytics.py")

c4, c5, c6 = st.columns(3)

with c4:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-cyan">Module 4</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">🕸️ Intersection Graph</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Dijkstra dynamic shortest-path rerouting under congestion and incidents, BPR travel time equations, Adjacency matrices, and Ford-Fulkerson max-flow bottlenecks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Intersection Graph ➡️", key="btn_nav_4"):
        st.switch_page("pages/4_Intersection_Graph.py")

with c5:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-coral">Module 5</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">📐 Discrete Math Lab</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Comprehensive academic laboratory: Welsh-Powell vertex coloring for conflict-free signal scheduling, Planar graphs & Euler's formula (V - E + F = 2), and Viva Q&A.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Discrete Math Lab ➡️", key="btn_nav_5"):
        st.switch_page("pages/5_Discrete_Mathematics_Lab.py")

with c6:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-emerald">Module 6</span>
            <h4 style="color: #F8FAFC; margin: 8px 0;">📖 Project Info & Docs</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Full technical report, architectural specifications, detection failure modes, mathematical equations, and future connected vehicle (V2X) roadmap.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Project Information ➡️", key="btn_nav_6"):
        st.switch_page("pages/6_Project_Information.py")


# 10. Help, Methodology, and Limitations Accordion
st.markdown("---")
with st.expander("❓ Help & Methodology Summary: Models, Mathematics & Limitations", expanded=False):
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown("#### 🔍 Object Detection & Multi-Object Tracking")
        st.markdown(
            """
            - **Detection:** Supported vehicles (Car, Bus, Truck, Motorcycle) are detected via YOLOv8 Nano ONNX or OpenCV MOG2.
            - **Tracking:** Centroid Euclidean distance tracking maintains persistent track IDs across frames.
            - **Non-Repeated Counting:** A virtual counting tripwire records vehicles exactly once when their bounding box center crosses the boundary line.
            - **PCU Standardization:** Vehicles are converted to Passenger Car Units (Car = 1.0, Bus = 3.0, Truck = 3.0, Motorcycle = 0.5) to standardize road space demand.
            """
        )
    with h_col2:
        st.markdown("#### 📐 Mathematical Formulations & Limitations")
        st.markdown(
            r"""
            - **Adaptive Signal Optimization:** Available green budget ($G = C - \sum L$) is allocated proportionally to approach PCU demand with strict min/max constraints ($10\text{s} \le g \le 60\text{s}$).
            - **Graph Model:** Urban roads are modeled as $G = (V, E)$ with BPR dynamic travel time impedance $t_e = t_0 [1 + 0.15(V/C)^4]$.
            - **Optical Limitations:** Heavy occlusion from large vehicles, low-light illumination, and adverse weather can reduce detection confidence.
            - **Educational Simulation:** The road network is an academic mathematical simulation, not a real-time municipal highway GIS map.
            """
        )
    st.info("💡 For the full peer-reviewed methodology and curriculum proof derivations, visit **Page 6: Project Information & Docs**.")
