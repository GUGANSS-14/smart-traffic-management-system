"""
Page 7: How to Use & User Operating Manual
==========================================
Comprehensive interactive documentation and workflow guide for the
Smart Traffic Management System.

Covers:
1. Quick-Start Ingestion Decision Matrix (Sample Demo vs Manual Upload)
2. Workflow 1: Sample Benchmark Demo Working (Zero-Dependency)
3. Workflow 2: Manual Video Upload Working (Real Surveillance Ingestion)
4. Workflow 3: Live Hardware Camera & WebRTC Streaming
5. Workflow 4: Macro Traffic Analytics & Adaptive Signal Optimization
6. Workflow 5: Intersection Graph Theory & Dynamic BPR Rerouting
7. Workflow 6: Discrete Mathematics Syllabus Laboratory (Units I–V)
8. API Key & Security Configuration Guide
9. Troubleshooting, Codec Specifications & FAQ
"""

import streamlit as st
from visualizations.charts import get_custom_css

# Page Configuration
st.set_page_config(
    page_title="How to Use - Smart Traffic Management",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header Section
col_head, col_badge = st.columns([2.8, 1.2])
with col_head:
    st.title("💡 How to Use & System Operating Manual")
    st.markdown(
        """
        **Step-by-Step Operator Guide, Ingestion Workflows & Technical Reference**  
        *Everything you need to know to run live classroom demos, upload surveillance footage, optimize signal cycles, and explore graph theory.*
        """
    )
with col_badge:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px;">
            <span class="badge badge-cyan">User Manual v2.5</span><br>
            <span style="font-size: 0.75rem; color: #94A3B8;">Comprehensive Guide</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🚦 Smart Traffic System")
    with st.expander("🧭 System Navigation", expanded=True):
        st.page_link("app.py", label="Executive Overview", icon="🏠")
        st.page_link("pages/1_Video_Analysis.py", label="1. Video Analysis", icon="🎥")
        st.page_link("pages/2_Live_Camera.py", label="2. Live Camera Feed", icon="📷")
        st.page_link("pages/3_Traffic_Analytics.py", label="3. Traffic Analytics", icon="📊")
        st.page_link("pages/4_Intersection_Graph.py", label="4. Intersection Graph", icon="🕸️")
        st.page_link("pages/5_Discrete_Mathematics_Lab.py", label="5. Discrete Math Lab", icon="📐")
        st.page_link("pages/6_Project_Information.py", label="6. Project Information & Docs", icon="📖")
        st.page_link("pages/7_How_to_Use.py", label="7. How to Use & User Guide", icon="💡")

    st.markdown("---")
    st.info("💡 **Quick Tip:** For an instant presentation without uploading files, choose **Sample Demo Mode** in Module 1.")

# Top Decision Matrix Callout
st.markdown(
    """
    <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.35); border-radius: 12px; padding: 18px 22px; margin-bottom: 24px;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
            🎯 Core Operating Modes: Choose Your Path
        </div>
        <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.6;">
            The application provides two completely separated ingestion workflows depending on your objective:
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px;">
            <div style="background: #0B1120; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <b style="color: #10B981; font-size: 0.95rem;">⚡ 1. Sample Demo Mode (Zero-Dependency)</b><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Instant classroom demonstration. Runs an automated 4-way intersection physics simulation with continuous kinematic vehicle spawning. 
                    <b>No files, camera permissions, or model weights required.</b>
                </span>
            </div>
            <div style="background: #0B1120; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <b style="color: #38BDF8; font-size: 0.95rem;">📁 2. Manual Video Upload Mode (Forensic CV)</b><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">
                    Upload your own MP4, AVI, MOV, or MKV surveillance recording. The deep learning pipeline runs object detection, 
                    assigns tracking IDs, evaluates tripwires, and outputs real-world traffic telemetry.
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabbed Walkthrough Guide
tab_demo, tab_upload, tab_cam, tab_analytics, tab_graph, tab_math, tab_keys, tab_faq = st.tabs([
    "⚡ 1. Sample Demo Guide",
    "📁 2. Manual Upload Guide",
    "📷 3. Live Camera Guide",
    "📊 4. Traffic Analytics",
    "🕸️ 5. Intersection Graph",
    "📐 6. Discrete Math Lab",
    "🔑 7. API Keys & Secrets",
    "❓ 8. FAQ & Troubleshooting",
])

# -----------------------------------------------------------------------------
# TAB 1: SAMPLE DEMO GUIDE
# -----------------------------------------------------------------------------
with tab_demo:
    st.markdown("### ⚡ Sample Benchmark Demo Mode (Instant Classroom Presentation)")
    st.markdown(
        """
        Use this mode when presenting to examiners or demonstrating algorithms without external video files.
        It runs a deterministic 4-approach procedural intersection simulation with collision-free kinematics.
        """
    )

    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown(
            """
            #### Step-by-Step Instructions:
            1. **Navigate to Module 1:** Click on **1. Video Analysis** in the sidebar or from the home dashboard.
            2. **Select Demo Tab:** Ensure the **⚡ Sample Benchmark Demo Working** tab is active.
            3. **Step or Run Continuous Frames:**
               - Click **▶️ Step 1 Frame** to inspect single-frame detection and track ID assignment.
               - Click **⚡ Run 10 Frames** or **⏩ Run 30 Frames** to advance traffic flow rapidly.
            4. **Observe Optical HUD Overlays:**
               - **Green Bounding Boxes:** Detected vehicles with confidence scores.
               - **Vehicle Numbers (`#1`, `#2`):** Persistent track IDs managed by centroid tracking.
               - **Lanes & Direction Vectors:** Cyan trajectory trails showing vehicular heading.
            5. **Test Emergency Vehicle Preemption:**
               - In the sidebar, select **Simulate Emergency Vehicle In** -> `North`.
               - Observe the adaptive signal controller immediately grant 100% green priority to clear the North emergency approach.
            6. **Inspect Telemetry:**
               - Scroll down to review real-time queue lengths (meters), PCU demand split donut charts, and the detection event log.
            """
        )
    with c_s2:
        st.markdown(
            """
            <div class="glass-card" style="padding: 16px;">
                <h4 style="margin-top:0; color:#10B981;">✔ Key Advantages</h4>
                <ul style="color:#CBD5E1; font-size:0.85rem; line-height:1.7;">
                    <li>Zero network or file upload dependencies.</li>
                    <li>Guaranteed signal-compliant vehicles with zero false positives.</li>
                    <li>Simulates all 4 approaches (North, South, East, West).</li>
                    <li>Includes Cars (1.0 PCU), Buses (3.0 PCU), Trucks (3.0 PCU), and Motorcycles (0.5 PCU).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Video_Analysis.py", label="🚀 Launch Sample Demo in Module 1", icon="⚡")


# -----------------------------------------------------------------------------
# TAB 2: MANUAL UPLOAD GUIDE
# -----------------------------------------------------------------------------
with tab_upload:
    st.markdown("### 📁 Manual Video Upload Mode (Custom Surveillance Analytics)")
    st.markdown(
        """
        Use this mode to analyze pre-recorded roadway footage (e.g., CCTV recordings, highway monitoring, intersection surveillance).
        """
    )

    c_u1, c_u2 = st.columns([1.6, 1.2])
    with c_u1:
        st.markdown(
            """
            #### Step-by-Step Instructions:
            1. **Prepare Your Video File:**
               - **Supported Containers:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
               - **Recommended Codec:** H.264 / AVC with standard resolution (720p or 1080p).
               - **File Size Limit:** Up to 200 MB per upload.
            2. **Upload Your Video:**
               - **Method A (Home Gateway):** On [Home Overview](app.py), choose *Custom Video File Upload* in the sidebar and drop your video into the **Custom Video Ingestion Gateway**.
               - **Method B (Module 1 Direct):** On [1. Video Analysis](pages/1_Video_Analysis.py), select the **📁 Manual Video Upload Working** tab and drop your video.
            3. **Inspect Video Metadata:**
               - The system automatically parses container headers and displays Resolution, Duration, FPS, and File Size.
            4. **Configure Spatial Filters in Sidebar:**
               - **Virtual Counting Line (Tripwire):** Place a tripwire across the stopline (e.g., at 50% frame height) to count crossed vehicles non-repetitively.
               - **Region of Interest (ROI Box):** Focus analysis on a specific approach corridor or the central junction conflict zone.
            5. **Select Detection Engine Backend:**
               - **Pretrained DNN / YOLO:** Real deep convolutional inference using `models/yolov8n.onnx`.
               - **OpenCV MOG2:** Classical background subtraction (works without pretrained model weights).
            6. **Run Processing & Export Professional Reports:**
               - Advance frames using the step controls (`▶️ Step 1 Video Frame`, `⚡ Run 10 Frames`).
               - **📗 Download Complete Excel Report (.xlsx):** Generates an official 5-sheet workbook:
                 - `Executive_Summary`: Video resolution, duration, FPS, and high-level KPIs.
                 - `Crossed_Vehicles_Log`: Complete verified log of all vehicles that crossed the cordon/tripwire with timestamps, frame index, approach, and PCU.
                 - `All_Vehicles_Inventory`: Full list of all tracked vehicles with their `Crossed? (YES/NO)` status, first/last seen times, and peak confidence.
                 - `Telemetry_Time_Series`: Frame-by-frame and second-by-second traffic progression.
                 - `Webster_Signal_Plan`: Computed green splits and delay reduction metrics.
               - **📄 Download Executive Summary (.md):** Markdown summary report ready for presentation.
               - **📥 Export Raw CSV Log (.csv):** Flat event stream for spreadsheet analysis.
            """
        )
    with c_u2:
        st.markdown(
            """
            <div class="glass-card" style="padding: 16px;">
                <h4 style="margin-top:0; color:#38BDF8;">📋 Codec Checklist</h4>
                <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                    <b>Best Performance:</b><br>
                    • MP4 container with H.264 video codec.<br>
                    • Frame rates between 20 FPS and 60 FPS.<br>
                    • Frame sampling interval: Use <code>2</code> or <code>3</code> in the sidebar on high-FPS videos to prevent CPU throttling.<br><br>
                    <b>Audio Streams:</b><br>
                    • Audio tracks are ignored; only video frames are analyzed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Video_Analysis.py", label="🚀 Open Manual Upload in Module 1", icon="📁")


# -----------------------------------------------------------------------------
# TAB 3: LIVE CAMERA GUIDE
# -----------------------------------------------------------------------------
with tab_cam:
    st.markdown("### 📷 Live Surveillance Camera Feed")
    st.markdown(
        """
        Module 2 supports real-time hardware optical sensing for edge-controller demonstrations.
        """
    )
    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.markdown(
            """
            #### Mode A: Local Hardware Camera (OpenCV)
            - **When to Use:** Running locally on your desktop or laptop with a USB webcam attached.
            - **How it Works:** Connects directly to hardware index `0` via `cv2.VideoCapture()`.
            - **Controls:** Click **▶️ Connect Local Hardware Camera**, then use single-frame capture or live stream mode.
            - **Safe Release:** Always click **🛑 Disconnect Camera** when finished to release operating system hardware handles.
            """
        )
    with c_c2:
        st.markdown(
            """
            #### Mode B: Browser Camera (WebRTC)
            - **When to Use:** Accessing the dashboard remotely over the local network or in cloud deployments.
            - **How it Works:** Uses Streamlit's native `st.camera_input()` HTML5 MediaDevices API.
            - **Controls:** Allows taking instantaneous snapshots from any phone, tablet, or client computer.
            - **Privacy Guarantee:** Frames are processed in volatile memory with zero unauthorized network transmission.
            """
        )
    st.page_link("pages/2_Live_Camera.py", label="🚀 Open Live Camera Module", icon="📷")


# -----------------------------------------------------------------------------
# TAB 4: TRAFFIC ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Macro Traffic Analytics & Adaptive Signal Optimization")
    st.markdown(
        """
        Module 3 evaluates intersection capacity, queue metrics, and compares **Webster's fixed-time baseline** against **AI Proportional Adaptive Signal Allocation**.
        """
    )
    st.markdown(
        """
        #### Key Features & How to Operate:
        1. **Select Data Scenario:**
           - Choose between the verified 4-approach benchmark dataset (`data/sample_traffic.csv`), deterministic synthetic stress tests (e.g., Heavy North-South Skew, Night-Time Empty, Peak Saturation), or upload a custom observation CSV.
        2. **Configure Signal Constraints in Sidebar:**
           - **Fixed Cycle Budget (C):** Total cycle duration (e.g. 90 seconds).
           - **Min Green Floor (g_min):** Minimum green time guaranteed to minor approaches (e.g. 10s) to prevent starvation.
           - **Max Green Cap (g_max):** Maximum green allocation per phase (e.g. 60s).
        3. **Compare Static vs Adaptive Timing:**
           - Inspect the **Signal Split Comparison Bar** showing how green time dynamically shifts toward heavier approaches while preserving cycle invariance:
             $$\\sum g_i + \\text{clearance} = C$$
        4. **Level of Service (LOS) Gauges:**
           - Monitored based on Highway Capacity Manual (HCM) standards: LOS A (Free Flow, $<10$s delay) through LOS F (Breakdown, $>80$s delay).
        """
    )
    st.page_link("pages/3_Traffic_Analytics.py", label="🚀 Open Traffic Analytics Module", icon="📊")


# -----------------------------------------------------------------------------
# TAB 5: INTERSECTION GRAPH
# -----------------------------------------------------------------------------
with tab_graph:
    st.markdown("### 🕸️ Urban Road Network & Intersection Graph Theory")
    st.markdown(
        """
        Module 4 models urban corridors as a directed weighted graph $G = (V, E)$, where intersections are vertices $V$ and road links are edges $E$.
        """
    )
    st.markdown(
        """
        #### How to Operate:
        1. **Origin & Destination Path Finding:**
           - Select **Origin Intersection** (e.g. $A$) and **Destination Intersection** (e.g. $F$).
           - The system calculates the optimal route using **Dijkstra's Shortest Path Algorithm** with dynamic travel times.
           - Compare route results with **Breadth-First Search (BFS)** (which optimizes hop count rather than travel time).
        2. **Simulate Traffic Congestion & BPR Impedance:**
           - In the sidebar, select a specific road link (e.g. Corridor $B \\to D$) and increase its vehicular load.
           - The system applies the **Bureau of Public Roads (BPR)** formulation:
             $$t = t_0 \\left[ 1 + \\alpha \\left(\\frac{V}{C}\\right)^\\beta \\right]$$
             where $\\alpha = 0.15$ and $\\beta = 4$. Observe how edge impedance increases exponentially near capacity.
        3. **Simulate Incident Road Closures:**
           - Toggle **Close Road Corridor** to simulate accidents or construction closures.
           - The graph automatically recalculates bypass routes or alerts if the graph becomes disconnected.
        4. **Inspect Discrete Math Matrices:**
           - View the binary adjacency matrix $A \\in \\{0, 1\\}^{n \\times n}$, weighted cost matrix, in-degree, out-degree, and Handshaking Lemma verification:
             $$\\sum_{v \\in V} \\text{deg}(v) = 2|E|$$
        """
    )
    st.page_link("pages/4_Intersection_Graph.py", label="🚀 Open Intersection Graph Module", icon="🕸️")


# -----------------------------------------------------------------------------
# TAB 6: DISCRETE MATH LAB
# -----------------------------------------------------------------------------
with tab_math:
    st.markdown("### 📐 Discrete Mathematics Capstone Laboratory (Units I – V)")
    st.markdown(
        """
        Module 5 contains worked, interactive proofs and mathematical demonstrations directly mapped to all 5 units of the university syllabus.
        """
    )
    st.markdown(
        """
        | Unit | Syllabus Domain | Traffic Application Demonstration |
        | :--- | :--- | :--- |
        | **Unit I** | Logic and Proofs | Truth tables, propositional congestion rules, DNF/CNF conversion, Modus Ponens/Tollens inference |
        | **Unit II** | Sets and Relations | Vehicle classification sets, Cartesian products, equivalence relations (same-lane), partial orders & lattices |
        | **Unit III** | Combinatorics | Phase cycle permutations $P(n, r)$, Pigeonhole Principle lane congestion guarantees, Induction proofs |
        | **Unit IV** | Algebraic Systems | Binary operations, Semigroups, Monoids, Phase Rotation Group $(\\mathbb{Z}_4, +_4)$, Klein 4-group, Homomorphisms |
        | **Unit V** | Graph Theory | Welsh-Powell vertex coloring for conflict-free phases, Euler's planar flyover formula $V - E + R = 2$, Laplacian spectra |
        """
    )
    st.page_link("pages/5_Discrete_Mathematics_Lab.py", label="🚀 Open Discrete Mathematics Lab", icon="📐")


# -----------------------------------------------------------------------------
# TAB 7: API KEYS & SECRETS
# -----------------------------------------------------------------------------
with tab_keys:
    st.markdown("### 🔑 API Key & Secrets Setup Guide")
    st.markdown(
        """
        The project features a **3-tier secrets architecture** ensuring that external credentials (e.g. Google Gemini AI, TomTom, OpenAI) are handled securely.
        """
    )
    c_k1, c_k2 = st.columns(2)
    with c_k1:
        st.markdown(
            """
            #### Option A: Streamlit Secrets (`.streamlit/secrets.toml`)
            Create `.streamlit/secrets.toml` in your project root:
            ```toml
            API_KEY = "your_actual_key_here"
            GEMINI_API_KEY = "your_gemini_key_here"

            [gemini]
            api_key = "your_gemini_key_here"
            ```
            - **Git Safe:** `.streamlit/secrets.toml` is in `.gitignore` and is never committed.
            - **Cloud Deployment:** In Streamlit Cloud, paste these exact key-value pairs into the *Secrets* settings modal.
            """
        )
    with c_k2:
        st.markdown(
            """
            #### Option B: Environment Variables (`.env`)
            Create `.env` in your project root:
            ```env
            API_KEY=your_actual_key_here
            GEMINI_API_KEY=your_gemini_key_here
            ```
            - **Git Safe:** `.env` is also in `.gitignore`.
            
            #### Option C: Interactive Sidebar Input
            - On the [Home Overview](app.py), expand **"🔑 API Key & Secrets Settings"** in the sidebar.
            - Enter your key into the masked password field to test it for the active session.
            """
        )


# -----------------------------------------------------------------------------
# TAB 8: FAQ & TROUBLESHOOTING
# -----------------------------------------------------------------------------
with tab_faq:
    st.markdown("### ❓ Frequently Asked Questions & Troubleshooting")
    
    with st.expander("Q1: What is the difference between Sample Demo Mode and Manual Upload Mode?", expanded=True):
        st.markdown(
            """
            - **Sample Demo Mode** uses an automated procedural simulation that renders vehicles continuously onto a synthetic 4-way intersection canvas. It requires no uploaded files, no camera, and works 100% offline.
            - **Manual Upload Mode** accepts real pre-recorded video files (`.mp4`, `.avi`, `.mov`) from surveillance cameras and runs full computer-vision object detection, persistent tracking, and stopline tripwire counts.
            """
        )

    with st.expander("Q2: Why does my uploaded video show an error or blank frames?"):
        st.markdown(
            """
            - Ensure the video is encoded using the **H.264 (AVC)** codec.
            - Some HEVC (H.265) or proprietary DVR formats (such as raw `.dav` files) lack native OpenCV codec support on Windows. Convert them to MP4 using standard tools (e.g., Handbrake or VLC).
            - Check that your video resolution is standard (720p or 1080p recommended).
            """
        )

    with st.expander("Q3: How do I download or configure the YOLOv8 model weights?"):
        st.markdown(
            """
            - Run `python setup_models.py` in your terminal to download the official 12.8 MB `yolov8n.onnx` model weights.
            - Alternatively, click **📥 Download YOLOv8 Weights** in the sidebar of Module 1 or in the Home Page Diagnostic Inspector.
            - If internet access is restricted, the system seamlessly falls back to **OpenCV MOG2 Background Subtraction** or **Procedural Simulation**.
            """
        )

    with st.expander("Q4: Can I run this system without a GPU?"):
        st.markdown(
            """
            Yes! The entire pipeline is optimized for standard CPU execution. 
            ONNX Runtime and OpenCV use optimized CPU vector instructions. 
            On high-FPS videos, increase the **Frame Sampling Interval** in the sidebar to `2` or `3` to reduce CPU workload.
            """
        )

    with st.expander("Q5: Are the calculated signal timings real municipal commands?"):
        st.markdown(
            """
            **No.** All adaptive signal timings, green splits, and queue metrics are computed strictly for **academic simulation and educational benchmarking**. They are not connected to physical municipal traffic signal controllers.
            """
        )

st.write("")
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
        Smart Traffic Management System | Department of Computer Science & Engineering<br>
        B.Tech 2nd Year Academic Capstone Project
    </div>
    """,
    unsafe_allow_html=True,
)
