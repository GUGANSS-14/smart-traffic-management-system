"""
Page 3: Traffic Analytics & Signal Optimization
Macro-level traffic analytics engine featuring:
1. Vehicle counts by class, approach lane, and configurable time intervals.
2. Spatial density proxy (vehicles / normalized area) and spatial occupancy.
3. Configurable 3-level traffic classification (Low / Moderate / High).
4. Estimated traffic flow from virtual counting tripwires (vehicles / minute).
5. Comprehensive documentation of metric assumptions and limitations.
6. Transparent, rule-based signal timing recommendations under fixed cycle budgets.
7. Strict non-negative boundary constraint satisfaction and cycle preservation.
8. Current simulated timing vs Recommended adaptive timing comparison with divergence explanation.
9. Starvation fairness safeguards for minor approaches.
10. Rich dark navy & cyan dashboards, KPI cards, interactive Plotly charts, and CSV export.
11. Deterministic synthetic scenario testing.
12. Explicit student simulation labeling.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from modules.traffic_analyzer import TrafficAnalyzer, LaneMetrics
from modules.signal_optimizer import SignalOptimizer, SignalPlan, SignalComparison, SIMULATION_DISCLAIMER
from visualizations.charts import (
    get_custom_css,
    DARK_LAYOUT_TEMPLATE,
    create_traffic_timeline_plot,
    create_static_vs_adaptive_comparison_bar,
    create_signal_split_comparison_bar,
    create_lane_multimetric_bar,
    create_traffic_state_donut,
)

# Page Setup
st.set_page_config(
    page_title="Traffic Analytics - Smart Traffic",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Main Title & Academic Simulation Disclaimer
st.title("📊 Traffic Analytics & Signal Optimization Engine")
st.markdown(
    """
    **Transparent Rule-Based Analytics, Spatial Density Proxies, and Deterministic Signal Optimization**  
    *Evaluates macro-level intersection telemetry, vehicle class distributions, and demand-responsive green allocations.*
    """
)

# Prominent Student Simulation Warning Callout
st.markdown(
    """
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.4rem;">⚠️</span>
            <div style="font-size:0.85rem; color:#FDE68A; line-height:1.5;">
                <b>ACADEMIC SIMULATION NOTICE:</b> All signal timings, phase splits, flow rates, and delay metrics 
                are algorithmic recommendations calculated strictly for <b>educational simulation and coursework benchmarking</b>. 
                They are <b>NOT</b> approved control commands for physical road traffic controllers or municipal infrastructure.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
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

    st.markdown("### 🎛️ Dataset & Scenario Controls")

    data_mode = st.radio(
        "Data Source",
        [
            "Default Dataset (data/sample_traffic.csv)",
            "Synthetic Scenario Presets",
            "Upload Custom Observation CSV",
        ],
    )

    if data_mode == "Default Dataset (data/sample_traffic.csv)":
        df_raw = TrafficAnalyzer.load_sample_dataset("data/sample_traffic.csv")
        active_scenario_name = "Default Intersection Benchmark"

    elif data_mode == "Synthetic Scenario Presets":
        scenario_preset = st.selectbox(
            "Select Synthetic Scenario",
            [
                "Heavily Skewed (Heavy North-South / Deserted East-West)",
                "Equal Symmetric Demand (Balanced Traffic)",
                "Empty Traffic (Night-Time 0 Vehicles)",
                "Emergency Corridor Approaching North",
                "High Congestion / Near Saturation",
            ],
        )
        active_scenario_name = scenario_preset

        # Deterministic synthetic generators
        if scenario_preset == "Empty Traffic (Night-Time 0 Vehicles)":
            df_raw = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime("2026-10-10 03:00:00"),
                    "intersection_id": "INT_SYNTH",
                    "approach_lane": lane,
                    "cars": 0, "buses": 0, "trucks": 0, "motorcycles": 0,
                    "total_vehicles": 0, "pcu_total": 0.0,
                    "average_speed_kmh": 45.0, "queue_length_m": 0.0,
                    "delay_seconds_static": 42.5, "delay_seconds_adaptive": 42.5,
                }
                for lane in ["North", "South", "East", "West"]
            ])

        elif scenario_preset == "Equal Symmetric Demand (Balanced Traffic)":
            df_raw = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime("2026-10-10 12:00:00"),
                    "intersection_id": "INT_SYNTH",
                    "approach_lane": lane,
                    "cars": 25, "buses": 2, "trucks": 1, "motorcycles": 10,
                    "total_vehicles": 38, "pcu_total": 39.0,
                    "average_speed_kmh": 32.0, "queue_length_m": 35.0,
                    "delay_seconds_static": 45.0, "delay_seconds_adaptive": 36.0,
                }
                for lane in ["North", "South", "East", "West"]
            ])

        elif scenario_preset == "Emergency Corridor Approaching North":
            df_raw = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime("2026-10-10 14:30:00"),
                    "intersection_id": "INT_SYNTH",
                    "approach_lane": lane,
                    "cars": 45 if lane == "North" else 15,
                    "buses": 5 if lane == "North" else 1,
                    "trucks": 3 if lane == "North" else 1,
                    "motorcycles": 20 if lane == "North" else 8,
                    "total_vehicles": 73 if lane == "North" else 25,
                    "pcu_total": 79.0 if lane == "North" else 25.0,
                    "average_speed_kmh": 15.0 if lane == "North" else 38.0,
                    "queue_length_m": 85.0 if lane == "North" else 20.0,
                    "delay_seconds_static": 68.0 if lane == "North" else 32.0,
                    "delay_seconds_adaptive": 32.0 if lane == "North" else 28.0,
                }
                for lane in ["North", "South", "East", "West"]
            ])

        elif scenario_preset == "High Congestion / Near Saturation":
            df_raw = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime("2026-10-10 18:00:00"),
                    "intersection_id": "INT_SYNTH",
                    "approach_lane": lane,
                    "cars": 60, "buses": 8, "trucks": 5, "motorcycles": 40,
                    "total_vehicles": 113, "pcu_total": 119.0,
                    "average_speed_kmh": 12.0, "queue_length_m": 125.0,
                    "delay_seconds_static": 85.0, "delay_seconds_adaptive": 46.0,
                }
                for lane in ["North", "South", "East", "West"]
            ])

        else:  # Heavily Skewed (Heavy North-South / Deserted East-West)
            df_raw = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime("2026-10-10 08:30:00"),
                    "intersection_id": "INT_SYNTH",
                    "approach_lane": lane,
                    "cars": 55 if lane in ["North", "South"] else 2,
                    "buses": 6 if lane in ["North", "South"] else 0,
                    "trucks": 4 if lane in ["North", "South"] else 0,
                    "motorcycles": 35 if lane in ["North", "South"] else 2,
                    "total_vehicles": 100 if lane in ["North", "South"] else 4,
                    "pcu_total": 102.5 if lane in ["North", "South"] else 3.0,
                    "average_speed_kmh": 14.0 if lane in ["North", "South"] else 45.0,
                    "queue_length_m": 110.0 if lane in ["North", "South"] else 4.0,
                    "delay_seconds_static": 75.0 if lane in ["North", "South"] else 20.0,
                    "delay_seconds_adaptive": 36.0 if lane in ["North", "South"] else 22.0,
                }
                for lane in ["North", "South", "East", "West"]
            ])

    else:  # Upload Custom Observation CSV
        custom_csv = st.file_uploader("Upload Traffic Observation CSV", type=["csv"])
        if custom_csv:
            df_raw = pd.read_csv(custom_csv)
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            active_scenario_name = "User Uploaded Dataset"
        else:
            df_raw = TrafficAnalyzer.load_sample_dataset("data/sample_traffic.csv")
            active_scenario_name = "Default Intersection Benchmark"

    st.markdown("---")
    st.markdown("### 🚦 Configurable Signal Constraints")
    cycle_budget = st.slider("Fixed Cycle Budget (seconds)", min_value=45, max_value=150, value=90, step=5)
    min_green_setting = st.slider("Min Green Floor (seconds - Fairness)", min_value=5, max_value=25, value=10, step=1)
    max_green_setting = st.slider("Max Green Cap (seconds)", min_value=35, max_value=90, value=60, step=5)
    emergency_lane_override = st.selectbox(
        "Emergency Preemption Override",
        ["None", "North", "South", "East", "West"],
    )
    emergency_target = None if emergency_lane_override == "None" else emergency_lane_override

    st.markdown("---")
    st.markdown("### 📐 Density & Classification Cutoffs")
    low_cutoff = st.slider("Low Traffic Cutoff (% of capacity)", min_value=15.0, max_value=45.0, value=35.0, step=1.0)
    high_cutoff = st.slider("High Traffic Cutoff (% of capacity)", min_value=55.0, max_value=85.0, value=70.0, step=1.0)
    norm_area = st.slider("ROI Normalized Area (Screen Fraction)", min_value=0.2, max_value=1.0, value=1.0, step=0.05)

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    lanes_selected = st.multiselect(
        "Filter Approach Lanes",
        options=["North", "South", "East", "West"],
        default=["North", "South", "East", "West"],
    )

# Enrich Dataset with Analytical Metrics
analyzer = TrafficAnalyzer(
    lane_capacity_pcu=30.0,
    low_threshold=low_cutoff,
    high_threshold=high_cutoff,
)

work_df = df_raw.copy()

# Ensure required columns exist
if "density_proxy" not in work_df.columns:
    if "total_vehicles" in work_df.columns:
        work_df["density_proxy"] = work_df["total_vehicles"].apply(
            lambda v: analyzer.calculate_density_proxy(v, norm_area)
        )
    else:
        work_df["density_proxy"] = 0.0

if "flow_rate_per_min" not in work_df.columns:
    if "total_vehicles" in work_df.columns:
        work_df["flow_rate_per_min"] = work_df["total_vehicles"].apply(
            lambda v: analyzer.estimate_traffic_flow(int(v), 60.0)
        )
    else:
        work_df["flow_rate_per_min"] = 0.0

# Dynamic 3-level traffic classification based on configurable cutoffs
if "pcu_total" in work_df.columns:
    work_df["capacity_pct"] = (work_df["pcu_total"] / 30.0) * 100.0
    work_df["traffic_state"] = work_df["capacity_pct"].apply(
        lambda cp: analyzer.classify_traffic(cp)
    )
else:
    work_df["traffic_state"] = "Low"

# Filter by selected lanes
filtered_df = work_df[work_df["approach_lane"].isin(lanes_selected)] if lanes_selected else work_df.copy()

# Compute global KPI metrics
kpis = TrafficAnalyzer.compute_kpi_summary(filtered_df)

# Evaluate Aggregate Approach Demand for Signal Optimization
lane_demand_dict: Dict[str, LaneMetrics] = {}
for lane in ["North", "South", "East", "West"]:
    lane_sub = filtered_df[filtered_df["approach_lane"] == lane]
    if not lane_sub.empty:
        mean_pcu = float(lane_sub["pcu_total"].mean()) if "pcu_total" in lane_sub.columns else 0.0
        mean_veh = int(round(lane_sub["total_vehicles"].mean())) if "total_vehicles" in lane_sub.columns else 0
        lane_demand_dict[lane] = LaneMetrics(
            lane_id=lane,
            pcu_total=mean_pcu,
            total_vehicles=mean_veh,
            capacity_pcu=30.0,
        )
    else:
        lane_demand_dict[lane] = LaneMetrics(lane_id=lane, pcu_total=0.0, total_vehicles=0)

# Run Rule-Based Deterministic Signal Optimizer
optimizer = SignalOptimizer(
    min_green=min_green_setting,
    max_green=max_green_setting,
    yellow_time=3,
    all_red=2,
    default_cycle=cycle_budget,
)

static_plan = optimizer.compute_static_plan(cycle_time=cycle_budget)
comparison: SignalComparison = optimizer.recommend_and_compare(
    lane_metrics=lane_demand_dict,
    current_plan=static_plan,
    cycle_time=cycle_budget,
    min_green=min_green_setting,
    max_green=max_green_setting,
    emergency_lane=emergency_target,
)
rec_plan = comparison.recommended_plan

# Top KPI Metric Cards
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
with kpi_c1:
    st.metric("Total Vehicles", f"{kpis['total_vehicles_processed']:,}", "Logged Volume")
with kpi_c2:
    st.metric("Average PCU Load", f"{kpis['avg_pcu']} PCU", "Per Approach")
with kpi_c3:
    avg_dens = round(float(filtered_df["density_proxy"].mean()), 1) if not filtered_df.empty else 0.0
    st.metric("Mean Density Proxy", f"{avg_dens}", "veh / unit area")
with kpi_c4:
    avg_flow = round(float(filtered_df["flow_rate_per_min"].mean()), 1) if not filtered_df.empty else 0.0
    st.metric("Estimated Flow", f"{avg_flow} vpm", "Vehicles / Minute")
with kpi_c5:
    st.metric("Delay Reduction", f"{rec_plan.delay_reduction_pct}%", "vs Static Webster", delta_color="normal")
with kpi_c6:
    st.metric("CO₂ Prevented", f"{kpis['co2_saved_kg']} kg", "Idling Averted")

st.write("")

# Signal Optimization & Timing Recommendation Section
st.markdown("### 🚦 Simulated Signal Timing Recommendation & Comparison")
st.markdown(
    """
    Transparent comparison between the **Current Simulated Fixed Timing (50/50 Baseline)** 
    and the **Recommended Adaptive Timing Plan**, subject to strict minimum green fairness floors and fixed cycle budgets.
    """
)

col_plans, col_bars = st.columns([1.1, 0.9])

with col_plans:
    # Side-by-side cards
    c_p1, c_p2 = st.columns(2)

    with c_p1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 3px solid #64748B;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">Simulated Baseline</div>
                <h4 style="margin:4px 0 12px 0; color:#F8FAFC;">Static Fixed Schedule</h4>
                <div style="margin-bottom:8px; font-size:0.85rem;">
                    <span style="color:#94A3B8;">Cycle Budget:</span> <b>{static_plan.cycle_time_seconds}s</b>
                </div>
                <div style="background:#070D18; border-radius:8px; padding:10px; margin-bottom:8px;">
                    <div style="color:#38BDF8; font-weight:700; font-size:0.85rem;">Phase 1 (North-South)</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#F8FAFC;">{static_plan.phases[0].green_time_seconds}s <span style="font-size:0.75rem; color:#94A3B8;">GREEN</span></div>
                    <div style="font-size:0.75rem; color:#64748B;">Clearance: 3s Yellow + 2s All-Red</div>
                </div>
                <div style="background:#070D18; border-radius:8px; padding:10px;">
                    <div style="color:#3B82F6; font-weight:700; font-size:0.85rem;">Phase 2 (East-West)</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#F8FAFC;">{static_plan.phases[1].green_time_seconds}s <span style="font-size:0.75rem; color:#94A3B8;">GREEN</span></div>
                    <div style="font-size:0.75rem; color:#64748B;">Clearance: 3s Yellow + 2s All-Red</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_p2:
        d_ns = comparison.phase_deltas.get("Phase 1 (North-South)", 0)
        d_ew = comparison.phase_deltas.get("Phase 2 (East-West)", 0)
        badge_ns = f"+{d_ns}s" if d_ns > 0 else (f"{d_ns}s" if d_ns < 0 else "0s")
        badge_ew = f"+{d_ew}s" if d_ew > 0 else (f"{d_ew}s" if d_ew < 0 else "0s")
        color_ns = "#10B981" if d_ns > 0 else ("#EF4444" if d_ns < 0 else "#94A3B8")
        color_ew = "#10B981" if d_ew > 0 else ("#EF4444" if d_ew < 0 else "#94A3B8")

        p1_rec = rec_plan.phases[0]
        p2_rec = rec_plan.phases[1]

        st.markdown(
            f"""
            <div class="glass-card glass-card-glow" style="border-top: 3px solid #06B6D4;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#22D3EE; font-weight:700;">Adaptive Recommendation</div>
                    <span class="badge badge-cyan" style="font-size:0.65rem;">Simulation Only</span>
                </div>
                <h4 style="margin:4px 0 12px 0; color:#F8FAFC;">Demand-Responsive Plan</h4>
                <div style="margin-bottom:8px; font-size:0.85rem;">
                    <span style="color:#94A3B8;">Cycle Budget:</span> <b>{rec_plan.cycle_time_seconds}s</b>
                </div>
                <div style="background:#070D18; border-radius:8px; padding:10px; margin-bottom:8px; border-left:3px solid #06B6D4;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#38BDF8; font-weight:700; font-size:0.85rem;">Phase 1 (North-South)</span>
                        <span style="color:{color_ns}; font-weight:700; font-size:0.8rem; font-family:monospace;">Δ {badge_ns}</span>
                    </div>
                    <div style="font-size:1.35rem; font-weight:800; color:#38BDF8;">{p1_rec.green_time_seconds}s <span style="font-size:0.75rem; color:#94A3B8;">GREEN</span></div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Demand: {p1_rec.demand_pcu} PCU | Status: <b>{p1_rec.priority_level}</b></div>
                </div>
                <div style="background:#070D18; border-radius:8px; padding:10px; border-left:3px solid #3B82F6;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#3B82F6; font-weight:700; font-size:0.85rem;">Phase 2 (East-West)</span>
                        <span style="color:{color_ew}; font-weight:700; font-size:0.8rem; font-family:monospace;">Δ {badge_ew}</span>
                    </div>
                    <div style="font-size:1.35rem; font-weight:800; color:#60A5FA;">{p2_rec.green_time_seconds}s <span style="font-size:0.75rem; color:#94A3B8;">GREEN</span></div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Demand: {p2_rec.demand_pcu} PCU | Status: <b>{p2_rec.priority_level}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_bars:
    fig_split = create_signal_split_comparison_bar(
        current_p1_green=static_plan.phases[0].green_time_seconds,
        current_p2_green=static_plan.phases[1].green_time_seconds,
        rec_p1_green=rec_plan.phases[0].green_time_seconds,
        rec_p2_green=rec_plan.phases[1].green_time_seconds,
        clearance_time=static_plan.total_lost_time,
    )
    st.plotly_chart(fig_split, use_container_width=True)

# Divergence Explanation & Fairness Safeguards Box
col_expl, col_fair = st.columns([1.1, 0.9])

with col_expl:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #38BDF8;">
            <h5 style="color:#38BDF8; margin-top:0; font-size:0.95rem;">💡 Why Does the Recommendation Differ?</h5>
            <p style="color:#E2E8F0; font-size:0.85rem; line-height:1.6; margin-bottom:0;">
                {comparison.divergence_explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_fair:
    fairness_status_icon = "🛡️" if comparison.fairness_safeguard_active else "✅"
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #10B981;">
            <h5 style="color:#34D399; margin-top:0; font-size:0.95rem;">{fairness_status_icon} Fairness Safeguards & Invariants</h5>
            <p style="color:#E2E8F0; font-size:0.85rem; line-height:1.6; margin-bottom:6px;">
                {comparison.fairness_notice}
            </p>
            <div style="font-size:0.8rem; color:#94A3B8;">
                • Cycle Budget: <b>{rec_plan.cycle_time_seconds}s</b> (Preserved: {rec_plan.total_green_time}s green + {rec_plan.total_lost_time}s clearance = {rec_plan.cycle_time_seconds}s)<br>
                • Minimum Green Constraint: <b>g ≥ {optimizer.min_green}s</b> strictly guaranteed (no negative splits).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Multi-Tab Analytics Views
tab_trends, tab_fleet, tab_lanes, tab_assumptions, tab_data = st.tabs([
    "📈 Temporal Volume & Flow Trends",
    "🚗 Vehicle Modal Composition",
    "⚖️ Lane Demand & Traffic States",
    "📘 Metric Assumptions & Limitations",
    "📋 Tabular Data & CSV Export",
])

with tab_trends:
    st.markdown("### ⏱️ Temporal Volume Trends & Congestion Heatmap")
    fig_timeline = create_traffic_timeline_plot(filtered_df)
    st.plotly_chart(fig_timeline, use_container_width=True)

    c_heat1, c_heat2 = st.columns(2)
    with c_heat1:
        if not filtered_df.empty and "timestamp" in filtered_df.columns:
            df_heat = filtered_df.copy()
            df_heat["hour"] = df_heat["timestamp"].dt.strftime("%H:%M")
            pivot = df_heat.pivot_table(index="approach_lane", columns="hour", values="pcu_total", aggfunc="mean").fillna(0)
            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Time of Day", y="Approach Corridor", color="PCU Load"),
                color_continuous_scale="Viridis",
                title="<b>Approach Congestion Intensity Heatmap (PCU Load)</b>",
            )
            fig_heat.update_layout(**DARK_LAYOUT_TEMPLATE, height=320)
            st.plotly_chart(fig_heat, use_container_width=True)

    with c_heat2:
        if not filtered_df.empty and "timestamp" in filtered_df.columns:
            fig_flow = px.line(
                filtered_df,
                x="timestamp",
                y="flow_rate_per_min",
                color="approach_lane",
                title="<b>Estimated Flow Rate Over Time (Vehicles / Minute)</b>",
                color_discrete_sequence=["#38BDF8", "#06B6D4", "#10B981", "#F59E0B"],
            )
            fig_flow.update_layout(**DARK_LAYOUT_TEMPLATE, height=320)
            st.plotly_chart(fig_flow, use_container_width=True)

    # Configurable Interval Aggregation Sub-Section
    st.markdown("#### ⏳ Resampled Vehicle Counts by Interval")
    interval_choice = st.selectbox("Aggregation Time Window", ["5min", "15min", "30min", "1h"], index=1)
    df_interval = TrafficAnalyzer.calculate_vehicle_counts_by_interval(
        df=filtered_df,
        interval=interval_choice,
        time_column="timestamp",
        lane_column="approach_lane",
    )
    if not df_interval.empty:
        st.dataframe(df_interval, use_container_width=True)
    else:
        st.info("No time series records available for interval resampling.")

with tab_fleet:
    st.markdown("### 🚗 Fleet Modal Composition & Passenger Car Unit (PCU) Weighting")
    col_donut, col_pcu_table = st.columns([1.1, 0.9])

    with col_donut:
        veh_counts = {
            "Cars (PCU 1.0)": int(filtered_df["cars"].sum()) if "cars" in filtered_df.columns else 0,
            "Buses (PCU 3.0)": int(filtered_df["buses"].sum()) if "buses" in filtered_df.columns else 0,
            "Trucks (PCU 3.0)": int(filtered_df["trucks"].sum()) if "trucks" in filtered_df.columns else 0,
            "Motorcycles (PCU 0.5)": int(filtered_df["motorcycles"].sum()) if "motorcycles" in filtered_df.columns else 0,
        }

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=list(veh_counts.keys()),
                    values=list(veh_counts.values()),
                    hole=0.55,
                    marker=dict(colors=["#06B6D4", "#F59E0B", "#EF4444", "#10B981"]),
                )
            ]
        )
        fig_pie.update_layout(**DARK_LAYOUT_TEMPLATE, title="<b>Observed Vehicle Classification Share</b>", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_pcu_table:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38BDF8; margin-top:0;">📘 Passenger Car Unit (PCU / PCE) Factors</h4>
                <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    Heterogeneous traffic streams cannot be managed purely by vehicle counts.
                    A commercial transit bus consumes more physical lane headway, generates larger blind spots,
                    and exhibits slower acceleration compared to a passenger car.
                </p>
                <table style="width:100%; font-size:0.85rem; color:#E2E8F0; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #1E293B;">
                        <th style="text-align:left; padding:6px;">Class</th>
                        <th style="text-align:center; padding:6px;">PCU Weight</th>
                        <th style="text-align:right; padding:6px;">Equivalent Footprint</th>
                    </tr>
                    <tr style="border-bottom:1px solid #1E293B;">
                        <td style="padding:6px;">Standard Passenger Car</td>
                        <td style="text-align:center; color:#38BDF8; font-weight:700;">1.00</td>
                        <td style="text-align:right;">1.00 Baseline Unit</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1E293B;">
                        <td style="padding:6px;">Transit Bus</td>
                        <td style="text-align:center; color:#F59E0B; font-weight:700;">3.00</td>
                        <td style="text-align:right;">3.00 Car Equivalents</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1E293B;">
                        <td style="padding:6px;">Heavy Cargo Truck</td>
                        <td style="text-align:center; color:#EF4444; font-weight:700;">3.00</td>
                        <td style="text-align:right;">3.00 Car Equivalents</td>
                    </tr>
                    <tr>
                        <td style="padding:6px;">Motorcycle / Two-Wheeler</td>
                        <td style="text-align:center; color:#10B981; font-weight:700;">0.50</td>
                        <td style="text-align:right;">0.50 Car Equivalents</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_lanes:
    st.markdown("### ⚖️ Approach Corridor Comparison & Traffic State Classification")
    c_l1, c_l2 = st.columns([1.2, 0.8])

    with c_l1:
        fig_multibar = create_lane_multimetric_bar(filtered_df)
        st.plotly_chart(fig_multibar, use_container_width=True)

    with c_l2:
        fig_donut_state = create_traffic_state_donut(filtered_df)
        st.plotly_chart(fig_donut_state, use_container_width=True)

    # Delay Reduction Comparison
    st.markdown("#### ⏱️ Delay Reduction by Corridor (Static Webster vs Recommended Adaptive)")
    fig_comp = create_static_vs_adaptive_comparison_bar(filtered_df)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab_assumptions:
    st.markdown("### 📘 Metric Formulations, Assumptions, and Practical Limitations")
    st.markdown(
        """
        To maintain transparency and scientific rigor, this system documents the exact mathematical formulation,
        traffic engineering assumptions, and physical camera limitations for every computed metric.
        """
    )

    doc_metrics = TrafficAnalyzer.explain_metric_assumptions()

    for m_key, m_info in doc_metrics.items():
        with st.expander(f"📌 {m_info['name']} — Formulation & Limitations", expanded=(m_key in ["density_proxy", "traffic_flow"])):
            st.markdown(
                f"""
                <div style="background:#070D18; padding:12px 16px; border-radius:8px; border-left:3px solid #06B6D4; margin-bottom:12px;">
                    <div style="font-family:monospace; color:#38BDF8; font-size:0.95rem; font-weight:700;">
                        {m_info['formula']}
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                    <div style="background:#0F172A; padding:14px; border-radius:8px; border:1px solid #1E293B;">
                        <h5 style="color:#10B981; margin-top:0;">✅ Engineering Assumptions</h5>
                        <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6; margin-bottom:0;">
                            {m_info['assumptions']}
                        </p>
                    </div>
                    <div style="background:#0F172A; padding:14px; border-radius:8px; border:1px solid #1E293B;">
                        <h5 style="color:#F59E0B; margin-top:0;">⚠️ Practical & Camera Limitations</h5>
                        <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6; margin-bottom:0;">
                            {m_info['limitations']}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Additional Civil Engineering Theory
    with st.expander("📐 Webster's Delay Formula & Cycle Budget Optimization Theory"):
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38BDF8; margin-top:0;">Classical Webster Signal Timing Model</h4>
                <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    The delay per vehicle at a signalized intersection is modeled by Webster's fundamental equation:
                </p>
                <div style="background:#070D18; padding:12px; border-radius:8px; font-family:monospace; color:#38BDF8; text-align:center; font-size:0.9rem; margin:10px 0;">
                    d = [C(1 - λ)² / 2(1 - λx)] + [x² / 2q(1 - x)] - 0.65(C / q²)^(1/3) * x^(2 + 5λ)
                </div>
                <p style="color:#94A3B8; font-size:0.8rem; line-height:1.5;">
                    Where:
                    <br>• <b>C</b> = Cycle length (seconds)
                    <br>• <b>λ</b> = Green ratio (g / C) for the given phase
                    <br>• <b>q</b> = Arrival flow rate (vehicles/second)
                    <br>• <b>x</b> = Degree of saturation (q / (s · λ)), where <i>s</i> is saturation flow rate (~1800 pcu/hr/lane).
                </p>
                <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                    <b>Dynamic AI Advantage:</b> Pre-timed fixed controllers enforce static <i>λ</i> regardless of actual <i>q</i>, 
                    causing <i>x &gt; 1.0</i> (oversaturation) during peak surges. Our rule-based adaptive controller dynamically reallocates 
                    surplus cycle budget <i>G_surplus</i> in real time, keeping <i>x</i> balanced across all approaches while strictly protecting 
                    the <i>g_min</i> fairness floor.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_data:
    st.markdown("### 📋 Filtered Traffic Observation Records & Export")
    st.markdown(
        f"""
        Displaying **{len(filtered_df):,} records** from *{active_scenario_name}*.
        Includes calculated spatial density proxy, estimated flow rates, and 3-level traffic classifications.
        """
    )

    # Add recommended timings to export table for transparency
    export_df = filtered_df.copy()
    export_df["recommended_cycle_s"] = rec_plan.cycle_time_seconds
    export_df["recommended_phase1_green_s"] = rec_plan.phases[0].green_time_seconds
    export_df["recommended_phase2_green_s"] = rec_plan.phases[1].green_time_seconds
    export_df["simulation_notice"] = SIMULATION_DISCLAIMER

    st.dataframe(export_df, use_container_width=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Analytics Report (CSV)",
        data=csv_data,
        file_name="smart_traffic_analytics_report.csv",
        mime="text/csv",
    )

