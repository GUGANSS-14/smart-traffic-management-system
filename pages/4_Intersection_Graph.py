"""
Page 4: Intersection Graph & Discrete Mathematics Network Modeling
Models urban intersections and connected road networks as discrete graphs G = (V, E).
Features:
- Directed and Undirected graph representations
- Four-way intersection model and 6-junction connected urban network
- Binary and weighted adjacency matrices A in {0, 1}^(n x n)
- Vertex degrees (in-degree, out-degree, total degree) and Handshaking Lemma
- Connectivity analysis (connected components, strongly/weakly connected, bridges, cut vertices)
- Breadth-First Search (BFS) vs Dijkstra's Algorithm (with strict non-negative weight validation)
- Visual road-network graph with edge labels, distances, and congestion weights
- Real-time route calculation between user-selected start and destination
- Graph modification demonstration (traffic congestion impedance via BPR formula and road closures)
- Student-friendly explanations of vertices, edges, adjacency matrices, paths, connectivity, and route costs
- Explicit academic simulation labeling
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from modules.graph_model import IntersectionGraphNetwork, FourWayIntersectionGraph, SIMULATION_DISCLAIMER
from visualizations.charts import (
    get_custom_css,
    create_network_graph_plot,
    DARK_LAYOUT_TEMPLATE,
)

# Page Setup
st.set_page_config(
    page_title="Intersection Graph - Smart Traffic",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🕸️ Urban Road Network & Intersection Graph G = (V, E)")
st.markdown(
    """
    **Discrete Mathematics Applications in Intelligent Transportation Systems**  
    *Represents intersections as vertices $V$, road corridors as edges $E$, and evaluates shortest paths via BFS and Dijkstra's algorithm.*
    """
)

# Prominent Academic Simulation Disclaimer Callout
st.markdown(
    f"""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.4rem;">⚠️</span>
            <div style="font-size:0.85rem; color:#FDE68A; line-height:1.5;">
                <b>ACADEMIC SIMULATION NOTICE:</b> {SIMULATION_DISCLAIMER}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# State Management for Graph Network
if "city_network" not in st.session_state:
    st.session_state.city_network = IntersectionGraphNetwork(is_directed=True, network_type="connected_grid")

network: IntersectionGraphNetwork = st.session_state.city_network

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

    st.markdown("### 🕸️ Graph Representation Settings")

    network_type_choice = st.radio(
        "Network Topology",
        [
            "Connected Urban Grid (6 Junctions J1-J6)",
            "Four-Way Intersection Detail (North, South, East, West Arms & Center)",
        ],
    )
    selected_net_type = "four_way_intersection" if "Four-Way" in network_type_choice else "connected_grid"

    mode_choice = st.radio(
        "Graph Directionality Mode",
        [
            "Directed Graph (DiGraph: Ordered Pairs u → v)",
            "Undirected Graph (Graph: Symmetric Two-Way u ↔ v)",
        ],
    )
    is_directed_selected = "Directed" in mode_choice

    # Update network topology or mode if changed
    if network.network_type != selected_net_type or network.is_directed != is_directed_selected:
        network.set_network_type(selected_net_type)
        network.set_directed_mode(is_directed_selected)
        st.session_state.city_network = network

    st.markdown("---")
    st.markdown("### 🧭 Route Planning & Navigation")
    nodes_list = sorted(list(network.graph.nodes()))

    if len(nodes_list) >= 2:
        source_node = st.selectbox("Origin Intersection (Source s)", nodes_list, index=0)
        dest_candidates = [n for n in nodes_list if n != source_node]
        default_target_idx = len(dest_candidates) - 1 if dest_candidates else 0
        target_node = st.selectbox(
            "Destination Intersection (Target t)",
            dest_candidates if dest_candidates else nodes_list,
            index=default_target_idx,
        )
    else:
        source_node = nodes_list[0] if nodes_list else "J1"
        target_node = nodes_list[0] if nodes_list else "J1"

    routing_algo = st.selectbox(
        "Routing Algorithm",
        [
            "Dijkstra's Algorithm (Fastest Dynamic Travel Time, w ≥ 0)",
            "Dijkstra's Algorithm (Shortest Physical Distance in km, w ≥ 0)",
            "Breadth-First Search (BFS: Minimum Number of Edge Hops)",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚠️ Dynamic Graph Modification")
    st.markdown("Inject congestion or road closures to observe real-time algorithmic rerouting:")

    edge_options = [f"{u} → {v}" if network.is_directed else f"{u} ↔ {v}" for u, v in network.graph.edges()]
    if edge_options:
        default_edge_idx = min(14, len(edge_options) - 1)
        chosen_edge = st.selectbox("Target Road Segment Edge", edge_options, index=default_edge_idx)

        separator = " → " if " → " in chosen_edge else " ↔ "
        u_sel, v_sel = chosen_edge.split(separator)

        incident_active = st.checkbox("Simulate Severe Road Blockage / Closure", value=False)
        veh_slider = st.slider("Vehicles on Segment (Volume V)", min_value=0, max_value=95, value=85 if incident_active else 20)

        # Apply modification to the graph
        network.update_road_traffic(u_sel, v_sel, vehicle_count=veh_slider, incident=incident_active)

    if st.button("🔄 Reset Network to Default Flow", use_container_width=True):
        network.build_network()
        st.rerun()

# Execute Route Calculation
if "Breadth-First" in routing_algo:
    route_result = network.breadth_first_search(source_node, target_node)
    is_bfs = True
    weight_metric_name = "Edge Hops"
else:
    is_bfs = False
    weight_key_choice = "length" if "Physical Distance" in routing_algo else "weight"
    weight_metric_name = "km" if weight_key_choice == "length" else "min"
    try:
        route_result = network.dijkstra_shortest_path(source_node, target_node, weight_key=weight_key_choice)
    except ValueError as ve:
        route_result = {"success": False, "error": str(ve)}

# Connectivity Assessment
conn_info = network.get_connectivity_analysis()

# Top KPI Metric Cards
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)

with kpi_c1:
    st.metric("Vertices |V|", f"{network.graph.number_of_nodes()}", "Intersections")

with kpi_c2:
    st.metric("Edges |E|", f"{network.graph.number_of_edges()}", "Road Corridors")

with kpi_c3:
    if route_result.get("success"):
        path_str = " → ".join(route_result["path"])
        st.metric("Optimal Route", path_str, f"{source_node} to {target_node}")
    else:
        st.metric("Optimal Route", "No Path", "Disconnected")

with kpi_c4:
    if route_result.get("success"):
        if is_bfs:
            st.metric("Path Hops", f"{route_result.get('hop_count', 0)} hops", "Unweighted BFS")
        else:
            cost_val = route_result.get("total_cost", 0.0)
            unit_lbl = "km" if "Distance" in routing_algo else "min"
            st.metric("Route Cost", f"{cost_val} {unit_lbl}", "Accumulated Weight")
    else:
        st.metric("Route Cost", "N/A", "Unreachable")

with kpi_c5:
    dist_val = route_result.get("total_distance_km", 0.0) if route_result.get("success") else 0.0
    st.metric("Physical Length", f"{dist_val} km", "Road Distance")

with kpi_c6:
    if network.is_directed:
        conn_label = "Strongly Connected" if conn_info.get("is_strongly_connected") else ("Weakly Connected" if conn_info.get("is_weakly_connected") else "Disconnected")
    else:
        conn_label = "Connected" if conn_info.get("is_connected") else "Disconnected"
    st.metric("Connectivity", conn_label, "Graph Invariant")

st.write("")

# Main View: Network Graph Visualization & Route Telemetry
col_graph, col_telemetry = st.columns([1.3, 1.0])

with col_graph:
    active_path = route_result.get("path", []) if route_result.get("success") else None

    # Ford-Fulkerson cut bottleneck if 6-junction grid
    bottleneck_edges = []
    if network.network_type == "connected_grid" and source_node in network.graph and target_node in network.graph:
        flow_res = network.compute_max_flow_bottleneck(source_node, target_node)
        bottleneck_edges = flow_res.get("bottleneck_cut_edges", [])

    fig_net = create_network_graph_plot(
        network.graph,
        shortest_path=active_path,
        bottleneck_edges=bottleneck_edges,
        show_edge_labels=True,
    )
    st.plotly_chart(fig_net, use_container_width=True)

with col_telemetry:
    st.markdown("### 📍 Path Optimization & Algorithm Trace")

    if route_result.get("success"):
        algo_title = route_result.get("algorithm", "Shortest Path Algorithm")
        badge_style = "badge-cyan" if "Dijkstra" in algo_title else "badge-emerald"

        st.markdown(
            f"""
            <div class="glass-card glass-card-glow">
                <span class="badge {badge_style}">{algo_title}</span>
                <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; margin:8px 0;">
                    {' → '.join(route_result['path'])}
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; line-height:1.6;">
                    • <b>Hops Traversed:</b> <span style="color:#38BDF8;">{route_result.get('hop_count', 0)} road segments</span><br>
                    • <b>Physical Distance:</b> <span style="color:#E2E8F0;">{route_result.get('total_distance_km', 0.0)} km</span><br>
                    • <b>Route Cost:</b> <span style="color:#22D3EE; font-weight:700;">{route_result.get('total_cost', 0.0)} {weight_metric_name}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 Step-by-Step Algorithm Execution Trace", expanded=True):
            if "step_log" in route_result and route_result["step_log"]:
                for step_txt in route_result["step_log"][:6]:
                    st.markdown(f"- `{step_txt}`")
                if len(route_result["step_log"]) > 6:
                    st.caption(f"... and {len(route_result['step_log']) - 6} more relaxation steps.")
            elif "visited_order" in route_result:
                st.markdown(f"**FIFO Queue Discovery Sequence:** `{ ' → '.join(route_result['visited_order']) }`")
                st.markdown(f"**Hop Distances from Source:** `{route_result.get('hop_distances', {})}`")

    else:
        st.error(f"❌ Route Unreachable: {route_result.get('error', 'No path exists.')}")

    # BPR Equation Callout
    st.markdown(
        """
        <div class="glass-card" style="margin-top:10px;">
            <h5 style="color:#38BDF8; margin-top:0;">🏎️ Bureau of Public Roads (BPR) Impedance Function</h5>
            <p style="color:#CBD5E1; font-size:0.82rem; line-height:1.5;">
                Edge weight represents dynamic travel time impedance:
            </p>
            <div style="background:#070D18; padding:8px; border-radius:6px; font-family:monospace; color:#22D3EE; font-size:0.85rem; text-align:center;">
                t = t₀ × [ 1 + 0.15 × (V / C)⁴ ]
            </div>
            <p style="color:#94A3B8; font-size:0.75rem; margin-top:6px; line-height:1.4;">
                Where <i>t₀</i> = free-flow travel time, <i>V</i> = vehicle volume, and <i>C</i> = corridor capacity. 
                As <i>V</i> approaches <i>C</i>, edge weight escalates non-linearly, compelling Dijkstra's algorithm to route traffic around congestion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# In-Depth Discrete Mathematics Tabs
st.markdown("### 🔢 Discrete Mathematics Graph Theoretical Formulations")

tab_matrices, tab_degrees, tab_comparison, tab_glossary = st.tabs([
    "🔢 Adjacency & Cost Matrices",
    "📊 Degrees & Graph Connectivity",
    "⚖️ BFS vs Dijkstra Comparison",
    "📘 Student Discrete Math Guide",
])

with tab_matrices:
    st.markdown(r"#### 1. Binary Adjacency Matrix $A \in \{0, 1\}^{n \times n}$")
    st.markdown(
        r"""
        For a graph with vertices $V = \{v_1, v_2, \dots, v_n\}$, entry $A_{i, j} = 1$ if an edge exists from $v_i$ to $v_j$, else $0$.  
        *Property:* In an undirected graph, $A$ is strictly symmetric ($A = A^T$). In a directed graph, asymmetry indicates one-way corridors.
        """
    )
    df_adj = network.get_adjacency_matrix()
    st.dataframe(df_adj, use_container_width=True)

    c_w1, c_w2 = st.columns(2)
    with c_w1:
        st.markdown("#### 2. Weighted Cost Matrix $W$ (Dynamic Travel Time in Minutes)")
        df_weights = network.get_travel_time_matrix()
        st.dataframe(df_weights, use_container_width=True)

    with c_w2:
        st.markdown("#### 3. Physical Distance Matrix $D$ (Kilometers)")
        df_dist = network.get_distance_matrix()
        st.dataframe(df_dist, use_container_width=True)

    st.markdown(
        """
        <div class="glass-card">
            <h5 style="color:#38BDF8; margin-top:0;">💡 Discrete Math Insight: Matrix Powers & Paths</h5>
            <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.5;">
                By the fundamental theorem of adjacency matrices, the entry <b>(Aᵏ)ᵢ,ⱼ</b> in the <i>k</i>-th power of the adjacency matrix
                gives the exact number of distinct walks of length <b>k</b> between vertex <i>i</i> and vertex <i>j</i>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_degrees:
    st.markdown("#### 📊 Vertex Degrees & The Handshaking Lemma")
    df_degrees = network.get_vertex_degrees()
    st.dataframe(df_degrees, use_container_width=True, hide_index=True)

    num_e = network.graph.number_of_edges()
    if network.is_directed:
        sum_in = df_degrees["In-Degree (d⁻)"].sum()
        sum_out = df_degrees["Out-Degree (d⁺)"].sum()
        handshake_text = f"Directed Degree Invariant: ∑ d⁻(v) = ∑ d⁺(v) = |E| = {num_e} (Verified: {sum_in} = {sum_out} = {num_e})."
    else:
        sum_deg = df_degrees["Degree (d)"].sum()
        handshake_text = f"Euler's Handshaking Lemma: ∑ deg(v) = 2|E| = {2 * num_e} (Verified: {sum_deg} = {2 * num_e})."

    st.markdown(
        f"""
        <div style="background:#070D18; border-left:3px solid #10B981; border-radius:6px; padding:10px 14px; font-size:0.85rem; color:#A7F3D0; margin-bottom:14px;">
            <b>Handshaking Theorem Verification:</b> {handshake_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🛡️ Graph Connectivity & Resilience")
    c_res1, c_res2 = st.columns(2)

    with c_res1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h5 style="color:#38BDF8; margin-top:0;">🔗 Connected Components</h5>
                <p style="color:#CBD5E1; font-size:0.85rem;">
                    • <b>Directional Mode:</b> {'Directed (DiGraph)' if network.is_directed else 'Undirected (Graph)'}<br>
                    • <b>Strongly Connected:</b> {'Yes ✅' if conn_info.get('is_strongly_connected') else 'No ❌'}<br>
                    • <b>Weakly Connected:</b> {'Yes ✅' if conn_info.get('is_weakly_connected') else 'No ❌'}<br>
                    • <b>Component Sets:</b> {conn_info.get('components', conn_info.get('strongly_connected_components', []))}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_res2:
        bridges_list = conn_info.get("bridges", [])
        cuts_list = conn_info.get("cut_vertices", [])
        bridges_str = ", ".join([f"{u} ↔ {v}" for u, v in bridges_list]) if bridges_list else "None (Resilient Ring)"
        cuts_str = ", ".join(cuts_list) if cuts_list else "None"

        st.markdown(
            f"""
            <div class="glass-card">
                <h5 style="color:#F59E0B; margin-top:0;">🌉 Bridges & Cut Vertices</h5>
                <p style="color:#CBD5E1; font-size:0.85rem;">
                    • <b>Bridge Edges (Cut Edges):</b> {bridges_str}<br>
                    • <b>Cut Vertices (Articulation Points):</b> {cuts_str}<br>
                    <span style="font-size:0.75rem; color:#94A3B8;">
                        A bridge is a critical road whose removal disconnects the graph into disjoint components.
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_comparison:
    st.markdown("### ⚖️ Algorithmic Comparison: Breadth-First Search (BFS) vs. Dijkstra")
    st.markdown(
        """
        Both algorithms find shortest paths, but they operate under fundamentally different discrete assumptions:
        """
    )

    comp_table = pd.DataFrame([
        {
            "Property": "Core Objective",
            "Breadth-First Search (BFS)": "Minimizes number of edge hops (unweighted shortest path)",
            "Dijkstra's Algorithm": "Minimizes cumulative edge weights (travel time / distance)",
        },
        {
            "Property": "Underlying Data Structure",
            "Breadth-First Search (BFS)": "First-In-First-Out (FIFO) Queue",
            "Dijkstra's Algorithm": "Priority Queue (Min-Heap keyed on tentative distance)",
        },
        {
            "Property": "Time Complexity",
            "Breadth-First Search (BFS)": "O(|V| + |E|)",
            "Dijkstra's Algorithm": "O((|V| + |E|) log |V|) with binary heap",
        },
        {
            "Property": "Edge Weight Requirement",
            "Breadth-First Search (BFS)": "Uniform unit weights w = 1",
            "Dijkstra's Algorithm": "Strictly non-negative weights: w(e) ≥ 0",
        },
        {
            "Property": "Negative Weight Behavior",
            "Breadth-First Search (BFS)": "Not applicable (ignores weights)",
            "Dijkstra's Algorithm": "FAILS! Greedy assumption broken; requires Bellman-Ford",
        },
        {
            "Property": "Traffic Domain Application",
            "Breadth-First Search (BFS)": "Fewest turn maneuvers or signal stops",
            "Dijkstra's Algorithm": "Fastest total travel time avoiding traffic queues",
        },
    ])
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="glass-card">
            <h5 style="color:#EF4444; margin-top:0;">⚠️ Why Dijkstra Strictly Requires Non-Negative Weights (w ≥ 0)</h5>
            <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                Dijkstra's algorithm relies on a <b>greedy invariant</b>: once a vertex <i>u</i> is extracted from the min-heap,
                its shortest distance <i>d(u)</i> is finalized and will never decrease. 
                If negative edge weights existed, a previously finalized path could become cheaper via a negative cycle or backward shortcut,
                causing Dijkstra to compute incorrect routes or loop infinitely. For graphs with negative weights, 
                the dynamic programming <b>Bellman-Ford algorithm</b> (O(|V| · |E|)) must be utilized.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_glossary:
    st.markdown("### 📘 Student Discrete Mathematics Concepts & Definitions")
    concepts_dict = network.explain_discrete_math_concepts()

    for c_id, c_info in concepts_dict.items():
        with st.expander(f"📌 {c_info['concept']}"):
            st.markdown(
                f"""
                <div style="font-size:0.9rem; color:#F8FAFC; margin-bottom:8px;">
                    <b>Mathematical Definition:</b> {c_info['definition']}
                </div>
                <div style="font-size:0.85rem; color:#38BDF8;">
                    <b>Transportation Engineering Application:</b> {c_info['traffic_mapping']}
                </div>
                """,
                unsafe_allow_html=True,
            )

