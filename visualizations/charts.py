"""
Charts and Styling Module
Provides interactive Plotly visualizations and Dark Navy / Cyan CSS styling
for the Smart Traffic Management System.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


def get_custom_css() -> str:
    """
    Returns custom CSS to style Streamlit into a polished dark navy & cyan dashboard.
    Features glassmorphism cards, glowing cyan accents, modern typography,
    and responsive layout controls.
    """
    return """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-main: #0B1120;
        --bg-card: #0F172A;
        --bg-card-hover: #1E293B;
        --cyan-glow: #06B6D4;
        --cyan-light: #38BDF8;
        --cyan-dim: rgba(6, 182, 212, 0.15);
        --text-primary: #F8FAFC;
        --text-muted: #94A3B8;
        --border-color: #1E293B;
        --border-highlight: rgba(6, 182, 212, 0.4);
    }

    /* Streamlit Global Theme Overrides */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em !important;
    }

    h1 {
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 60%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.25s ease-in-out !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--cyan-glow) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -4px rgba(6, 182, 212, 0.25) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Custom CSS Card Component */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: all 0.25s ease-in-out;
    }

    .glass-card:hover {
        border-color: var(--border-highlight);
    }

    .glass-card-glow {
        border-left: 4px solid var(--cyan-glow);
    }

    /* Academic / Status Pill Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 8px;
    }

    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .badge-coral {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0F172A !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid #1E293B !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #22D3EE !important;
        font-weight: 600 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0F172A !important;
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
        color: #E2E8F0 !important;
    }

    /* Dataframe tables */
    [data-testid="stDataFrame"] {
        background-color: #0F172A !important;
        border-radius: 10px !important;
        border: 1px solid #1E293B !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #070D18 !important;
        border-right: 1px solid #1E293B !important;
    }
    </style>
    """


DARK_LAYOUT_TEMPLATE = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0B1120",
    font=dict(family="Outfit, sans-serif", color="#E2E8F0"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#1E293B",
        showline=True,
        linecolor="#334155",
        tickfont=dict(color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#1E293B",
        showline=True,
        linecolor="#334155",
        tickfont=dict(color="#94A3B8"),
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="#1E293B",
        borderwidth=1,
        font=dict(color="#E2E8F0"),
    ),
)


def create_network_graph_plot(
    graph: Union[nx.Graph, nx.DiGraph],
    shortest_path: Optional[List[str]] = None,
    bottleneck_edges: Optional[List[Tuple[str, str, int]]] = None,
    show_edge_labels: bool = True,
) -> go.Figure:
    """
    Renders an interactive Plotly visualization of the urban intersection road network.
    Highlights shortest path in glowing cyan, bottleneck cuts in coral red,
    and displays physical distances and dynamic congestion weights.
    Supports both directed (DiGraph) and undirected (Graph) modes.
    """
    fig = go.Figure()
    is_directed = graph.is_directed()

    # Extract node positions
    pos = {node: data["pos"] for node, data in graph.nodes(data=True)}

    # Convert shortest path nodes into an edge set for quick lookup
    path_edges = set()
    if shortest_path and len(shortest_path) > 1:
        for i in range(len(shortest_path) - 1):
            u_p, v_p = shortest_path[i], shortest_path[i + 1]
            path_edges.add((u_p, v_p))
            if not is_directed:
                path_edges.add((v_p, u_p))

    bottleneck_set = set()
    if bottleneck_edges:
        for u, v, _ in bottleneck_edges:
            bottleneck_set.add((u, v))
            if not is_directed:
                bottleneck_set.add((v, u))

    # Draw regular edges
    for u, v, data in graph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 1.0)
        length = data.get("length", 1.0)
        vehicles = data.get("vehicles", 10)
        capacity = data.get("capacity", 50)
        has_incident = data.get("has_incident", False) or weight >= 900.0

        is_in_path = (u, v) in path_edges or (not is_directed and (v, u) in path_edges)
        is_bottleneck = (u, v) in bottleneck_set or (not is_directed and (v, u) in bottleneck_set)

        if has_incident:
            edge_color = "#EF4444"  # Red blocked
            line_width = 3.5
            dash_style = "dot"
        elif is_in_path:
            edge_color = "#06B6D4"  # Cyan shortest path
            line_width = 4.5
            dash_style = "solid"
        elif is_bottleneck:
            edge_color = "#F59E0B"  # Amber bottleneck
            line_width = 3.5
            dash_style = "dash"
        else:
            # Color by volume-to-capacity
            vc = vehicles / max(1, capacity)
            if vc > 0.8:
                edge_color = "#E11D48"
            elif vc > 0.5:
                edge_color = "#F59E0B"
            else:
                edge_color = "#334155"
            line_width = 2.0
            dash_style = "solid"

        # Directed vs undirected connector symbol
        arrow_sym = " → " if is_directed else " ↔ "

        # Edge line
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=line_width, color=edge_color, dash=dash_style),
                hoverinfo="text",
                hovertext=(
                    f"Road: {u}{arrow_sym}{v}<br>"
                    f"Length: {length:.1f} km<br>"
                    f"Congestion Weight: {weight:.1f} min<br>"
                    f"Vehicles: {vehicles}/{capacity}<br>"
                    f"Incident: {'Blocked / Congested' if has_incident else 'Clear'}"
                ),
                showlegend=False,
            )
        )

        # Draw midpoint label showing distance and congestion weight
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        edge_label = f"{length}km | {weight:.1f}m" if show_edge_labels else ""
        symbol_choice = "arrow-up" if is_directed else "circle"

        fig.add_trace(
            go.Scatter(
                x=[mx],
                y=[my],
                mode="markers+text" if show_edge_labels else "markers",
                text=[edge_label],
                textposition="top center",
                textfont=dict(size=8, color="#94A3B8", family="JetBrains Mono, monospace"),
                marker=dict(size=6, color=edge_color, symbol=symbol_choice),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw nodes (Intersections)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        name = data.get("name", node)
        zone = data.get("zone", "General")
        deg = graph.degree(node)

        in_path = shortest_path and node in shortest_path
        color = "#06B6D4" if in_path else "#38BDF8"
        node_color.append(color)

        node_hover.append(
            f"<b>{node}: {name}</b><br>Zone: {zone}<br>Connections: {deg} roads<br>Degree: {deg}"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=13, family="Outfit, sans-serif"),
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(
                size=24,
                color=node_color,
                line=dict(color="#FFFFFF", width=2),
                symbol="circle",
            ),
            name="Intersections (V)",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Urban Road Network Graph G = (V, E)</b>",
        height=520,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


def create_conflict_graph_plot(conflict_data: Dict[str, Any]) -> go.Figure:
    """
    Renders the traffic movement Conflict Graph G_conflict.
    Vertices represent vehicular movements, and vertices are colored according
    to Welsh-Powell chromatic coloring (Phase 1, Phase 2, etc.).
    """
    node_colors_map = conflict_data.get("node_colors", {})
    palette = ["#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    # Generate circular layout for 8 movements
    movements = list(node_colors_map.keys())
    n = len(movements)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    radius = 10.0

    pos = {}
    for i, m in enumerate(movements):
        pos[m] = (radius * math.cos(angles[i]), radius * math.sin(angles[i]))

    # Conflict pairs
    from modules.discrete_math import ConflictGraphModel
    c_pairs = ConflictGraphModel.CONFLICT_PAIRS

    fig = go.Figure()

    # Draw conflict edges
    for u, v in c_pairs:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=1.2, color="rgba(148, 163, 184, 0.25)"),
                    hoverinfo="text",
                    hovertext=f"Conflict: {u} ↔ {v}",
                    showlegend=False,
                )
            )

    # Draw movement nodes
    nx_coords = [pos[m][0] for m in movements]
    ny_coords = [pos[m][1] for m in movements]
    labels = movements
    colors = [
        palette[(node_colors_map.get(m, 1) - 1) % len(palette)]
        for m in movements
    ]
    hovers = [
        f"<b>{m}</b><br>Assigned Phase: Phase {node_colors_map.get(m, 1)}<br>Color: {colors[i]}"
        for i, m in enumerate(movements)
    ]

    fig.add_trace(
        go.Scatter(
            x=nx_coords,
            y=ny_coords,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            textfont=dict(color="#0B1120", size=10, family="Outfit, sans-serif"),
            hoverinfo="text",
            hovertext=hovers,
            marker=dict(
                size=36,
                color=colors,
                line=dict(color="#FFFFFF", width=2),
            ),
            name="Movements",
            showlegend=False,
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title=f"<b>Welsh-Powell Conflict Graph (Chromatic Number χ = {conflict_data.get('chromatic_number', 0)})</b>",
        height=480,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


def create_traffic_timeline_plot(df: pd.DataFrame) -> go.Figure:
    """
    Renders time-series trends comparing vehicle volume and PCU load across time.
    """
    if df.empty or "timestamp" not in df.columns:
        return go.Figure()

    grouped = df.groupby("timestamp")[["total_vehicles", "pcu_total"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["timestamp"],
            y=grouped["total_vehicles"],
            name="Total Vehicles",
            mode="lines+markers",
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["timestamp"],
            y=grouped["pcu_total"],
            name="Passenger Car Units (PCU)",
            mode="lines+markers",
            line=dict(color="#06B6D4", width=2.5, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Intersection Traffic Flow Trend (Vehicles vs PCU)</b>",
        xaxis_title="Time of Observation",
        yaxis_title="Volume / Equivalent PCU",
        height=380,
    )
    return fig


def create_static_vs_adaptive_comparison_bar(df: pd.DataFrame) -> go.Figure:
    """
    Visualizes Delay Comparison: Static Fixed Signal vs AI Adaptive Controller.
    """
    if df.empty or "delay_seconds_static" not in df.columns:
        return go.Figure()

    grouped = (
        df.groupby("approach_lane")[["delay_seconds_static", "delay_seconds_adaptive"]]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["approach_lane"],
            y=grouped["delay_seconds_static"],
            name="Static Fixed-Time Delay (s)",
            marker_color="#EF4444",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped["approach_lane"],
            y=grouped["delay_seconds_adaptive"],
            name="AI Adaptive Signal Delay (s)",
            marker_color="#10B981",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Delay Reduction by Approach Lane (Static vs AI Adaptive)</b>",
        barmode="group",
        xaxis_title="Approach Corridor",
        yaxis_title="Average Vehicle Delay (seconds)",
        height=380,
    )
    return fig


def create_lane_distribution_donut(metrics_dict: Dict[str, Any]) -> go.Figure:
    """
    Renders donut chart breakdown of traffic distribution across approaches.
    """
    labels = list(metrics_dict.keys())
    values = [m.pcu_total for m in metrics_dict.values()]

    colors = ["#06B6D4", "#38BDF8", "#10B981", "#F59E0B"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0B1120", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Demand Split by Approach (PCU)</b>",
        height=320,
    )
    return fig


def create_congestion_gauge(congestion_index: float, lane_name: str) -> go.Figure:
    """
    Renders an engineering Level-of-Service (LOS) gauge for a specific lane.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=congestion_index,
            title={"text": f"<b>{lane_name} Congestion Index</b>", "font": {"color": "#E2E8F0", "size": 14}},
            number={"suffix": "%", "font": {"color": "#38BDF8", "size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                "bar": {"color": "#06B6D4"},
                "bgcolor": "#0B1120",
                "borderwidth": 1,
                "bordercolor": "#1E293B",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.75,
                    "value": 85,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0F172A",
        font=dict(family="Outfit, sans-serif", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
    )
    return fig


def create_queue_length_bar(metrics_dict: Dict[str, Any]) -> go.Figure:
    """
    Horizontal bar chart depicting current estimated queue lengths in meters.
    """
    lanes = list(metrics_dict.keys())
    queues = [m.queue_length_m for m in metrics_dict.values()]

    colors = []
    for q in queues:
        if q > 80:
            colors.append("#EF4444")
        elif q > 40:
            colors.append("#F59E0B")
        else:
            colors.append("#10B981")

    fig = go.Figure(
        go.Bar(
            y=lanes,
            x=queues,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0B1120", width=1)),
            text=[f"{q} m" for q in queues],
            textposition="auto",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Queue Length by Approach (Meters)</b>",
        xaxis_title="Queue Length (m)",
        yaxis_title="Approach Lane",
        height=280,
    )
    return fig


def create_signal_split_comparison_bar(
    current_p1_green: int,
    current_p2_green: int,
    rec_p1_green: int,
    rec_p2_green: int,
    clearance_time: int = 10,
) -> go.Figure:
    """
    Renders a horizontal stacked bar chart comparing signal phase allocations
    between Current Simulated Plan and Recommended Plan.
    """
    plans = ["Current Simulated Plan", "Recommended Plan"]
    p1_greens = [current_p1_green, rec_p1_green]
    p2_greens = [current_p2_green, rec_p2_green]
    clearances = [clearance_time, clearance_time]

    fig = go.Figure()

    # Phase 1 Green
    fig.add_trace(
        go.Bar(
            y=plans,
            x=p1_greens,
            name="Phase 1 (North-South) Green",
            orientation="h",
            marker=dict(color="#06B6D4"),
            text=[f"{v}s" for v in p1_greens],
            textposition="inside",
            insidetextanchor="middle",
        )
    )

    # Phase 2 Green
    fig.add_trace(
        go.Bar(
            y=plans,
            x=p2_greens,
            name="Phase 2 (East-West) Green",
            orientation="h",
            marker=dict(color="#3B82F6"),
            text=[f"{v}s" for v in p2_greens],
            textposition="inside",
            insidetextanchor="middle",
        )
    )

    # Clearance (Yellow + All-Red)
    fig.add_trace(
        go.Bar(
            y=plans,
            x=clearances,
            name="Clearance (Yellow + Red)",
            orientation="h",
            marker=dict(color="#F59E0B"),
            text=[f"{v}s" for v in clearances],
            textposition="inside",
            insidetextanchor="middle",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        barmode="stack",
        title="<b>Signal Cycle Split Allocation: Current vs Recommended (Seconds)</b>",
        xaxis_title="Total Cycle Duration (Seconds)",
        yaxis_title="Timing Schedule",
        height=260,
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#E2E8F0"),
        )
    )
    return fig


def create_lane_multimetric_bar(df: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart comparing Total Vehicles, PCU Load, and Density Proxy across approach corridors.
    """
    if df.empty or "approach_lane" not in df.columns:
        return go.Figure()

    agg_dict = {"total_vehicles": "mean", "pcu_total": "mean"}
    if "density_proxy" in df.columns:
        agg_dict["density_proxy"] = "mean"

    grouped = df.groupby("approach_lane").agg(agg_dict).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["approach_lane"],
            y=grouped["total_vehicles"].round(1),
            name="Mean Vehicles",
            marker_color="#38BDF8",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped["approach_lane"],
            y=grouped["pcu_total"].round(1),
            name="Mean PCU Load",
            marker_color="#06B6D4",
        )
    )
    if "density_proxy" in grouped.columns:
        fig.add_trace(
            go.Bar(
                x=grouped["approach_lane"],
                y=grouped["density_proxy"].round(1),
                name="Density Proxy (veh/area)",
                marker_color="#10B981",
            )
        )

    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Approach Lane Demand & Density Comparison</b>",
        barmode="group",
        xaxis_title="Approach Corridor",
        yaxis_title="Metric Magnitude",
        height=360,
    )
    return fig


def create_traffic_state_donut(df: pd.DataFrame) -> go.Figure:
    """
    Renders donut chart visualizing the proportion of traffic states (Low, Moderate, High).
    """
    if df.empty:
        return go.Figure()

    col = "traffic_state" if "traffic_state" in df.columns else ("congestion_level" if "congestion_level" in df.columns else None)
    if not col:
        return go.Figure()

    counts = df[col].value_counts().to_dict()
    # Normalize labels
    mapped_counts = {"Low": 0, "Moderate": 0, "High": 0}
    for k, v in counts.items():
        k_str = str(k).strip()
        if k_str in ["Low", "Free Flow"]:
            mapped_counts["Low"] += v
        elif k_str in ["Moderate", "Medium"]:
            mapped_counts["Moderate"] += v
        elif k_str in ["High", "Severe", "Severe Gridlock"]:
            mapped_counts["High"] += v
        else:
            mapped_counts["Moderate"] += v

    labels = [k for k, v in mapped_counts.items() if v > 0]
    values = [mapped_counts[k] for k in labels]
    colors = {
        "Low": "#10B981",
        "Moderate": "#F59E0B",
        "High": "#EF4444",
    }
    pie_colors = [colors.get(l, "#38BDF8") for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=pie_colors),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT_TEMPLATE,
        title="<b>Traffic State Classification Distribution</b>",
        height=340,
    )
    return fig

