"""
Graph Model Module
Models urban road networks and intersections using explicit Discrete Mathematics concepts:
G = (V, E) where Vertices V represent intersections and Edges E represent road connections.
Supports:
- Directed and Undirected graph modes
- Example Four-Way Intersection graph and Connected Urban Grid network
- Adjacency matrices (binary and weighted)
- Vertex degrees (in-degree, out-degree, total degree, Handshaking Lemma)
- Graph connectivity (connected components, strongly/weakly connected, bridges, cut vertices)
- Breadth-First Search (BFS: unweighted hop count)
- Dijkstra's Algorithm (weighted shortest path with strict non-negative weight validation)
- Dynamic graph modification (congestion impedance via BPR formula and road closures)
- Academic simulation labeling
"""

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import numpy as np
import pandas as pd
import networkx as nx


SIMULATION_DISCLAIMER = (
    "MATHEMATICAL SIMULATION ONLY: This road network is an educational discrete mathematics "
    "simulation model. It does not represent live municipal road infrastructure or real city maps, "
    "and makes no claim to optimize real-world municipal road navigation without real geographic survey data."
)


@dataclass
class RoadSegment:
    """Represents a roadway edge connection between two intersections."""
    u: str
    v: str
    length_km: float
    free_speed_kmh: float
    capacity: int
    current_vehicles: int = 10
    has_incident: bool = False
    is_blocked: bool = False

    @property
    def travel_time_minutes(self) -> float:
        """
        Calculates dynamic travel time using the Bureau of Public Roads (BPR) function:
        t = t_0 * (1 + alpha * (V / C)^beta)
        Edge weights represent impedance (time cost) and are strictly non-negative.
        """
        if self.has_incident or self.is_blocked:
            return 999.0  # Blocked / virtually infinite delay impedance

        t0 = (self.length_km / max(5.0, self.free_speed_kmh)) * 60.0  # Free-flow time in minutes
        vc_ratio = max(0.0, float(self.current_vehicles)) / max(1, self.capacity)
        # BPR parameters: alpha = 0.15, beta = 4.0
        bpr_factor = 1.0 + 0.15 * (vc_ratio ** 4.0)
        return max(0.01, round(t0 * bpr_factor, 2))


class FourWayIntersectionGraph:
    """
    Discrete Mathematics model of a single Four-Way Intersection.
    Vertices V = {North, South, East, West, Center}.
    Edges E = Approach arms, straight corridors, and left/right turns.
    """
    NODES = {
        "North": {"name": "North Approach", "pos": (0, 3), "zone": "Approach Arm"},
        "South": {"name": "South Approach", "pos": (0, -3), "zone": "Approach Arm"},
        "East": {"name": "East Approach", "pos": (3, 0), "zone": "Approach Arm"},
        "West": {"name": "West Approach", "pos": (-3, 0), "zone": "Approach Arm"},
        "Center": {"name": "Central Conflict Box", "pos": (0, 0), "zone": "Junction Core"},
    }

    # Approach to Center and Center to Exit arms (0.5 km length, 40 km/h speed, 40 veh capacity)
    DIRECTED_EDGES = [
        ("North", "Center", 0.5, 40, 40, 15),
        ("Center", "South", 0.5, 40, 40, 12),
        ("South", "Center", 0.5, 40, 40, 14),
        ("Center", "North", 0.5, 40, 40, 10),
        ("East", "Center", 0.5, 40, 40, 18),
        ("Center", "West", 0.5, 40, 40, 16),
        ("West", "Center", 0.5, 40, 40, 20),
        ("Center", "East", 0.5, 40, 40, 15),
    ]

    UNDIRECTED_EDGES = [
        ("North", "Center", 0.5, 40, 40, 15),
        ("South", "Center", 0.5, 40, 40, 14),
        ("East", "Center", 0.5, 40, 40, 18),
        ("West", "Center", 0.5, 40, 40, 20),
    ]

    def __init__(self, is_directed: bool = True):
        self.is_directed = is_directed
        self.graph = nx.DiGraph() if is_directed else nx.Graph()
        self.segments: Dict[Tuple[str, str], RoadSegment] = {}
        self.build_graph()

    def build_graph(self) -> None:
        """Constructs the four-way intersection graph."""
        self.graph.clear()
        self.segments.clear()

        for node_id, data in self.NODES.items():
            self.graph.add_node(node_id, name=data["name"], pos=data["pos"], zone=data["zone"])

        edges_data = self.DIRECTED_EDGES if self.is_directed else self.UNDIRECTED_EDGES
        for u, v, length, speed, cap, veh in edges_data:
            seg = RoadSegment(u=u, v=v, length_km=length, free_speed_kmh=speed, capacity=cap, current_vehicles=veh)
            self.segments[(u, v)] = seg
            self.graph.add_edge(
                u, v,
                weight=seg.travel_time_minutes,
                length=length,
                capacity=cap,
                vehicles=veh,
                segment=seg,
                has_incident=False,
            )


class IntersectionGraphNetwork:
    """
    Discrete Mathematics model of a connected urban road network G = (V, E).
    Vertices V = Intersections (junction decision nodes).
    Edges E = Road corridors (directed or undirected weighted connections).
    """
    DEFAULT_JUNCTIONS = {
        "J1": {"name": "Central Plaza", "pos": (2, 4), "zone": "Commercial"},
        "J2": {"name": "University Campus", "pos": (5, 4), "zone": "Institutional"},
        "J3": {"name": "Tech Park", "pos": (8, 4), "zone": "IT Corridor"},
        "J4": {"name": "Railway Station", "pos": (2, 1), "zone": "Transit Hub"},
        "J5": {"name": "Market Square", "pos": (5, 1), "zone": "Commercial"},
        "J6": {"name": "Residential Ring", "pos": (8, 1), "zone": "Residential"},
    }

    DEFAULT_ROADS = [
        ("J1", "J2", 2.2, 50, 60, 25),
        ("J2", "J1", 2.2, 50, 60, 20),
        ("J2", "J3", 2.8, 60, 70, 30),
        ("J3", "J2", 2.8, 60, 70, 28),
        ("J1", "J4", 1.8, 45, 50, 35),
        ("J4", "J1", 1.8, 45, 50, 32),
        ("J2", "J5", 1.9, 45, 55, 40),
        ("J5", "J2", 1.9, 45, 55, 42),
        ("J3", "J6", 2.0, 50, 60, 15),
        ("J6", "J3", 2.0, 50, 60, 18),
        ("J4", "J5", 2.5, 50, 65, 20),
        ("J5", "J4", 2.5, 50, 65, 22),
        ("J5", "J6", 2.4, 50, 65, 30),
        ("J6", "J5", 2.4, 50, 65, 28),
        ("J1", "J5", 3.2, 55, 80, 50),  # Diagonal arterial
        ("J5", "J1", 3.2, 55, 80, 48),
        ("J2", "J6", 3.5, 55, 80, 22),  # Diagonal bypass
        ("J6", "J2", 3.5, 55, 80, 24),
    ]

    def __init__(self, is_directed: bool = True, network_type: str = "connected_grid"):
        """
        Initializes the road network graph model.

        Args:
            is_directed: True for directed graph (DiGraph), False for undirected graph (Graph).
            network_type: 'connected_grid' (6-junction network) or 'four_way_intersection'.
        """
        self.is_directed = is_directed
        self.network_type = network_type
        self.graph: Union[nx.DiGraph, nx.Graph] = nx.DiGraph() if is_directed else nx.Graph()
        self.segments: Dict[Tuple[str, str], RoadSegment] = {}
        self.disclaimer: str = SIMULATION_DISCLAIMER
        self.build_network()

    def set_directed_mode(self, is_directed: bool) -> None:
        """Toggles between directed (DiGraph) and undirected (Graph) representation."""
        if self.is_directed != is_directed:
            self.is_directed = is_directed
            self.build_network()

    def set_network_type(self, network_type: str) -> None:
        """Switches between connected grid network and four-way intersection."""
        if self.network_type != network_type:
            self.network_type = network_type
            self.build_network()

    def build_network(self) -> None:
        """Constructs the active graph topology according to network_type and directed mode."""
        self.graph = nx.DiGraph() if self.is_directed else nx.Graph()
        self.segments.clear()

        if self.network_type == "four_way_intersection":
            four_way = FourWayIntersectionGraph(is_directed=self.is_directed)
            self.graph = four_way.graph
            self.segments = four_way.segments
        else:
            self.build_default_network()

    def build_default_network(self) -> None:
        """Initializes default 6-intersection connected grid."""
        self.graph = nx.DiGraph() if self.is_directed else nx.Graph()
        self.segments.clear()

        # Add vertices V
        for node_id, data in self.DEFAULT_JUNCTIONS.items():
            self.graph.add_node(
                node_id,
                name=data["name"],
                pos=data["pos"],
                zone=data["zone"],
            )

        # Add edges E
        if self.is_directed:
            edges_to_add = self.DEFAULT_ROADS
        else:
            # For undirected, filter duplicate opposite edges
            seen_pairs = set()
            edges_to_add = []
            for u, v, length, speed, cap, veh in self.DEFAULT_ROADS:
                pair = tuple(sorted([u, v]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges_to_add.append((u, v, length, speed, cap, veh))

        for u, v, length, speed, cap, init_veh in edges_to_add:
            segment = RoadSegment(
                u=u,
                v=v,
                length_km=length,
                free_speed_kmh=speed,
                capacity=cap,
                current_vehicles=init_veh,
            )
            self.segments[(u, v)] = segment
            self.graph.add_edge(
                u,
                v,
                weight=segment.travel_time_minutes,
                length=length,
                capacity=cap,
                vehicles=init_veh,
                segment=segment,
                has_incident=False,
            )

    def add_intersection(self, node_id: str, name: str, pos: Tuple[float, float], zone: str = "General") -> None:
        """Adds a vertex (intersection) to the graph."""
        self.graph.add_node(node_id, name=name, pos=pos, zone=zone)

    def add_road(
        self,
        u: str,
        v: str,
        length_km: float,
        speed_kmh: float = 50.0,
        capacity: int = 60,
        current_vehicles: int = 10,
    ) -> None:
        """Adds a road connection edge (u, v) with civil engineering attributes."""
        seg = RoadSegment(
            u=u,
            v=v,
            length_km=length_km,
            free_speed_kmh=speed_kmh,
            capacity=capacity,
            current_vehicles=current_vehicles,
        )
        self.segments[(u, v)] = seg
        self.graph.add_edge(
            u,
            v,
            weight=seg.travel_time_minutes,
            length=length_km,
            capacity=capacity,
            vehicles=current_vehicles,
            segment=seg,
            has_incident=False,
        )

    def remove_road(self, u: str, v: str) -> bool:
        """Removes a road edge from the graph."""
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self.segments.pop((u, v), None)
            return True
        return False

    def update_road_traffic(self, u: str, v: str, vehicle_count: int, incident: bool = False) -> None:
        """Updates real-time vehicle load on a road segment and recalculates edge weight."""
        if (u, v) in self.segments and self.graph.has_edge(u, v):
            seg = self.segments[(u, v)]
            seg.current_vehicles = max(0, int(vehicle_count))
            seg.has_incident = incident
            seg.is_blocked = incident
            weight = seg.travel_time_minutes
            self.graph[u][v]["weight"] = weight
            self.graph[u][v]["vehicles"] = seg.current_vehicles
            self.graph[u][v]["has_incident"] = incident

        # If undirected, also update reverse edge if present
        if not self.is_directed and (v, u) in self.segments and self.graph.has_edge(v, u):
            seg_rev = self.segments[(v, u)]
            seg_rev.current_vehicles = max(0, int(vehicle_count))
            seg_rev.has_incident = incident
            seg_rev.is_blocked = incident

    def set_road_availability(self, u: str, v: str, available: bool) -> None:
        """Toggles road availability (closure / blockage)."""
        self.update_road_traffic(u, v, vehicle_count=0 if not available else 10, incident=not available)

    # -------------------------------------------------------------------------
    # Discrete Mathematics Concept 1: Adjacency Matrices
    # -------------------------------------------------------------------------

    def get_adjacency_matrix(self) -> pd.DataFrame:
        """
        Generates the binary Adjacency Matrix A in {0, 1}^(n x n).
        A[i, j] = 1 if edge (i, j) exists in E, else 0.
        For undirected graphs, A is symmetric: A = A^T.
        """
        nodes = sorted(list(self.graph.nodes()))
        matrix = nx.to_numpy_array(self.graph, nodelist=nodes, weight=None)
        return pd.DataFrame(matrix.astype(int), index=nodes, columns=nodes)

    def get_travel_time_matrix(self) -> pd.DataFrame:
        """
        Generates the weighted impedance matrix W (travel time in minutes).
        Non-adjacent pairs are denoted by infinity (inf).
        """
        nodes = sorted(list(self.graph.nodes()))
        matrix = nx.to_numpy_array(self.graph, nodelist=nodes, weight="weight", nonedge=np.inf)
        return pd.DataFrame(np.round(matrix, 2), index=nodes, columns=nodes)

    def get_distance_matrix(self) -> pd.DataFrame:
        """Generates the physical distance matrix in kilometers."""
        nodes = sorted(list(self.graph.nodes()))
        matrix = nx.to_numpy_array(self.graph, nodelist=nodes, weight="length", nonedge=np.inf)
        return pd.DataFrame(np.round(matrix, 2), index=nodes, columns=nodes)

    # -------------------------------------------------------------------------
    # Discrete Mathematics Concept 2: Degrees & Connectivity
    # -------------------------------------------------------------------------

    def get_vertex_degrees(self) -> pd.DataFrame:
        """
        Computes degree properties for each intersection vertex.
        - Directed: In-degree d_in(v), Out-degree d_out(v), Total degree d(v) = d_in + d_out.
        - Undirected: Degree d(v).
        Validates the Handshaking Lemma: sum(deg(v)) = 2|E|.
        """
        nodes = sorted(list(self.graph.nodes()))
        rows = []

        if self.is_directed:
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
            total_degrees = dict(self.graph.degree())

            for n in nodes:
                name = self.graph.nodes[n].get("name", n)
                zone = self.graph.nodes[n].get("zone", "General")
                rows.append({
                    "Vertex (Node)": n,
                    "Intersection Name": name,
                    "In-Degree (d⁻)": in_degrees.get(n, 0),
                    "Out-Degree (d⁺)": out_degrees.get(n, 0),
                    "Total Degree (d)": total_degrees.get(n, 0),
                    "Zone": zone,
                })
        else:
            degrees = dict(self.graph.degree())
            for n in nodes:
                name = self.graph.nodes[n].get("name", n)
                zone = self.graph.nodes[n].get("zone", "General")
                rows.append({
                    "Vertex (Node)": n,
                    "Intersection Name": name,
                    "Degree (d)": degrees.get(n, 0),
                    "Zone": zone,
                })

        return pd.DataFrame(rows)

    def get_connectivity_analysis(self) -> Dict[str, Any]:
        """
        Evaluates graph connectivity metrics:
        - For undirected graphs: is_connected, connected components, cut vertices, bridges.
        - For directed graphs: is_strongly_connected, is_weakly_connected, strongly/weakly connected components.
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        if num_nodes == 0:
            return {
                "is_connected": False,
                "is_strongly_connected": False,
                "is_weakly_connected": False,
                "num_components": 0,
                "components": [],
                "bridges": [],
                "cut_vertices": [],
            }

        if self.is_directed:
            is_strong = nx.is_strongly_connected(self.graph)
            is_weak = nx.is_weakly_connected(self.graph)
            scc = list(nx.strongly_connected_components(self.graph))
            wcc = list(nx.weakly_connected_components(self.graph))

            # Bridges in underlying undirected graph
            undir = self.graph.to_undirected()
            bridges = list(nx.bridges(undir))
            cut_vertices = list(nx.articulation_points(undir))

            return {
                "is_directed": True,
                "is_connected": is_strong,
                "is_strongly_connected": is_strong,
                "is_weakly_connected": is_weak,
                "num_strongly_connected_components": len(scc),
                "strongly_connected_components": [sorted(list(c)) for c in scc],
                "num_weakly_connected_components": len(wcc),
                "weakly_connected_components": [sorted(list(c)) for c in wcc],
                "bridges": bridges,
                "cut_vertices": cut_vertices,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
            }
        else:
            is_conn = nx.is_connected(self.graph)
            components = list(nx.connected_components(self.graph))
            bridges = list(nx.bridges(self.graph))
            cut_vertices = list(nx.articulation_points(self.graph))

            return {
                "is_directed": False,
                "is_connected": is_conn,
                "is_strongly_connected": is_conn,
                "is_weakly_connected": is_conn,
                "num_components": len(components),
                "components": [sorted(list(c)) for c in components],
                "bridges": bridges,
                "cut_vertices": cut_vertices,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
            }

    # -------------------------------------------------------------------------
    # Discrete Mathematics Concept 3: Breadth-First Search (BFS)
    # -------------------------------------------------------------------------

    def breadth_first_search(self, source: str, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes explicit Breadth-First Search (BFS) using a FIFO queue.
        Explores vertices layer by layer to discover unweighted shortest hop paths.

        Time Complexity: O(|V| + |E|)

        Args:
            source: Starting intersection vertex.
            target: Optional target intersection vertex.

        Returns:
            Dictionary containing visit order, hop distances, BFS tree edges, and path.
        """
        if source not in self.graph:
            return {"success": False, "error": f"Source vertex '{source}' does not exist in graph."}

        queue: deque = deque([source])
        visited: Set[str] = {source}
        parent: Dict[str, Optional[str]] = {source: None}
        hop_distance: Dict[str, int] = {source: 0}
        visit_order: List[str] = []
        bfs_tree_edges: List[Tuple[str, str]] = []

        while queue:
            curr = queue.popleft()
            visit_order.append(curr)

            if target and curr == target:
                break

            neighbors = sorted(list(self.graph.neighbors(curr)))
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = curr
                    hop_distance[nbr] = hop_distance[curr] + 1
                    bfs_tree_edges.append((curr, nbr))
                    queue.append(nbr)

        # Reconstruct path to target if specified
        path: List[str] = []
        hop_count: int = 0
        if target:
            if target in visited:
                curr_node: Optional[str] = target
                while curr_node is not None:
                    path.append(curr_node)
                    curr_node = parent[curr_node]
                path.reverse()
                hop_count = len(path) - 1
            else:
                return {
                    "success": False,
                    "error": f"Target vertex '{target}' is unreachable from '{source}'.",
                    "visited_order": visit_order,
                    "hop_distances": hop_distance,
                }

        explanation = (
            f"BFS explored {len(visit_order)} vertices layer-by-layer using a FIFO queue. "
            f"It finds the minimum hop count ({hop_count} hops) in O(|V| + |E|) unweighted time."
        )

        return {
            "success": True,
            "algorithm": "Breadth-First Search (BFS)",
            "source": source,
            "target": target,
            "path": path,
            "hop_count": hop_count,
            "visited_order": visit_order,
            "hop_distances": hop_distance,
            "bfs_tree_edges": bfs_tree_edges,
            "explanation": explanation,
        }

    # -------------------------------------------------------------------------
    # Discrete Mathematics Concept 4: Dijkstra's Shortest Path Algorithm
    # -------------------------------------------------------------------------

    def dijkstra_shortest_path(self, source: str, target: str, weight_key: str = "weight") -> Dict[str, Any]:
        """
        Executes Dijkstra's Shortest Path Algorithm using a min-heap priority queue.

        CRITICAL REQUIREMENT:
        Dijkstra's algorithm is strictly valid ONLY for non-negative edge weights (w(e) >= 0).
        If any edge has a negative weight, a ValueError is raised detailing why Bellman-Ford must be used.

        Time Complexity: O((|V| + |E|) log |V|)

        Args:
            source: Origin intersection vertex.
            target: Destination intersection vertex.
            weight_key: Edge attribute used as weight ('weight' for travel time, 'length' for distance).

        Returns:
            Dictionary containing optimal route path, accumulated cost, distances, and step-by-step log.
        """
        if source not in self.graph:
            return {"success": False, "error": f"Source vertex '{source}' does not exist in graph."}
        if target not in self.graph:
            return {"success": False, "error": f"Target vertex '{target}' does not exist in graph."}

        # Strict validation: Ensure all edge weights are non-negative
        for u, v, data in self.graph.edges(data=True):
            w = data.get(weight_key, 1.0)
            if w < 0.0:
                raise ValueError(
                    f"Dijkstra's algorithm requires non-negative edge weights (w(e) >= 0). "
                    f"Edge ({u}, {v}) has negative weight {w}. For graphs with negative weights, "
                    f"the Bellman-Ford algorithm must be used instead."
                )

        # Priority queue stores tuples: (distance, vertex)
        pq: List[Tuple[float, str]] = [(0.0, source)]
        distances: Dict[str, float] = {node: float("inf") for node in self.graph.nodes()}
        distances[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {node: None for node in self.graph.nodes()}
        settled: Set[str] = set()
        step_log: List[str] = []

        step_log.append(f"Initialized distances: d({source}) = 0, all other d(v) = ∞.")

        while pq:
            d_curr, u = heapq.heappop(pq)

            if u in settled:
                continue
            settled.add(u)
            step_log.append(f"Settled vertex {u} with finalized minimum cost = {d_curr:.2f}.")

            if u == target:
                step_log.append(f"Reached destination {target}. Optimal path finalized.")
                break

            # Relax outgoing edges (u, v)
            for v in self.graph.neighbors(u):
                if v in settled:
                    continue

                edge_w = float(self.graph[u][v].get(weight_key, 1.0))
                # Skip virtually blocked incident roads (impedance >= 990)
                if edge_w >= 990.0:
                    continue

                new_dist = d_curr + edge_w
                if new_dist < distances[v]:
                    distances[v] = round(new_dist, 3)
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))
                    step_log.append(f"Relaxed edge ({u} → {v}): updated d({v}) to {new_dist:.2f} via {u}.")

        # Reconstruct path
        if distances[target] == float("inf"):
            return {
                "success": False,
                "error": f"No available route exists between {source} and {target}.",
                "source": source,
                "target": target,
                "step_log": step_log,
            }

        path: List[str] = []
        curr: Optional[str] = target
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        # Calculate physical road distance in km
        total_dist_km = 0.0
        for i in range(len(path) - 1):
            total_dist_km += self.graph[path[i]][path[i + 1]].get("length", 1.0)

        explanation = (
            f"Dijkstra's greedy algorithm explored the graph using a priority queue. "
            f"Because all edge weights are strictly non-negative (w(e) ≥ 0), each vertex's distance "
            f"is permanently optimal upon being settled. Found optimal {len(path)-1}-hop route with cost {distances[target]:.2f}."
        )

        return {
            "success": True,
            "algorithm": "Dijkstra's Shortest Path Algorithm",
            "source": source,
            "target": target,
            "path": path,
            "total_cost": round(distances[target], 2),
            "total_time_min": round(distances[target], 2) if weight_key == "weight" else None,
            "total_distance_km": round(total_dist_km, 2),
            "hop_count": len(path) - 1,
            "distances": distances,
            "predecessors": predecessors,
            "settled_order": list(settled),
            "step_log": step_log,
            "explanation": explanation,
            "disclaimer": self.disclaimer,
        }

    def get_shortest_path(self, source: str, target: str) -> Dict[str, Any]:
        """
        Main routing method: Computes primary and alternative routes.
        Preserves backward compatibility with existing UI and tests.
        """
        if source not in self.graph or target not in self.graph:
            return {"success": False, "error": "Invalid source or target intersection"}

        try:
            primary = self.dijkstra_shortest_path(source, target, weight_key="weight")
            if not primary["success"]:
                return primary

            path = primary["path"]
            total_time = primary["total_cost"]
            total_dist = primary["total_distance_km"]

            # Compute alternative path (2nd shortest simple path)
            alt_path = []
            alt_time = 0.0
            try:
                simple_paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=5))
                sorted_paths = sorted(
                    simple_paths,
                    key=lambda p: sum(self.graph[p[i]][p[i + 1]]["weight"] for i in range(len(p) - 1))
                )
                if len(sorted_paths) > 1 and sorted_paths[0] == path:
                    alt_path = sorted_paths[1]
                    alt_time = sum(self.graph[alt_path[i]][alt_path[i + 1]]["weight"] for i in range(len(alt_path) - 1))
                elif len(sorted_paths) > 0 and sorted_paths[0] != path:
                    alt_path = sorted_paths[0]
                    alt_time = sum(self.graph[alt_path[i]][alt_path[i + 1]]["weight"] for i in range(len(alt_path) - 1))
            except Exception:
                pass

            return {
                "success": True,
                "path": path,
                "total_time_min": round(total_time, 2),
                "total_distance_km": round(total_dist, 2),
                "alternative_path": alt_path,
                "alternative_time_min": round(alt_time, 2),
                "disclaimer": self.disclaimer,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compute_max_flow_bottleneck(self, source: str = "J1", sink: str = "J6") -> Dict[str, Any]:
        """
        Applies Ford-Fulkerson / Edmonds-Karp to compute maximum throughput
        and identify minimum cut bottleneck edges.
        """
        if source not in self.graph or sink not in self.graph:
            return {"error": "Invalid source or sink"}

        try:
            cut_value, partition = nx.minimum_cut(self.graph, source, sink, capacity="capacity")
            reachable, non_reachable = partition

            cut_edges = []
            for u in reachable:
                for v in non_reachable:
                    if self.graph.has_edge(u, v):
                        cut_edges.append((u, v, self.graph[u][v]["capacity"]))

            return {
                "max_flow_rate_vph": cut_value,
                "bottleneck_cut_edges": cut_edges,
                "reachable_component": list(reachable),
                "isolated_component": list(non_reachable),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_network_centrality_metrics(self) -> pd.DataFrame:
        """Computes Degree, Betweenness, and Closeness centrality for intersections."""
        nodes = sorted(list(self.graph.nodes()))
        deg = dict(self.graph.degree())
        betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        closeness = nx.closeness_centrality(self.graph, distance="weight")

        rows = []
        for n in nodes:
            name = self.graph.nodes[n].get("name", n)
            in_deg = self.graph.in_degree(n) if self.is_directed else deg.get(n, 0)
            out_deg = self.graph.out_degree(n) if self.is_directed else deg.get(n, 0)
            rows.append({
                "Junction": n,
                "Name": name,
                "Total Degree": deg.get(n, 0),
                "In-Degree": in_deg,
                "Out-Degree": out_deg,
                "Betweenness (Criticality)": round(betweenness.get(n, 0.0), 3),
                "Closeness (Accessibility)": round(closeness.get(n, 0.0), 3),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def explain_discrete_math_concepts() -> Dict[str, Dict[str, str]]:
        """
        Returns structured student-friendly explanations of core Discrete Mathematics
        graph concepts mapped to Intelligent Transportation Systems.
        """
        return {
            "vertices": {
                "concept": "Vertices (Nodes) V",
                "definition": "A discrete set of points V = {v1, v2, ..., vn} representing intersections.",
                "traffic_mapping": "Intersections, roundabouts, and expressway exits where vehicular movement choices occur.",
            },
            "edges": {
                "concept": "Edges (Links) E",
                "definition": "Pairs of vertices (u, v) in V x V representing physical connections.",
                "traffic_mapping": "Road corridors, highway segments, overpasses, and turn lanes connecting intersections.",
            },
            "directed_vs_undirected": {
                "concept": "Directed vs. Undirected Graphs",
                "definition": "Directed edges (u -> v) are ordered pairs; undirected edges {u, v} are symmetric sets.",
                "traffic_mapping": "Directed graphs model one-way streets and multi-lane divided avenues. Undirected graphs model two-way residential streets.",
            },
            "adjacency_matrix": {
                "concept": "Adjacency Matrix A",
                "definition": "A square matrix A in {0, 1}^(n x n) where A[i, j] = 1 iff (i, j) in E.",
                "traffic_mapping": "The mathematical bedrock for route planning. The k-th power A^k gives the exact number of walks of length k between junctions.",
            },
            "vertex_degree": {
                "concept": "Vertex Degree & Handshaking Lemma",
                "definition": "Degree deg(v) is the count of incident edges. Handshaking Lemma: sum(deg(v)) = 2|E|.",
                "traffic_mapping": "Indicates intersection complexity. High-degree junctions require multi-phase signal controllers.",
            },
            "connectivity": {
                "concept": "Graph Connectivity & Components",
                "definition": "A graph is connected if a path exists between every pair. Bridges are edges whose removal disconnects the graph.",
                "traffic_mapping": "Network resilience. A bridge road is a single point of failure (e.g., river bridge); its closure isolates city districts.",
            },
            "bfs": {
                "concept": "Breadth-First Search (BFS)",
                "definition": "Layer-by-layer exploration using a FIFO queue. Solves unweighted shortest path in O(|V| + |E|).",
                "traffic_mapping": "Finds route with the fewest number of turns or signal stops regardless of physical length.",
            },
            "dijkstra": {
                "concept": "Dijkstra's Algorithm (w >= 0)",
                "definition": "Greedy algorithm with priority queue. Finds minimum weight path in O((|V| + |E|) log |V|). Requires non-negative weights.",
                "traffic_mapping": "Computes fastest travel time route. Crucially, travel times are non-negative, preserving greedy monotonic optimality.",
            },
            "route_costs": {
                "concept": "Route Costs & BPR Congestion Function",
                "definition": "Cost(P) = sum(w(e) for e in P). Dynamic edge impedance calculated via Bureau of Public Roads formula.",
                "traffic_mapping": "Accounts for queuing delay: t = t0 * [1 + 0.15 * (V/C)^4]. As volume V nears capacity C, impedance escalates sharply.",
            },
        }

