"""
Unit Tests for Graph Model Module.
Validates:
- Graph construction (vertices, edges, directed & undirected modes)
- Four-way intersection model
- Adjacency matrices (binary, weighted, symmetry for undirected)
- Graph connectivity and disconnected component creation
- Breadth-First Search (BFS) hop counts
- Dijkstra shortest path with strict non-negative weight validation
- Dynamic rerouting on congestion and incident road closure
- Handshaking lemma degree conservation
- Ford-Fulkerson max-flow bottlenecks
"""

import unittest
import numpy as np
import pandas as pd
from modules.graph_model import (
    IntersectionGraphNetwork,
    FourWayIntersectionGraph,
    RoadSegment,
    SIMULATION_DISCLAIMER,
)


class TestGraphModel(unittest.TestCase):
    def setUp(self):
        self.net = IntersectionGraphNetwork(is_directed=True)

    def test_network_initialization(self):
        """Verifies default 6-intersection grid construction."""
        self.assertEqual(self.net.graph.number_of_nodes(), 6)
        self.assertGreater(self.net.graph.number_of_edges(), 10)
        self.assertTrue(self.net.graph.has_node("J1"))
        self.assertTrue(self.net.graph.has_node("J6"))
        self.assertTrue(self.net.is_directed)
        self.assertIn("MATHEMATICAL SIMULATION", self.net.disclaimer)

    def test_undirected_graph_construction(self):
        """Feature Test: Undirected graph mode support."""
        undir_net = IntersectionGraphNetwork(is_directed=False)
        self.assertFalse(undir_net.is_directed)
        self.assertEqual(undir_net.graph.number_of_nodes(), 6)
        # In undirected graph, edge (u, v) is symmetric with (v, u)
        self.assertTrue(undir_net.graph.has_edge("J1", "J2"))
        self.assertTrue(undir_net.graph.has_edge("J2", "J1"))

    def test_four_way_intersection_construction(self):
        """Feature Test: Represent example four-way intersection as a graph."""
        four_way = FourWayIntersectionGraph(is_directed=True)
        self.assertEqual(four_way.graph.number_of_nodes(), 5)
        for arm in ["North", "South", "East", "West", "Center"]:
            self.assertTrue(four_way.graph.has_node(arm))
        self.assertTrue(four_way.graph.has_edge("North", "Center"))
        self.assertTrue(four_way.graph.has_edge("Center", "South"))

        # Undirected 4-way intersection
        four_way_undir = FourWayIntersectionGraph(is_directed=False)
        self.assertEqual(four_way_undir.graph.number_of_nodes(), 5)
        self.assertEqual(four_way_undir.graph.number_of_edges(), 4)

    def test_bpr_travel_time(self):
        """Verifies Bureau of Public Roads (BPR) congestion impedance calculation."""
        seg = RoadSegment(u="J1", v="J2", length_km=2.0, free_speed_kmh=60.0, capacity=100, current_vehicles=0)
        # Zero vehicles -> free-flow travel time = (2.0 / 60.0) * 60 = 2.0 minutes
        self.assertAlmostEqual(seg.travel_time_minutes, 2.0, places=1)

        # Under heavy congestion (100 vehicles on 100 capacity), travel time escalates
        seg_congested = RoadSegment(u="J1", v="J2", length_km=2.0, free_speed_kmh=60.0, capacity=100, current_vehicles=100)
        self.assertGreater(seg_congested.travel_time_minutes, seg.travel_time_minutes)

    def test_adjacency_matrices(self):
        """Feature Test: Binary and weighted adjacency matrix generation."""
        # 1. Directed adjacency matrix
        df_adj_dir = self.net.get_adjacency_matrix()
        self.assertEqual(df_adj_dir.shape, (6, 6))
        self.assertIn("J1", df_adj_dir.columns)
        self.assertIn("J6", df_adj_dir.index)

        # 2. Undirected adjacency matrix symmetry: A = A^T
        undir_net = IntersectionGraphNetwork(is_directed=False)
        df_adj_undir = undir_net.get_adjacency_matrix()
        arr = df_adj_undir.to_numpy()
        self.assertTrue(np.array_equal(arr, arr.T), "Undirected adjacency matrix must be symmetric (A = A^T)")

        # 3. Weighted travel time matrix
        df_weights = self.net.get_travel_time_matrix()
        self.assertEqual(df_weights.shape, (6, 6))
        # Direct edge J1 -> J2 should have finite positive weight
        self.assertGreater(df_weights.loc["J1", "J2"], 0.0)
        self.assertNotEqual(df_weights.loc["J1", "J2"], np.inf)

    def test_connected_and_disconnected_graphs(self):
        """Feature Test: Graph connectivity and disconnected component creation."""
        # Default network is connected
        conn_init = self.net.get_connectivity_analysis()
        self.assertTrue(conn_init["is_connected"])
        self.assertTrue(conn_init["is_strongly_connected"])

        # Create disconnected graph by isolating junction J6
        # Remove all incident edges to/from J6: (J3, J6), (J6, J3), (J5, J6), (J6, J5), (J2, J6), (J6, J2)
        edges_to_remove = [("J3", "J6"), ("J6", "J3"), ("J5", "J6"), ("J6", "J5"), ("J2", "J6"), ("J6", "J2")]
        for u, v in edges_to_remove:
            self.net.remove_road(u, v)

        conn_disc = self.net.get_connectivity_analysis()
        self.assertFalse(conn_disc["is_connected"])
        self.assertFalse(conn_disc["is_strongly_connected"])
        self.assertGreater(conn_disc["num_strongly_connected_components"], 1)

        # Path to disconnected node should fail gracefully
        path_res = self.net.get_shortest_path("J1", "J6")
        self.assertFalse(path_res["success"])
        self.assertIn("error", path_res)

    def test_breadth_first_search(self):
        """Feature Test: Breadth-First Search (BFS) finds minimum unweighted hop path."""
        bfs_res = self.net.breadth_first_search("J1", "J6")
        self.assertTrue(bfs_res["success"])
        self.assertEqual(bfs_res["path"][0], "J1")
        self.assertEqual(bfs_res["path"][-1], "J6")
        self.assertGreater(bfs_res["hop_count"], 0)
        self.assertIn("visited_order", bfs_res)
        self.assertIn("hop_distances", bfs_res)
        self.assertIn("FIFO", bfs_res["explanation"])

    def test_dijkstra_shortest_path(self):
        """Feature Test: Dijkstra's shortest path algorithm."""
        dijk_res = self.net.dijkstra_shortest_path("J1", "J6", weight_key="weight")
        self.assertTrue(dijk_res["success"])
        self.assertEqual(dijk_res["path"][0], "J1")
        self.assertEqual(dijk_res["path"][-1], "J6")
        self.assertGreater(dijk_res["total_cost"], 0.0)
        self.assertGreater(dijk_res["total_distance_km"], 0.0)
        self.assertIn("step_log", dijk_res)
        self.assertGreater(len(dijk_res["step_log"]), 0)

    def test_dijkstra_non_negative_weights_requirement(self):
        """
        Critical Requirement Test:
        Use Dijkstra ONLY when edge weights are non-negative (w >= 0).
        Raises ValueError if negative weight is detected.
        """
        # Inject negative edge weight into graph
        self.net.graph["J1"]["J2"]["weight"] = -2.5

        with self.assertRaises(ValueError) as ctx:
            self.net.dijkstra_shortest_path("J1", "J6", weight_key="weight")
        self.assertIn("non-negative edge weights", str(ctx.exception))
        self.assertIn("Bellman-Ford", str(ctx.exception))

    def test_dynamic_rerouting_on_incident(self):
        """Feature Test: Graph modification when edge becomes congested or unavailable."""
        initial = self.net.get_shortest_path("J1", "J5")
        self.assertTrue(initial["success"])

        # Block the direct road segment (J1 -> J5)
        self.net.update_road_traffic("J1", "J5", vehicle_count=80, incident=True)

        rerouted = self.net.get_shortest_path("J1", "J5")
        self.assertTrue(rerouted["success"])
        # Rerouted path must not use the blocked direct edge
        path = rerouted["path"]
        used_direct = any(path[i] == "J1" and path[i + 1] == "J5" for i in range(len(path) - 1))
        self.assertFalse(used_direct, "Rerouted path should bypass blocked road segment")

    def test_handshaking_lemma_and_degrees(self):
        """Feature Test: Vertex degrees and Handshaking Lemma validation."""
        # 1. Directed graph: sum(d_in) == sum(d_out) == |E|
        df_deg_dir = self.net.get_vertex_degrees()
        num_edges_dir = self.net.graph.number_of_edges()
        self.assertEqual(df_deg_dir["In-Degree (d⁻)"].sum(), num_edges_dir)
        self.assertEqual(df_deg_dir["Out-Degree (d⁺)"].sum(), num_edges_dir)

        # 2. Undirected graph: sum(deg(v)) == 2|E|
        undir_net = IntersectionGraphNetwork(is_directed=False)
        df_deg_undir = undir_net.get_vertex_degrees()
        num_edges_undir = undir_net.graph.number_of_edges()
        self.assertEqual(df_deg_undir["Degree (d)"].sum(), 2 * num_edges_undir)

    def test_max_flow_bottleneck(self):
        """Verifies Ford-Fulkerson min-cut bottleneck detection."""
        flow_result = self.net.compute_max_flow_bottleneck("J1", "J6")
        self.assertIn("max_flow_rate_vph", flow_result)
        self.assertGreater(flow_result["max_flow_rate_vph"], 0)
        self.assertTrue(len(flow_result["bottleneck_cut_edges"]) > 0)

    def test_discrete_math_concepts_explanation(self):
        """Verifies presence of student-friendly explanation dictionary."""
        concepts = IntersectionGraphNetwork.explain_discrete_math_concepts()
        expected = ["vertices", "edges", "adjacency_matrix", "vertex_degree", "connectivity", "bfs", "dijkstra"]
        for c in expected:
            self.assertIn(c, concepts)
            self.assertIn("concept", concepts[c])
            self.assertIn("definition", concepts[c])
            self.assertIn("traffic_mapping", concepts[c])

    def test_bpr_impedance_monotonicity(self):
        """Verifies that BPR impedance increases monotonically as volume/capacity ratio grows."""
        road = RoadSegment(u="J1", v="J2", length_km=2.0, free_speed_kmh=60.0, capacity=1000, current_vehicles=0)
        t_free = road.travel_time_minutes

        road.current_vehicles = 500
        t_half = road.travel_time_minutes
        road.current_vehicles = 1000
        t_full = road.travel_time_minutes
        road.current_vehicles = 2000
        t_double = road.travel_time_minutes

        self.assertLess(t_free, t_half)
        self.assertLess(t_half, t_full)
        self.assertLess(t_full, t_double)

    def test_bfs_hop_optimality_vs_dijkstra(self):
        """Verifies that BFS unweighted path hop count is always <= Dijkstra weighted path hop count."""
        bfs_res = self.net.breadth_first_search("J1", "J6")
        dijk_res = self.net.dijkstra_shortest_path("J1", "J6", weight_key="weight")

        self.assertTrue(bfs_res["success"])
        self.assertTrue(dijk_res["success"])
        # Number of hops in BFS path must be <= Dijkstra path hops
        self.assertLessEqual(bfs_res["hop_count"], len(dijk_res["path"]) - 1)

    def test_unreachable_node_handling(self):
        """Verifies graceful handling when destination node is unreachable."""
        # Create an isolated node with no incoming edges
        self.net.graph.add_node("ISOLATED_NODE", pos=(10, 10), name="Isolated Junction")
        res = self.net.dijkstra_shortest_path("J1", "ISOLATED_NODE")
        self.assertFalse(res["success"])
        self.assertIn("No available route exists", res["error"])


if __name__ == "__main__":
    unittest.main()

