"""
Unit Tests for Comprehensive Discrete Mathematics Lab Module.
Validates all 5 Syllabus Units:
  - Unit I:   Logic and Proofs (Propositions, Truth Tables, DNF/CNF, Inference Rules)
  - Unit II:  Sets and Relations (Set Ops, Equivalence Relations, Poset/Lattice, Boolean Algebra)
  - Unit III: Combinatorics (Permutations, Combinations, PHP, PIE, Induction)
  - Unit IV:  Algebraic Systems (Groups, (Z4, +4), Klein 4-Group, Homomorphisms, Linear Independence)
  - Unit V:   Graph Theory (Special Graphs, Walk Powers, Isomorphism, MST, Welsh-Powell, Planarity, Laplacian)
"""

import unittest
import networkx as nx
import pandas as pd
from modules.discrete_math import (
    DiscreteMathLab,
    ConflictGraphModel,
    Unit1LogicProofs,
    Unit2SetsRelations,
    Unit3Combinatorics,
    Unit4AlgebraicSystems,
    Unit5GraphTheory,
    SYLLABUS_CHECKLIST,
)


class TestDiscreteMath(unittest.TestCase):
    def setUp(self):
        self.lab = DiscreteMathLab()

    # ========================================================================
    # SYLLABUS CHECKLIST TEST
    # ========================================================================
    def test_syllabus_checklist_complete(self):
        checklist = self.lab.get_syllabus_checklist()
        self.assertEqual(len(checklist), 5)
        for unit_key in ["UNIT I", "UNIT II", "UNIT III", "UNIT IV", "UNIT V"]:
            self.assertIn(unit_key, checklist)
            unit_data = checklist[unit_key]
            self.assertTrue(len(unit_data["topics"]) >= 4)
            for title, desc, implemented in unit_data["topics"]:
                self.assertTrue(implemented, f"Topic '{title}' in {unit_key} must be implemented")

    # ========================================================================
    # UNIT I: LOGIC AND PROOFS
    # ========================================================================
    def test_unit1_proposition_congestion_rule(self):
        # Case 1: High density, queue exceeded, no emergency, green extended => Satisfied
        eval1 = Unit1LogicProofs.evaluate_traffic_congestion_proposition(
            density_high=True, queue_exceeded=True, emergency_vehicle=False, extend_green=True
        )
        self.assertTrue(eval1["Premise_((P_and_Q)_or_R)"])
        self.assertTrue(eval1["Rule_Satisfied_(((P_and_Q)_or_R)_implies_S)"])

        # Case 2: Emergency present, but green NOT extended => Rule violated
        eval2 = Unit1LogicProofs.evaluate_traffic_congestion_proposition(
            density_high=False, queue_exceeded=False, emergency_vehicle=True, extend_green=False
        )
        self.assertTrue(eval2["Premise_((P_and_Q)_or_R)"])
        self.assertFalse(eval2["Rule_Satisfied_(((P_and_Q)_or_R)_implies_S)"])

    def test_unit1_truth_table_and_normal_forms(self):
        df_tt = Unit1LogicProofs.generate_truth_table_traffic_rule()
        # 4 binary variables => 2^4 = 16 rows
        self.assertEqual(len(df_tt), 16)
        self.assertIn("((P ∧ Q) ∨ R) → S", df_tt.columns)

        # Compute DNF and CNF
        forms = Unit1LogicProofs.compute_normal_forms(
            variables=["P (Density)", "Q (Queue)", "R (Emergency)", "S (Extend)"],
            truth_table_df=df_tt,
            target_col="((P ∧ Q) ∨ R) → S",
        )
        self.assertTrue(forms["is_contingency"])
        self.assertFalse(forms["is_tautology"])
        self.assertGreater(forms["minterm_count"], 0)
        self.assertGreater(forms["maxterm_count"], 0)
        self.assertEqual(forms["minterm_count"] + forms["maxterm_count"], 16)

    def test_unit1_inference_rules(self):
        rules = Unit1LogicProofs.demonstrate_inference_rules()
        self.assertIn("modus_ponens", rules)
        self.assertIn("modus_tollens", rules)
        self.assertIn("disjunctive_syllogism", rules)
        self.assertIn("hypothetical_syllogism", rules)
        for r_key, r_dict in rules.items():
            self.assertIn("Valid Tautology", r_dict["validity"])

    # ========================================================================
    # UNIT II: SETS AND RELATIONS
    # ========================================================================
    def test_unit2_set_operations(self):
        set_a = {"Car_1", "Bus_1", "Bike_1"}
        set_b = {"Car_1", "Truck_1"}
        ops = Unit2SetsRelations.demonstrate_set_operations(set_a, set_b)

        self.assertEqual(set(ops["Union_A_cup_B"]), {"Car_1", "Bus_1", "Bike_1", "Truck_1"})
        self.assertEqual(set(ops["Intersection_A_cap_B"]), {"Car_1"})
        self.assertEqual(set(ops["Difference_A_minus_B"]), {"Bus_1", "Bike_1"})
        self.assertEqual(set(ops["Difference_B_minus_A"]), {"Truck_1"})
        self.assertEqual(ops["Power_Set_Cardinality_2_pow_A"], 8)

    def test_unit2_same_lane_equivalence_relation(self):
        rel = Unit2SetsRelations.analyze_same_lane_relation()
        self.assertTrue(rel["is_reflexive"], "Same-lane must be reflexive")
        self.assertTrue(rel["is_symmetric"], "Same-lane must be symmetric")
        self.assertTrue(rel["is_transitive"], "Same-lane must be transitive")
        self.assertTrue(rel["is_equivalence_relation"], "Same-lane must be an equivalence relation")
        self.assertIn("North", rel["equivalence_classes"])

    def test_unit2_partial_order_lattice(self):
        poset = Unit2SetsRelations.demonstrate_partial_order_lattice()
        self.assertTrue(poset["is_lattice"])
        self.assertTrue(poset["is_bounded_lattice"])
        self.assertTrue(poset["has_bottom_element"])
        self.assertTrue(poset["has_top_element"])

    def test_unit2_boolean_algebra_simplification(self):
        ba = Unit2SetsRelations.demonstrate_boolean_algebra_traffic_rule()
        self.assertEqual(ba["simplified_expression"], "Emergency")
        self.assertEqual(len(ba["proof_steps"]), 4)

    # ========================================================================
    # UNIT III: COMBINATORICS
    # ========================================================================
    def test_unit3_permutations_and_combinations(self):
        # 4 approaches, choose 2
        comb = Unit3Combinatorics.compute_permutations_and_combinations(n=4, r=2)
        # P(4, 2) = 4 * 3 = 12
        self.assertEqual(comb["permutations_P_n_r"], 12)
        # C(4, 2) = 4 * 3 / 2 = 6
        self.assertEqual(comb["combinations_C_n_r"], 6)
        # Circular (4 - 1)! = 3! = 6
        self.assertEqual(comb["circular_permutations_(n-1)!"], 6)

    def test_unit3_pigeonhole_principle(self):
        # 13 vehicles in 4 lanes => ceil(13 / 4) = 4
        php = Unit3Combinatorics.demonstrate_pigeonhole_principle(num_vehicles=13, num_lanes=4)
        self.assertEqual(php["ceiling_n_over_k"], 4)
        self.assertTrue(php["is_guaranteed"])

    def test_unit3_inclusion_exclusion(self):
        pie = Unit3Combinatorics.demonstrate_inclusion_exclusion(
            count_a=40, count_b=30, count_c=20,
            ab=10, ac=8, bc=6, abc=3,
        )
        # Naive sum = 40 + 30 + 20 = 90
        self.assertEqual(pie["naive_sum_double_counted"], 90)
        # PIE = 40 + 30 + 20 - (10 + 8 + 6) + 3 = 90 - 24 + 3 = 69
        self.assertEqual(pie["correct_PIE_union"], 69)
        self.assertEqual(pie["overcount_error"], 21)

    def test_unit3_mathematical_induction(self):
        ind = Unit3Combinatorics.demonstrate_mathematical_induction(n_junctions=8)
        self.assertEqual(ind["calculated_road_links"], 7)
        self.assertEqual(len(ind["induction_steps"]), 4)

    # ========================================================================
    # UNIT IV: ALGEBRAIC SYSTEMS
    # ========================================================================
    def test_unit4_z4_cyclic_group(self):
        z4 = Unit4AlgebraicSystems.demonstrate_cyclic_phase_group()
        self.assertTrue(z4["is_closed"])
        self.assertTrue(z4["is_associative"])
        self.assertTrue(z4["has_identity"])
        self.assertEqual(z4["identity_element"], 0)
        self.assertTrue(z4["has_inverses"])
        self.assertTrue(z4["is_commutative"])
        self.assertEqual(z4["classification"], "Abelian (Commutative) Group")

    def test_unit4_klein_four_group(self):
        k4 = Unit4AlgebraicSystems.demonstrate_klein_four_group()
        self.assertTrue(k4["is_closed"])
        self.assertTrue(k4["has_identity"])
        self.assertEqual(k4["identity_element"], "(0,0)")
        self.assertTrue(k4["has_inverses"])
        # In Klein 4-group, every element is its own inverse
        for elem, inv in k4["inverse_map"].items():
            self.assertEqual(elem, inv)
        self.assertEqual(k4["classification"], "Abelian (Commutative) Group")

    def test_unit4_homomorphism(self):
        hom = Unit4AlgebraicSystems.demonstrate_homomorphism()
        self.assertTrue(hom["preserves_operation"])
        self.assertEqual(set(hom["kernel_ker_f"]), {0, 2})

    def test_unit4_vector_space_linear_independence(self):
        # Linearly independent 3D vectors
        vecs_indep = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        res_indep = Unit4AlgebraicSystems.demonstrate_vector_space_linear_independence(vecs_indep)
        self.assertTrue(res_indep["is_linearly_independent"])
        self.assertEqual(res_indep["matrix_rank"], 3)
        self.assertAlmostEqual(res_indep["determinant"], 1.0)

        # Linearly dependent vectors
        vecs_dep = [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],  # 2 * v1
            [0.0, 1.0, 0.0],
        ]
        res_dep = Unit4AlgebraicSystems.demonstrate_vector_space_linear_independence(vecs_dep)
        self.assertFalse(res_dep["is_linearly_independent"])
        self.assertLess(res_dep["matrix_rank"], 3)

    def test_unit4_algebra_vs_automata_explanation(self):
        expl = Unit4AlgebraicSystems.explain_algebra_vs_state_automata()
        self.assertIn("automaton_definition", expl)
        self.assertIn("why_not_a_group", expl)

    # ========================================================================
    # UNIT V: GRAPH THEORY
    # ========================================================================
    def test_unit5_special_graphs(self):
        specs = Unit5GraphTheory.generate_special_graphs()
        self.assertEqual(specs["complete_K4"]["edges_count"], 6)
        self.assertEqual(specs["bipartite_K33"]["edges_count"], 9)
        self.assertFalse(specs["bipartite_K33"]["is_planar"])
        self.assertEqual(specs["cycle_C5"]["edges_count"], 5)

    def test_unit5_adjacency_walk_powers(self):
        g = nx.path_graph(4)  # 0 - 1 - 2 - 3
        powers = Unit5GraphTheory.compute_adjacency_walk_powers(g, max_k=3)
        self.assertIn("A^1", powers)
        self.assertIn("A^2", powers)
        self.assertIn("A^3", powers)
        # In path 0-1-2-3, walks of length 2 from 0 to 2 is exactly 1 (0 -> 1 -> 2)
        df_a2 = powers["A^2"]
        self.assertEqual(df_a2.loc[0, 2], 1)
        # Walks of length 2 from 0 to 0 is 1 (0 -> 1 -> 0)
        self.assertEqual(df_a2.loc[0, 0], 1)

    def test_unit5_graph_isomorphism(self):
        g1 = nx.cycle_graph(4)
        # Create isomorphic g2 with relabeled nodes
        g2 = nx.relabel_nodes(g1, {0: "A", 1: "B", 2: "C", 3: "D"})
        iso_res = Unit5GraphTheory.test_graph_isomorphism(g1, g2)
        self.assertTrue(iso_res["is_isomorphic"])
        self.assertIsNotNone(iso_res["vertex_mapping"])

        # Non-isomorphic graph (path graph vs cycle graph)
        g_path = nx.path_graph(4)
        non_iso_res = Unit5GraphTheory.test_graph_isomorphism(g1, g_path)
        self.assertFalse(non_iso_res["is_isomorphic"])

    def test_unit5_minimum_spanning_tree(self):
        g = nx.cycle_graph(4)
        # Add weights
        for u, v in g.edges():
            g[u][v]["weight"] = 2.0
        mst_res = Unit5GraphTheory.compute_minimum_spanning_tree(g)
        self.assertEqual(mst_res["mst_edge_count"], 3)  # V - 1 = 4 - 1 = 3
        self.assertAlmostEqual(mst_res["total_cabling_weight"], 6.0)

    # ========================================================================
    # PRESERVED EXISTING METHODS & VIVA VOCE
    # ========================================================================
    def test_conflict_graph_welsh_powell(self):
        result = self.lab.get_conflict_coloring()
        self.assertEqual(result["total_vertices"], 8)
        self.assertGreater(result["total_conflict_edges"], 10)
        self.assertGreaterEqual(result["chromatic_number"], 3)
        self.assertLessEqual(result["chromatic_number"], 6)

        cg = self.lab.conflict_model.graph
        node_colors = result["node_colors"]
        for u, v in cg.edges():
            self.assertNotEqual(
                node_colors[u],
                node_colors[v],
                f"Conflict edge ({u}, {v}) has identical color {node_colors[u]}",
            )

    def test_euler_formula_planarity(self):
        res = DiscreteMathLab.verify_planarity_and_euler(num_v=6, num_e=7, num_f=3)
        self.assertTrue(res["is_valid_euler"])
        self.assertEqual(res["euler_characteristic"], 2)
        self.assertTrue(res["is_theoretically_planar"])

    def test_eulerian_and_hamiltonian(self):
        c4 = nx.cycle_graph(4)
        analysis = DiscreteMathLab.analyze_eulerian_and_hamiltonian(c4)
        self.assertTrue(analysis["is_connected"])
        self.assertEqual(analysis["odd_degree_count"], 0)
        self.assertTrue(analysis["has_eulerian_circuit"])

    def test_laplacian_spectral(self):
        g = nx.path_graph(4)
        spectral = DiscreteMathLab.compute_laplacian_spectral(g)
        self.assertEqual(len(spectral["eigenvalues"]), 4)
        self.assertAlmostEqual(spectral["eigenvalues"][0], 0.0, places=2)
        self.assertGreater(spectral["fiedler_value"], 0.0)

    def test_viva_questions_present(self):
        viva = DiscreteMathLab.get_viva_questions()
        self.assertGreaterEqual(len(viva), 10)
        for item in viva:
            self.assertIn("question", item)
            self.assertIn("answer", item)
            self.assertTrue(len(item["answer"]) > 20)


    def test_mathematical_demonstrations_demorgan(self):
        """Test De Morgan's Laws over all 2^2 Boolean valuation truth assignments."""
        for p in [True, False]:
            for q in [True, False]:
                # Law 1: not(P or Q) == (not P and not Q)
                self.assertEqual(not (p or q), (not p) and (not q))
                # Law 2: not(P and Q) == (not P or not Q)
                self.assertEqual(not (p and q), (not p) or (not q))

    def test_mathematical_demonstrations_cartesian_and_handshaking(self):
        """Verify Cartesian product cardinality |A x B| = |A| * |B| and Handshaking Lemma sum(deg) = 2|E|."""
        set_a = {"Car", "Bus", "Bike"}
        set_b = {"North", "South", "East", "West"}
        cartesian = {(a, b) for a in set_a for b in set_b}
        self.assertEqual(len(cartesian), len(set_a) * len(set_b))

        # Handshaking lemma test on an arbitrary network graph
        g = nx.erdos_renyi_graph(n=7, p=0.4, seed=42)
        degree_sum = sum(dict(g.degree()).values())
        self.assertEqual(degree_sum, 2 * g.number_of_edges())


if __name__ == "__main__":
    unittest.main()

