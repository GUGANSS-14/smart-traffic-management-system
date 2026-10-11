"""
Discrete Mathematics Lab Module for Smart Traffic Management
=============================================================
Complete Academic Curriculum Implementation mapping the Smart Traffic System
to the 5 Standard Discrete Mathematics Units:

UNIT I   - LOGIC AND PROOFS
           Propositions, Truth Tables, Logical Equivalences, DNF/CNF Normal Forms,
           Congestion Rule Propositional Modeling, and Inference Rules (Modus Ponens/Tollens).

UNIT II  - SETS AND RELATIONS
           Vehicle Sets, Subsets, Cartesian Products, Set Algebra,
           Equivalence Relations (Same-Lane), Partial Orders (Poset),
           Lattices, and Boolean Algebra with Traffic Priority Simplification.

UNIT III - COMBINATORICS
           Permutations & Combinations in Signal Phase Scheduling,
           Pigeonhole Principle in Lane Congestion Guarantees,
           Principle of Inclusion-Exclusion (PIE) for Multi-Attribute Vehicles,
           and Mathematical Induction for Arterial Road Networks.

UNIT IV  - ALGEBRAIC SYSTEMS
           Binary Operations, Semigroups, Monoids, Groups, Abelian Groups,
           Cyclic Phase Group (Z_4, +_4), Klein 4-Group, Homomorphisms with Kernels,
           Vector Spaces & Linear Independence of Detector Sensor Readings,
           and Rigorous Distinction between Algebraic Groups and Automata.

UNIT V   - GRAPH THEORY
           Special Graph Types (Kn, Km,n, Cn, Regular), Graph Isomorphism,
           Adjacency Matrix Walk Powers, Eulerian Patrol Trails, Hamiltonian Maintenance Cycles,
           Trees, Minimum Spanning Trees (MST / Kruskal), Welsh-Powell Conflict Coloring,
           Euler's Planarity Characteristic, and Laplacian Spectral Connectivity.

DISCLAIMER: All discrete mathematical demonstrations are educational academic
models designed to bridge theoretical CSE concepts with intelligent transportation.
They do not claim to optimize real-world municipal highways without live GIS data.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Callable
import itertools
import math
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================================
# SYLLABUS COVERAGE CHECKLIST
# ============================================================================

SYLLABUS_CHECKLIST: Dict[str, Dict[str, Any]] = {
    "UNIT I": {
        "title": "LOGIC AND PROOFS",
        "topics": [
            ("Propositions & Truth Tables", "Truth table generator evaluating composite propositions", True),
            ("Logical Equivalences & De Morgan", "Equivalence proof for implication and De Morgan laws", True),
            ("Normal Forms (DNF & CNF)", "Algorithmic synthesis of DNF minterms and CNF maxterms", True),
            ("Congestion Rule Propositional Model", "Traffic extension formula ((P ∧ Q) ∨ R) → S", True),
            ("Rules of Inference", "Worked examples of Modus Ponens, Modus Tollens, and Syllogism", True),
        ],
    },
    "UNIT II": {
        "title": "SETS AND RELATIONS",
        "topics": [
            ("Vehicle Sets & Operations", "Union, intersection, difference, Cartesian product of traffic sets", True),
            ("Subsets & Power Sets", "Power set generation of active signal movements (2^|S|)", True),
            ("Relations & Properties", "Same-lane equivalence relation (Reflexive, Symmetric, Transitive)", True),
            ("Partial Orders & Lattices", "Poset (P, ≤), Hasse diagram, Meet (∧) and Join (∨) lattice", True),
            ("Boolean Algebra", "Application of Boolean laws to simplify emergency traffic override rules", True),
        ],
    },
    "UNIT III": {
        "title": "COMBINATORICS",
        "topics": [
            ("Permutations & Combinations", "Phase sequencing P(n, r) and surveillance selection C(n, r)", True),
            ("Pigeonhole Principle (PHP)", "Guaranteed lane congestion threshold ⌈n / k⌉ under queue limits", True),
            ("Principle of Inclusion-Exclusion", "2-set and 3-set PIE preventing vehicle double-counting", True),
            ("Mathematical Induction", "Proof that any linear corridor of n junctions has n - 1 road links", True),
        ],
    },
    "UNIT IV": {
        "title": "ALGEBRAIC SYSTEMS",
        "topics": [
            ("Binary Operations & Group Axioms", "Closure, Associativity, Identity, and Inverses verification", True),
            ("Cyclic Signal Phase Group (Z₄, +₄)", "4-phase cyclic signal transitions as an Abelian group", True),
            ("Klein 4-Group (V₄)", "Dual-axis toggle switches under component-wise XOR", True),
            ("Group Homomorphisms", "Valid homomorphism f: (Z₄, +₄) → (Z₂, +₂) and kernel ker(f)", True),
            ("Vector Spaces & Linear Independence", "Flow rate vectors in R^k and matrix rank / determinant test", True),
            ("Algebraic Groups vs State Machines", "Rigorous distinction between groups and irreversible automata", True),
        ],
    },
    "UNIT V": {
        "title": "GRAPH THEORY",
        "topics": [
            ("Vertices, Edges & Special Graphs", "Complete Kn, Bipartite Km,n, Cycles Cn, and Regular graphs", True),
            ("Adjacency Matrix & Walk Powers", "Walk count theorem: (A^k)ᵢ,ⱼ gives walks of length k", True),
            ("Graph Isomorphism", "Structural bijection f: V₁ → V₂ preserving edge adjacency", True),
            ("Eulerian Paths & Circuits", "Police patrol route traversing every road link exactly once", True),
            ("Hamiltonian Cycles", "Junction maintenance routing and Dirac's condition", True),
            ("Trees & Minimum Spanning Tree", "Kruskal's MST for minimal sensor cabling and Cayley's n^(n-2)", True),
            ("Vertex Coloring (Welsh-Powell)", "Conflict graph coloring minimizing safe green phases", True),
            ("Planar Graphs & Euler's Formula", "V - E + F = 2 and criteria for grade-separated flyovers", True),
            ("Laplacian Spectral Connectivity", "L = D - A and Fiedler algebraic connectivity eigenvalue λ₂", True),
        ],
    },
}


# ============================================================================
# UNIT I: LOGIC AND PROOFS
# ============================================================================

class Unit1LogicProofs:
    """
    Implements Discrete Mathematics Unit I:
    Propositions, truth tables, normal forms (DNF/CNF),
    propositional traffic congestion modeling, and rules of inference.
    """

    @staticmethod
    def evaluate_traffic_congestion_proposition(
        density_high: bool, queue_exceeded: bool, emergency_vehicle: bool, extend_green: bool
    ) -> Dict[str, Any]:
        """
        Propositional Formula:
          P: Density is high (density >= 0.70)
          Q: Queue length exceeded threshold (queue >= 5)
          R: Emergency vehicle approaching
          S: Extend green signal duration

        Traffic Rule:
          ((P ∧ Q) ∨ R) → S
        """
        p_and_q = density_high and queue_exceeded
        premise = p_and_q or emergency_vehicle
        # Implication: Premise -> S is equivalent to (not Premise) or S
        rule_satisfied = (not premise) or extend_green

        return {
            "P_density_high": density_high,
            "Q_queue_exceeded": queue_exceeded,
            "R_emergency_vehicle": emergency_vehicle,
            "S_extend_green": extend_green,
            "P_and_Q": p_and_q,
            "Premise_((P_and_Q)_or_R)": premise,
            "Rule_Satisfied_(((P_and_Q)_or_R)_implies_S)": rule_satisfied,
            "explanation": (
                f"With P={density_high}, Q={queue_exceeded}, R={emergency_vehicle}, "
                f"the premise evaluates to {premise}. Since S={extend_green}, "
                f"the implication premise → S evaluates to {rule_satisfied}."
            ),
        }

    @staticmethod
    def generate_truth_table_traffic_rule() -> pd.DataFrame:
        """
        Generates the complete 16-row truth table for:
          P: High Density
          Q: Long Queue
          R: Emergency
          S: Extend Green
          Premise = (P ∧ Q) ∨ R
          Implication = Premise → S
        """
        rows = []
        for p, q, r, s in itertools.product([True, False], repeat=4):
            p_and_q = p and q
            premise = p_and_q or r
            implication = (not premise) or s
            rows.append({
                "P (Density)": p,
                "Q (Queue)": q,
                "R (Emergency)": r,
                "S (Extend)": s,
                "P ∧ Q": p_and_q,
                "(P ∧ Q) ∨ R": premise,
                "((P ∧ Q) ∨ R) → S": implication,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_normal_forms(variables: List[str], truth_table_df: pd.DataFrame, target_col: str) -> Dict[str, str]:
        """
        Computes Disjunctive Normal Form (DNF / Sum of Minterms)
        and Conjunctive Normal Form (CNF / Product of Maxterms).
        """
        dnf_terms = []
        cnf_terms = []

        for _, row in truth_table_df.iterrows():
            output_val = row[target_col]
            if output_val:
                # DNF Minterm: AND of literals
                literals = []
                for var in variables:
                    if row[var]:
                        literals.append(var)
                    else:
                        literals.append(f"¬{var}")
                dnf_terms.append(f"({' ∧ '.join(literals)})")
            else:
                # CNF Maxterm: OR of negated literals
                literals = []
                for var in variables:
                    if row[var]:
                        literals.append(f"¬{var}")
                    else:
                        literals.append(var)
                cnf_terms.append(f"({' ∨ '.join(literals)})")

        dnf_str = " ∨ ".join(dnf_terms) if dnf_terms else "FALSE"
        cnf_str = " ∧ ".join(cnf_terms) if cnf_terms else "TRUE"

        return {
            "DNF": dnf_str,
            "CNF": cnf_str,
            "minterm_count": len(dnf_terms),
            "maxterm_count": len(cnf_terms),
            "is_tautology": len(cnf_terms) == 0,
            "is_contradiction": len(dnf_terms) == 0,
            "is_contingency": len(dnf_terms) > 0 and len(cnf_terms) > 0,
        }

    @staticmethod
    def demonstrate_inference_rules() -> Dict[str, Any]:
        """
        Demonstrates fundamental Rules of Inference applied to intelligent traffic control:
        1. Modus Ponens: P → Q, P ⊢ Q
        2. Modus Tollens: P → Q, ¬Q ⊢ ¬P
        3. Disjunctive Syllogism: P ∨ Q, ¬P ⊢ Q
        4. Hypothetical Syllogism: P → Q, Q → R ⊢ P → R
        """
        return {
            "modus_ponens": {
                "name": "Modus Ponens (Rule of Detachment)",
                "schema": "Premise 1: P → Q, Premise 2: P  ⊢  Conclusion: Q",
                "traffic_premise_1": "If lane density exceeds 85% (P), then green time is extended to 50s (Q).",
                "traffic_premise_2": "Camera computer vision measures lane density = 91% (P holds True).",
                "traffic_conclusion": "Therefore, green time is extended to 50s (Q is deduced).",
                "validity": "Valid Tautology: ((P → Q) ∧ P) → Q is unconditionally True.",
            },
            "modus_tollens": {
                "name": "Modus Tollens (Denying the Consequent)",
                "schema": "Premise 1: P → Q, Premise 2: ¬Q  ⊢  Conclusion: ¬P",
                "traffic_premise_1": "If an emergency siren/vehicle is detected (P), then all crossing phases switch to RED (Q).",
                "traffic_premise_2": "Crossing phases are observed operating in normal green cycle, not all RED (¬Q).",
                "traffic_conclusion": "Therefore, no emergency vehicle is currently detected (¬P is deduced).",
                "validity": "Valid Tautology: ((P → Q) ∧ ¬Q) → ¬P is unconditionally True.",
            },
            "disjunctive_syllogism": {
                "name": "Disjunctive Syllogism",
                "schema": "Premise 1: P ∨ Q, Premise 2: ¬P  ⊢  Conclusion: Q",
                "traffic_premise_1": "Vehicles are routed via Arterial Avenue (P) OR Bypass Expressway (Q).",
                "traffic_premise_2": "Arterial Avenue is fully closed due to utility pipeline maintenance (¬P).",
                "traffic_conclusion": "Therefore, vehicles must be routed via Bypass Expressway (Q is deduced).",
                "validity": "Valid Tautology: ((P ∨ Q) ∧ ¬P) → Q is unconditionally True.",
            },
            "hypothetical_syllogism": {
                "name": "Hypothetical Syllogism (Chain Rule)",
                "schema": "Premise 1: P → Q, Premise 2: Q → R  ⊢  Conclusion: P → R",
                "traffic_premise_1": "If heavy rain creates hydroplaning hazard (P), average approach speed drops below 30 km/h (Q).",
                "traffic_premise_2": "If average approach speed drops below 30 km/h (Q), headway gap intervals increase (R).",
                "traffic_conclusion": "Therefore, if heavy rain creates hazard (P), headway gap intervals increase (P → R).",
                "validity": "Valid Tautology: ((P → Q) ∧ (Q → R)) → (P → R) is unconditionally True.",
            },
        }


# ============================================================================
# UNIT II: SETS AND RELATIONS
# ============================================================================

class Unit2SetsRelations:
    """
    Implements Discrete Mathematics Unit II:
    Sets of traffic entities, operations, relations, equivalence relations,
    partial orders (poset), lattices, and Boolean algebra.
    """

    @staticmethod
    def demonstrate_set_operations(
        set_a: Optional[Set[str]] = None,
        set_b: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Demonstrates union, intersection, difference, symmetric difference,
        Cartesian product, and subset relations for traffic sets.
        """
        # Default sets: Vehicles detected in Lane 1 vs Lane 2
        if set_a is None:
            set_a = {"Car_101", "Car_102", "Bus_201", "Bike_301"}
        if set_b is None:
            set_b = {"Car_102", "Truck_401", "Bike_301", "Van_501"}

        universal = set_a | set_b | {"Car_999", "Emergency_001"}

        union = set_a | set_b
        intersection = set_a & set_b
        diff_a_b = set_a - set_b
        diff_b_a = set_b - set_a
        sym_diff = set_a ^ set_b
        is_subset = set_a.issubset(set_b)
        cartesian_sample = list(itertools.product(sorted(list(set_a))[:2], sorted(list(set_b))[:2]))
        power_set_size = 2 ** len(set_a)

        return {
            "Set_A": sorted(list(set_a)),
            "Set_B": sorted(list(set_b)),
            "Universal_Set": sorted(list(universal)),
            "Union_A_cup_B": sorted(list(union)),
            "Intersection_A_cap_B": sorted(list(intersection)),
            "Difference_A_minus_B": sorted(list(diff_a_b)),
            "Difference_B_minus_A": sorted(list(diff_b_a)),
            "Symmetric_Difference": sorted(list(sym_diff)),
            "Is_A_subset_of_B": is_subset,
            "Cartesian_Product_Sample": cartesian_sample,
            "Power_Set_Cardinality_2_pow_A": power_set_size,
            "cardinality_A": len(set_a),
            "cardinality_B": len(set_b),
            "cardinality_Union": len(union),
            "cardinality_Intersection": len(intersection),
        }

    @staticmethod
    def analyze_same_lane_relation() -> Dict[str, Any]:
        """
        Defines a relation R on detected vehicles V:
          (v_i, v_j) ∈ R  ⟺  lane(v_i) == lane(v_j)
        Tests Reflexivity, Symmetry, Transitivity, and establishes Equivalence Relation.
        """
        vehicles = [
            {"id": "V1", "lane": "North"},
            {"id": "V2", "lane": "North"},
            {"id": "V3", "lane": "South"},
            {"id": "V4", "lane": "East"},
            {"id": "V5", "lane": "South"},
        ]
        ids = [v["id"] for v in vehicles]
        lane_map = {v["id"]: v["lane"] for v in vehicles}

        # Build relation pairs
        relation_pairs = set()
        for v1 in ids:
            for v2 in ids:
                if lane_map[v1] == lane_map[v2]:
                    relation_pairs.add((v1, v2))

        # Test Reflexive: (v, v) in R for all v
        is_reflexive = all((v, v) in relation_pairs for v in ids)

        # Test Symmetric: (u, v) in R => (v, u) in R
        is_symmetric = all((v, u) in relation_pairs for (u, v) in relation_pairs)

        # Test Transitive: (u, v) in R and (v, w) in R => (u, w) in R
        is_transitive = True
        for (u, v) in relation_pairs:
            for (v2, w) in relation_pairs:
                if v == v2 and (u, w) not in relation_pairs:
                    is_transitive = False
                    break

        is_equivalence = is_reflexive and is_symmetric and is_transitive

        # Equivalence classes
        classes = {}
        for v in ids:
            lane = lane_map[v]
            classes.setdefault(lane, []).append(v)

        return {
            "vehicles": vehicles,
            "relation_pairs": sorted(list(relation_pairs)),
            "is_reflexive": is_reflexive,
            "is_symmetric": is_symmetric,
            "is_transitive": is_transitive,
            "is_equivalence_relation": is_equivalence,
            "equivalence_classes": classes,
            "mathematical_insight": (
                "Because 'Same-Lane' is an Equivalence Relation, it partitions the set of detected vehicles "
                "into disjoint equivalence classes [L_North], [L_South], [L_East]. In traffic engineering, "
                "each equivalence class forms an independent queue vector for signal optimization."
            ),
        }

    @staticmethod
    def demonstrate_partial_order_lattice() -> Dict[str, Any]:
        """
        Demonstrates a Partial Order (Poset) and Lattice on Traffic Priority:
        Subset inclusion lattice (P(S), ⊆) for critical clearance permissions S = {Emergency, Transit, General}.
        A lattice requires that every pair of elements has a unique Least Upper Bound (LUB/Join)
        and Greatest Lower Bound (GLB/Meet).
        """
        base_set = {"E", "T"}  # E: Emergency, T: Transit
        subsets = [set(), {"E"}, {"T"}, {"E", "T"}]
        subset_names = ["∅ (Zero Priority)", "{E} (Emergency Only)", "{T} (Transit Only)", "{E, T} (Full Priority)"]

        # Form poset elements
        # Order relation: A ≤ B iff A ⊆ B
        order_pairs = []
        for i, s1 in enumerate(subsets):
            for j, s2 in enumerate(subsets):
                if s1.issubset(s2):
                    order_pairs.append((subset_names[i], subset_names[j]))

        # Lattice properties: Meet = Intersection, Join = Union
        sample_meet = subset_names[0]  # {E} ∩ {T} = ∅
        sample_join = subset_names[3]  # {E} ∪ {T} = {E, T}

        return {
            "base_set": sorted(list(base_set)),
            "poset_elements": subset_names,
            "order_pairs_count": len(order_pairs),
            "has_bottom_element": True,
            "bottom_element": "∅ (Zero Priority / Red)",
            "has_top_element": True,
            "top_element": "{E, T} (Full Priority / All Green)",
            "sample_meet_glb": f"Meet({{E}}, {{T}}) = {{E}} ∩ {{T}} = {sample_meet}",
            "sample_join_lub": f"Join({{E}}, {{T}}) = {{E}} ∪ {{T}} = {sample_join}",
            "is_lattice": True,
            "is_bounded_lattice": True,
            "explanation": (
                "The power set lattice (P({E, T}), ⊆) satisfies: Reflexivity (A ⊆ A), "
                "Antisymmetry (A ⊆ B ∧ B ⊆ A ⇒ A = B), and Transitivity (A ⊆ B ∧ B ⊆ C ⇒ A ⊆ C). "
                "Since every pair has a GLB (set intersection) and LUB (set union), it forms a complete Boolean Lattice."
            ),
        }

    @staticmethod
    def demonstrate_boolean_algebra_traffic_rule() -> Dict[str, Any]:
        """
        Demonstrates Boolean Algebra applied to traffic priority simplification:
        Expression: (Emergency ∧ PeakHour) ∨ (Emergency ∧ ¬PeakHour)
        By Distributive Law: Emergency ∧ (PeakHour ∨ ¬PeakHour)
        By Complement Law:   PeakHour ∨ ¬PeakHour = 1 (True)
        By Identity Law:     Emergency ∧ 1 = Emergency
        """
        laws_applied = [
            {"step": "Initial Logic", "expression": "(E ∧ P) ∨ (E ∧ ¬P)", "law": "Given rule formulation"},
            {"step": "Step 1: Factoring", "expression": "E ∧ (P ∨ ¬P)", "law": "Distributive Law: x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)"},
            {"step": "Step 2: Complement", "expression": "E ∧ 1", "law": "Complement Law: x ∨ ¬x = 1 (Tautology)"},
            {"step": "Step 3: Identity", "expression": "E", "law": "Identity Law: x ∧ 1 = x"},
        ]
        return {
            "initial_expression": "(Emergency ∧ PeakHour) ∨ (Emergency ∧ ¬PeakHour)",
            "simplified_expression": "Emergency",
            "proof_steps": laws_applied,
            "traffic_meaning": (
                "Boolean algebraic reduction proves that an emergency vehicle signal preemption "
                "is completely independent of whether it is peak hour or off-peak hour."
            ),
        }


# ============================================================================
# UNIT III: COMBINATORICS
# ============================================================================

class Unit3Combinatorics:
    """
    Implements Discrete Mathematics Unit III:
    Permutations, combinations, pigeonhole principle,
    principle of inclusion-exclusion (PIE), and mathematical induction.
    """

    @staticmethod
    def compute_permutations_and_combinations(n: int = 4, r: int = 2) -> Dict[str, Any]:
        """
        Computes Permutations P(n, r) and Combinations C(n, r):
          - Permutations: Distinct ordered sequences of servicing r approaches from n available directions.
          - Combinations: Selecting r lanes for simultaneous priority monitoring out of n lanes.
          - Circular Permutations: (n - 1)! for roundabout rotary flows.
        """
        p_val = math.perm(n, r) if n >= r else 0
        c_val = math.comb(n, r) if n >= r else 0
        circular_perm = math.factorial(n - 1) if n >= 1 else 0

        return {
            "n_total": n,
            "r_selected": r,
            "permutations_P_n_r": p_val,
            "combinations_C_n_r": c_val,
            "circular_permutations_(n-1)!": circular_perm,
            "permutation_formula": f"P({n}, {r}) = {n}! / ({n} - {r})! = {p_val}",
            "combination_formula": f"C({n}, {r}) = {n}! / ({r}! · ({n} - {r})!) = {c_val}",
            "traffic_application": (
                f"Permutations: There are {p_val} ways to establish an ordered sequence of {r} green phases from {n} approaches. "
                f"Combinations: There are {c_val} unordered ways to choose {r} approaches for concurrent dual-camera tracking."
            ),
        }

    @staticmethod
    def demonstrate_pigeonhole_principle(num_vehicles: int = 14, num_lanes: int = 4) -> Dict[str, Any]:
        """
        Pigeonhole Principle (PHP):
        If n items (vehicles) are placed into k containers (lanes), where n > k,
        then at least one container must contain at least ⌈n / k⌉ items.
        """
        min_in_one_lane = math.ceil(num_vehicles / num_lanes) if num_lanes > 0 else 0
        is_guaranteed = num_vehicles > num_lanes

        return {
            "num_vehicles_pigeons": num_vehicles,
            "num_lanes_holes": num_lanes,
            "ceiling_n_over_k": min_in_one_lane,
            "is_guaranteed": is_guaranteed,
            "formula": f"⌈{num_vehicles} / {num_lanes}⌉ = {min_in_one_lane}",
            "formal_statement": (
                f"By the Generalized Pigeonhole Principle, distributing {num_vehicles} vehicles across {num_lanes} lanes "
                f"guarantees that at least one lane contains at least ⌈{num_vehicles} / {num_lanes}⌉ = {min_in_one_lane} vehicles."
            ),
            "traffic_significance": (
                f"If each lane has a maximum queue capacity of {min_in_one_lane - 1} before spillback into the intersection occurs, "
                f"detecting {num_vehicles} vehicles mathematically proves a spillback condition exists, without needing per-lane sensor telemetry."
            ),
        }

    @staticmethod
    def demonstrate_inclusion_exclusion(
        count_a: int = 45,  # Speeding vehicles
        count_b: int = 30,  # Heavy trucks
        count_c: int = 25,  # Peak-hour violations
        ab: int = 12,       # Speeding AND Heavy
        ac: int = 10,       # Speeding AND Violation
        bc: int = 8,        # Heavy AND Violation
        abc: int = 4,       # Speeding AND Heavy AND Violation
    ) -> Dict[str, Any]:
        """
        Principle of Inclusion-Exclusion (PIE):
        Calculates distinct vehicles across overlapping categories without double-counting:
          |A ∪ B ∪ C| = |A| + |B| + |C| - (|A ∩ B| + |A ∩ C| + |B ∩ C|) + |A ∩ B ∩ C|
        """
        naive_sum = count_a + count_b + count_c
        correct_pie_total = count_a + count_b + count_c - (ab + ac + bc) + abc

        return {
            "count_A_Speeding": count_a,
            "count_B_HeavyTrucks": count_b,
            "count_C_Violations": count_c,
            "intersection_A_B": ab,
            "intersection_A_C": ac,
            "intersection_B_C": bc,
            "intersection_A_B_C": abc,
            "naive_sum_double_counted": naive_sum,
            "correct_PIE_union": correct_pie_total,
            "overcount_error": naive_sum - correct_pie_total,
            "formula_string": (
                f"|A ∪ B ∪ C| = {count_a} + {count_b} + {count_c} - ({ab} + {ac} + {bc}) + {abc} = {correct_pie_total}"
            ),
            "explanation": (
                f"Naive summation yields {naive_sum} vehicles, which double-counts vehicles with multiple infractions. "
                f"Applying Inclusion-Exclusion yields the exact count of {correct_pie_total} unique offending vehicles."
            ),
        }

    @staticmethod
    def demonstrate_mathematical_induction(n_junctions: int = 6) -> Dict[str, Any]:
        """
        Mathematical Induction Worked Example:
        Theorem: Any connected linear arterial network of n traffic junctions contains exactly n - 1 road segments.
        Base Step: For n = 1 junction, road segments = 1 - 1 = 0. (True)
        Inductive Hypothesis: Assume true for k junctions (has k - 1 segments).
        Inductive Step: Adding the (k + 1)-th junction requires connecting it with 1 segment.
                        Total segments = (k - 1) + 1 = k = (k + 1) - 1. (Q.E.D.)
        """
        steps = [
            {
                "step_name": "Base Case (n = 1)",
                "mathematical_statement": "For n = 1 isolated intersection, number of road links E = 1 - 1 = 0.",
                "valid": True,
            },
            {
                "step_name": "Inductive Hypothesis (Assume n = k)",
                "mathematical_statement": "Assume any linear corridor of k intersections has E(k) = k - 1 road links.",
                "valid": True,
            },
            {
                "step_name": "Inductive Step (Prove for n = k + 1)",
                "mathematical_statement": (
                    "When connecting an additional (k + 1)-th junction to the linear corridor, exactly 1 new road link is added. "
                    "Thus, E(k + 1) = E(k) + 1 = (k - 1) + 1 = k = (k + 1) - 1. The formula holds for k + 1."
                ),
                "valid": True,
            },
            {
                "step_name": "Conclusion by PMI",
                "mathematical_statement": (
                    f"By the Principle of Mathematical Induction, a connected linear corridor of n junctions has n - 1 road links for all n ≥ 1. "
                    f"For n = {n_junctions}, road links = {n_junctions - 1}."
                ),
                "valid": True,
            },
        ]
        return {
            "n_junctions": n_junctions,
            "calculated_road_links": n_junctions - 1,
            "induction_steps": steps,
        }


# ============================================================================
# UNIT IV: ALGEBRAIC SYSTEMS
# ============================================================================

class Unit4AlgebraicSystems:
    """
    Implements Discrete Mathematics Unit IV:
    Binary operations, group axioms (Closure, Associativity, Identity, Inverses),
    cyclic group (Z_4, +_4), Klein 4-group, group homomorphisms,
    vector spaces & linear independence, and distinction from automata.
    """

    @staticmethod
    def verify_algebraic_structure(
        elements: List[Any],
        op_func: Callable[[Any, Any], Any],
        structure_name: str = "Candidate Structure"
    ) -> Dict[str, Any]:
        """
        Tests whether a finite set S and binary operation * satisfy:
          1. Closure
          2. Associativity
          3. Identity existence
          4. Inverse existence
          5. Commutativity
        Classifies as: Magma, Semigroup, Monoid, Group, or Abelian Group.
        """
        elem_set = set(elements)
        n = len(elements)

        # 1. Closure
        is_closed = True
        for a in elements:
            for b in elements:
                res = op_func(a, b)
                if res not in elem_set:
                    is_closed = False
                    break
            if not is_closed:
                break

        # 2. Associativity: (a * b) * c == a * (b * c)
        is_associative = is_closed
        if is_closed:
            for a in elements:
                for b in elements:
                    for c in elements:
                        if op_func(op_func(a, b), c) != op_func(a, op_func(b, c)):
                            is_associative = False
                            break
                    if not is_associative:
                        break
                if not is_associative:
                    break

        # 3. Identity element: e such that a * e == e * a == a
        identity_elem = None
        if is_closed:
            for candidate in elements:
                if all(op_func(a, candidate) == a and op_func(candidate, a) == a for a in elements):
                    identity_elem = candidate
                    break
        has_identity = identity_elem is not None

        # 4. Inverses: for every a, exists b such that a * b == b * a == e
        has_inverses = False
        inverse_map = {}
        if has_identity:
            all_have_inverse = True
            for a in elements:
                inv = None
                for b in elements:
                    if op_func(a, b) == identity_elem and op_func(b, a) == identity_elem:
                        inv = b
                        break
                if inv is not None:
                    inverse_map[a] = inv
                else:
                    all_have_inverse = False
                    break
            has_inverses = all_have_inverse

        # 5. Commutativity: a * b == b * a
        is_commutative = is_closed
        if is_closed:
            for a in elements:
                for b in elements:
                    if op_func(a, b) != op_func(b, a):
                        is_commutative = False
                        break
                if not is_commutative:
                    break

        # Classification hierarchy
        if not is_closed:
            classification = "Not a Magma (Not Closed)"
        elif not is_associative:
            classification = "Magma (Closed only)"
        elif not has_identity:
            classification = "Semigroup (Closed + Associative)"
        elif not has_inverses:
            classification = "Monoid (Semigroup + Identity)"
        elif not is_commutative:
            classification = "Group (Monoid + Inverses)"
        else:
            classification = "Abelian (Commutative) Group"

        # Generate Cayley Table
        cayley_table = []
        for a in elements:
            row = [op_func(a, b) for b in elements]
            cayley_table.append(row)

        df_cayley = pd.DataFrame(cayley_table, index=elements, columns=elements)

        return {
            "structure_name": structure_name,
            "elements": elements,
            "is_closed": is_closed,
            "is_associative": is_associative,
            "has_identity": has_identity,
            "identity_element": identity_elem,
            "has_inverses": has_inverses,
            "inverse_map": inverse_map,
            "is_commutative": is_commutative,
            "classification": classification,
            "cayley_table": df_cayley,
        }

    @staticmethod
    def demonstrate_cyclic_phase_group() -> Dict[str, Any]:
        """
        Demonstrates the Cyclic Group (Z_4, +_4):
        Phase 0: North Green
        Phase 1: East Green
        Phase 2: South Green
        Phase 3: West Green
        Binary operation: (a + b) mod 4
        """
        phases = [0, 1, 2, 3]
        phase_names = {0: "0 (North)", 1: "1 (East)", 2: "2 (South)", 3: "3 (West)"}
        res = Unit4AlgebraicSystems.verify_algebraic_structure(
            elements=phases,
            op_func=lambda a, b: (a + b) % 4,
            structure_name="(Z₄, +₄) Signal Phase Cyclic Group",
        )
        res["phase_meanings"] = phase_names
        res["generator"] = 1  # 1 is a generator since <1> = {1, 2, 3, 0}
        res["order_of_group"] = 4
        return res

    @staticmethod
    def demonstrate_klein_four_group() -> Dict[str, Any]:
        """
        Demonstrates the Klein 4-Group V_4 ≅ Z_2 × Z_2:
        Elements represent independent binary states of North-South and East-West pedestrian crossing toggles:
          (0,0): Both off
          (0,1): EW on
          (1,0): NS on
          (1,1): Both on
        Operation: Component-wise XOR / addition mod 2.
        Every non-identity element has order 2 (is its own inverse).
        """
        elements = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]

        def xor_op(s1: str, s2: str) -> str:
            b1 = (int(s1[1]), int(s1[3]))
            b2 = (int(s2[1]), int(s2[3]))
            res = ((b1[0] ^ b2[0]), (b1[1] ^ b2[1]))
            return f"({res[0]},{res[1]})"

        res = Unit4AlgebraicSystems.verify_algebraic_structure(
            elements=elements,
            op_func=xor_op,
            structure_name="Klein 4-Group V₄ (Z₂ × Z₂) Pedestrian Toggle",
        )
        return res

    @staticmethod
    def demonstrate_homomorphism() -> Dict[str, Any]:
        """
        Demonstrates a valid Group Homomorphism:
          f: (Z_4, +_4) → (Z_2, +_2)
          f(x) = x mod 2
        Preserves operation: f(a +_4 b) = f(a) +_2 f(b).
        Kernel: ker(f) = {x ∈ Z_4 | f(x) = 0} = {0, 2} (Arterial NS phases).
        """
        z4 = [0, 1, 2, 3]
        f_map = {x: x % 2 for x in z4}

        # Verify operation preservation for all pairs
        preserves_op = True
        sample_verifications = []
        for a in z4:
            for b in z4:
                lhs = f_map[(a + b) % 4]
                rhs = (f_map[a] + f_map[b]) % 2
                if lhs != rhs:
                    preserves_op = False
                if len(sample_verifications) < 4:
                    sample_verifications.append(f"f({a} +₄ {b}) = f({(a+b)%4}) = {lhs} == f({a}) +₂ f({b}) = {rhs}")

        kernel = [x for x in z4 if f_map[x] == 0]

        return {
            "source_group": "(Z₄, +₄) with phases {0, 1, 2, 3}",
            "target_group": "(Z₂, +₂) with arterial axes {0: NS Axis, 1: EW Axis}",
            "mapping": f_map,
            "preserves_operation": preserves_op,
            "sample_proofs": sample_verifications,
            "kernel_ker_f": kernel,
            "kernel_meaning": "The kernel {0, 2} maps the North and South straight green phases to the 0 (NS) axis.",
            "first_isomorphism_theorem": "Z₄ / ker(f) ≅ Z₂, collapsing 4 signal phases into 2 orthogonal arterial directions.",
        }

    @staticmethod
    def demonstrate_vector_space_linear_independence(
        vectors: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Demonstrates Vector Space & Linear Independence:
        Traffic detector flow vectors v = (q_cars, q_trucks, q_bikes)^T in R^3 over field R.
        Forms matrix M = [v_1 | v_2 | v_3] and tests determinant / rank.
        If det(M) != 0, the vectors are Linearly Independent; no detector reading is redundant.
        """
        if vectors is None:
            # 3 sensor readings: [Cars/hr, Trucks/hr, Bikes/hr]
            vectors = [
                [120.0, 15.0, 45.0],   # Sensor 1 (Approach A)
                [80.0,  30.0, 10.0],   # Sensor 2 (Approach B)
                [60.0,  10.0, 80.0],   # Sensor 3 (Approach C)
            ]

        m_matrix = np.array(vectors, dtype=float).T
        det_val = float(np.linalg.det(m_matrix))
        rank_val = int(np.linalg.matrix_rank(m_matrix))
        is_independent = rank_val == len(vectors)

        return {
            "vectors": vectors,
            "matrix_M": m_matrix.tolist(),
            "determinant": round(det_val, 4),
            "matrix_rank": rank_val,
            "dimension": len(vectors),
            "is_linearly_independent": is_independent,
            "traffic_interpretation": (
                "The flow rate sensor vectors are LINEARLY INDEPENDENT (det(M) ≠ 0). "
                "Every sensor provides distinct, non-redundant traffic dimensionality to the controller."
                if is_independent
                else "Vectors are LINEARLY DEPENDENT (det(M) = 0). At least one sensor's reading is mathematically redundant."
            ),
        }

    @staticmethod
    def explain_algebra_vs_state_automata() -> Dict[str, Any]:
        """
        Explains why ordinary signal state transitions (Red -> Green -> Yellow -> Red)
        are directed graph automata / state machines, NOT algebraic groups.
        """
        return {
            "distinction": "Algebraic Group vs. Finite State Automaton",
            "automaton_definition": (
                "Traffic signal state transitions (Red → Green → Yellow → Red) form a directed cyclic graph "
                "or Finite State Machine (Q, Σ, δ, q₀, F). Transitions depend on temporal clocks and external events."
            ),
            "why_not_a_group": (
                "In general, transition composition lacks invertible inverses: once a light switches Yellow → Red, "
                "there is no physical backward operation that reverses the time-flow of cars. "
                "Furthermore, transitions with reset semantics violate algebraic identity axioms. "
                "Only symmetric permutations of indices (e.g., Z₄ modular phase shifts) form genuine algebraic groups."
            ),
        }


# ============================================================================
# UNIT V: GRAPH THEORY
# ============================================================================

class Unit5GraphTheory:
    """
    Implements Discrete Mathematics Unit V:
    Vertices, edges, special graphs (Kn, Km,n, Cn), adjacency matrices, walk powers,
    graph isomorphism, connectivity, Eulerian patrol trails, Hamiltonian maintenance cycles,
    trees, and Kruskal's Minimum Spanning Tree (MST).
    """

    @staticmethod
    def generate_special_graphs() -> Dict[str, Any]:
        """
        Generates and analyzes special graph families:
          1. Complete Graph K_4: E = n(n - 1) / 2
          2. Complete Bipartite Graph K_{3, 3}: E = m * n (Kuratowski's utility graph)
          3. Cycle Graph C_5: 2-regular polygon
          4. 3-Regular Cubic Graph
        """
        k4 = nx.complete_graph(4)
        k33 = nx.complete_bipartite_graph(3, 3)
        c5 = nx.cycle_graph(5)

        return {
            "complete_K4": {
                "name": "Complete Graph K₄",
                "nodes": list(k4.nodes()),
                "edges_count": k4.number_of_edges(),
                "formula": "E = n(n - 1) / 2 = 4(3) / 2 = 6",
                "is_planar": True,
            },
            "bipartite_K33": {
                "name": "Complete Bipartite Graph K₃,₃ (Utilities Graph)",
                "nodes": list(k33.nodes()),
                "edges_count": k33.number_of_edges(),
                "formula": "E = m · n = 3 · 3 = 9",
                "is_planar": False,
                "kuratowski_note": "Non-planar by Kuratowski's Theorem; represents utility crossing conflicts.",
            },
            "cycle_C5": {
                "name": "Cycle Graph C₅ (Roundabout Topology)",
                "nodes": list(c5.nodes()),
                "edges_count": c5.number_of_edges(),
                "is_regular": True,
                "regular_degree": 2,
            },
        }

    @staticmethod
    def compute_adjacency_walk_powers(graph: nx.Graph, max_k: int = 3) -> Dict[str, Any]:
        """
        Theorem on Adjacency Matrix Powers:
        Entry (A^k)_{i, j} in the k-th power of adjacency matrix A gives the exact number
        of distinct walks of length k between vertex i and vertex j.
        """
        nodes = sorted(list(graph.nodes()))
        a = nx.to_numpy_array(graph, nodelist=nodes, dtype=int)

        powers = {}
        curr_power = a.copy()
        for k in range(1, max_k + 1):
            powers[f"A^{k}"] = curr_power.copy()
            curr_power = curr_power @ a

        res = {
            "nodes": nodes,
            "theorem_statement": "Entry (Aᵏ)ᵢ,ⱼ gives the exact number of walks of length k between vertex i and vertex j.",
        }
        for k in range(1, max_k + 1):
            df_k = pd.DataFrame(powers[f"A^{k}"], index=nodes, columns=nodes)
            res[f"A_{k}"] = df_k
            res[f"A^{k}"] = df_k
        return res

    @staticmethod
    def test_graph_isomorphism(g1: nx.Graph, g2: nx.Graph) -> Dict[str, Any]:
        """
        Tests Graph Isomorphism:
        Two graphs G₁ = (V₁, E₁) and G₂ = (V₂, E₂) are isomorphic (G₁ ≅ G₂)
        if there exists a bijection f: V₁ → V₂ such that (u, v) ∈ E₁ ⟺ (f(u), f(v)) ∈ E₂.
        """
        is_iso = nx.is_isomorphic(g1, g2)
        mapping = None
        if is_iso:
            em = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
            mapping = em.mapping if em.is_isomorphic() else None

        return {
            "is_isomorphic": is_iso,
            "vertex_mapping": mapping,
            "g1_nodes": list(g1.nodes()),
            "g2_nodes": list(g2.nodes()),
            "g1_degrees": sorted([d for _, d in g1.degree()]),
            "g2_degrees": sorted([d for _, d in g2.degree()]),
            "explanation": (
                "Graphs are ISOMORPHIC (G₁ ≅ G₂): structurally identical under relabeling."
                if is_iso
                else "Graphs are NOT ISOMORPHIC: topological structure differs."
            ),
        }

    @staticmethod
    def compute_minimum_spanning_tree(graph: nx.Graph) -> Dict[str, Any]:
        """
        Computes the Minimum Spanning Tree (MST) using Kruskal's Algorithm:
        Identifies the optimal, loop-free fiber-optic sensor network connecting all junctions
        with minimal cabling cost.
        Also calculates Cayley's Formula: n^(n - 2) possible labeled spanning trees on K_n.
        """
        if not nx.is_connected(graph):
            return {"error": "Graph must be connected to compute Spanning Tree"}

        mst = nx.minimum_spanning_tree(graph, algorithm="kruskal")
        n = graph.number_of_nodes()
        total_weight = sum(d.get("weight", 1.0) for _, _, d in mst.edges(data=True))
        cayley_count = n ** (n - 2) if n >= 2 else 1

        mst_edges = [
            {"from": u, "to": v, "weight": d.get("weight", 1.0)}
            for u, v, d in mst.edges(data=True)
        ]

        return {
            "mst_edges": mst_edges,
            "mst_edge_count": mst.number_of_edges(),
            "expected_tree_edges_(V-1)": n - 1,
            "total_cabling_weight": round(total_weight, 2),
            "cayley_formula_spanning_trees_Kn": cayley_count,
            "traffic_application": (
                f"Kruskal's MST connects all {n} intersections with exactly {n - 1} fiber-optic communications links, "
                f"achieving zero-loop routing and minimizing total trenching/cabling cost to {total_weight:.2f} units."
            ),
        }


# ============================================================================
# CONFLICT GRAPH MODEL (Welsh-Powell Vertex Coloring)
# ============================================================================

class ConflictGraphModel:
    """
    Constructs the intersection vehicle movement conflict graph G = (V, E)
    and applies the Welsh-Powell greedy vertex coloring algorithm
    to deduce optimal, collision-free signal phases.
    """
    MOVEMENTS = [
        "N_Straight", "N_Left",
        "S_Straight", "S_Left",
        "E_Straight", "E_Left",
        "W_Straight", "W_Left",
    ]

    CONFLICT_PAIRS = [
        ("N_Straight", "E_Straight"),
        ("N_Straight", "E_Left"),
        ("N_Straight", "W_Straight"),
        ("N_Straight", "W_Left"),
        ("N_Straight", "S_Left"),

        ("S_Straight", "E_Straight"),
        ("S_Straight", "E_Left"),
        ("S_Straight", "W_Straight"),
        ("S_Straight", "W_Left"),
        ("S_Straight", "N_Left"),

        ("E_Straight", "W_Left"),
        ("E_Straight", "N_Straight"),
        ("E_Straight", "N_Left"),
        ("E_Straight", "S_Straight"),
        ("E_Straight", "S_Left"),

        ("W_Straight", "E_Left"),
        ("W_Straight", "N_Straight"),
        ("W_Straight", "N_Left"),
        ("W_Straight", "S_Straight"),
        ("W_Straight", "S_Left"),

        ("N_Left", "E_Straight"),
        ("N_Left", "W_Straight"),
        ("N_Left", "S_Straight"),
        ("N_Left", "E_Left"),
        ("N_Left", "W_Left"),

        ("S_Left", "E_Straight"),
        ("S_Left", "W_Straight"),
        ("S_Left", "N_Straight"),
        ("S_Left", "E_Left"),
        ("S_Left", "W_Left"),
    ]

    COLOR_PALETTE = [
        "#06B6D4",  # Cyan (Phase 1)
        "#10B981",  # Emerald (Phase 2)
        "#F59E0B",  # Amber (Phase 3)
        "#EF4444",  # Coral (Phase 4)
        "#8B5CF6",  # Violet (Phase 5)
        "#EC4899",  # Pink (Phase 6)
    ]

    def __init__(self):
        self.graph = nx.Graph()
        self.build_graph()

    def build_graph(self):
        """Builds conflict graph G where edges represent collisions."""
        self.graph.clear()
        for m in self.MOVEMENTS:
            self.graph.add_node(m)
        for u, v in self.CONFLICT_PAIRS:
            if u in self.MOVEMENTS and v in self.MOVEMENTS:
                self.graph.add_edge(u, v)

    def welsh_powell_coloring(self) -> Dict[str, Any]:
        """
        Step-by-step Welsh-Powell Algorithm:
        1. Sort vertices in descending order of degrees.
        2. Assign color 1 to first vertex and sequentially to non-adjacent vertices.
        3. Repeat with new colors until all vertices are colored.
        """
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)

        coloring: Dict[str, int] = {}
        steps_log: List[Dict[str, Any]] = []
        current_color = 0

        uncolored = list(sorted_nodes)

        while uncolored:
            current_color += 1
            colored_in_this_pass = []

            for node in list(uncolored):
                can_color = True
                for cnode in colored_in_this_pass:
                    if self.graph.has_edge(node, cnode):
                        can_color = False
                        break

                if can_color:
                    coloring[node] = current_color
                    colored_in_this_pass.append(node)
                    uncolored.remove(node)

            steps_log.append({
                "color_id": current_color,
                "color_hex": self.COLOR_PALETTE[(current_color - 1) % len(self.COLOR_PALETTE)],
                "assigned_movements": colored_in_this_pass,
                "phase_name": f"Phase {current_color}: Concurrent Safe Movements",
            })

        chromatic_number = max(coloring.values()) if coloring else 0

        return {
            "sorted_degrees": [(n, degrees[n]) for n in sorted_nodes],
            "node_colors": coloring,
            "chromatic_number": chromatic_number,
            "phases": steps_log,
            "total_vertices": self.graph.number_of_nodes(),
            "total_conflict_edges": self.graph.number_of_edges(),
        }


# ============================================================================
# DISCRETE MATH LAB MASTER SUITE
# ============================================================================

class DiscreteMathLab:
    """
    Complete Discrete Mathematics laboratory suite for traffic systems.
    Encapsulates all 5 syllabus units, retaining 100% backward compatibility
    with existing methods while exposing deep discrete mathematical engines.
    """
    def __init__(self):
        self.conflict_model = ConflictGraphModel()
        self.unit1 = Unit1LogicProofs()
        self.unit2 = Unit2SetsRelations()
        self.unit3 = Unit3Combinatorics()
        self.unit4 = Unit4AlgebraicSystems()
        self.unit5 = Unit5GraphTheory()

    @staticmethod
    def get_syllabus_checklist() -> Dict[str, Dict[str, Any]]:
        return SYLLABUS_CHECKLIST

    def get_conflict_coloring(self) -> Dict[str, Any]:
        return self.conflict_model.welsh_powell_coloring()

    @staticmethod
    def verify_planarity_and_euler(num_v: int, num_e: int, num_f: int) -> Dict[str, Any]:
        """
        Verifies Euler's Polyhedral Formula for Planar Graphs:
        V - E + F = 2 (for connected planar graphs).
        Explains why grade-separated overpasses are required when non-planar.
        """
        euler_characteristic = num_v - num_e + num_f
        is_euler_valid = (euler_characteristic == 2)
        max_edges_planar = 3 * num_v - 6 if num_v >= 3 else num_e
        is_theoretically_planar = num_e <= max_edges_planar

        return {
            "V": num_v,
            "E": num_e,
            "F": num_f,
            "euler_characteristic": euler_characteristic,
            "is_valid_euler": is_euler_valid,
            "max_allowed_edges_for_planarity": max_edges_planar,
            "is_theoretically_planar": is_theoretically_planar,
            "engineering_significance": (
                "A planar graph represents roads that intersect at-grade. "
                "If edges exceed the planarity bound (E > 3V - 6), the road network cannot be built "
                "on a flat plane without grade separation (flyovers, underpasses, or bridges)."
            ),
        }

    @staticmethod
    def analyze_eulerian_and_hamiltonian(graph: nx.Graph) -> Dict[str, Any]:
        """
        Tests Eulerian and Hamiltonian properties of a road network.
        - Eulerian: Police patrol traversing every road exactly once.
        - Hamiltonian: Streetlight inspection visiting every junction exactly once.
        """
        is_connected = nx.is_connected(graph) if not graph.is_directed() else nx.is_strongly_connected(graph)

        degrees = dict(graph.degree())
        odd_degree_nodes = [n for n, d in degrees.items() if d % 2 != 0]

        is_eulerian_circuit = is_connected and len(odd_degree_nodes) == 0
        is_eulerian_path = is_connected and len(odd_degree_nodes) in [0, 2]

        n = graph.number_of_nodes()
        min_deg = min(degrees.values()) if degrees else 0
        satisfies_dirac = (n >= 3 and min_deg >= n / 2)

        return {
            "is_connected": is_connected,
            "odd_degree_count": len(odd_degree_nodes),
            "odd_degree_nodes": odd_degree_nodes,
            "has_eulerian_circuit": is_eulerian_circuit,
            "has_eulerian_path": is_eulerian_path,
            "satisfies_dirac_hamiltonian": satisfies_dirac,
            "min_degree": min_deg,
            "threshold_n_over_2": n / 2 if n > 0 else 0,
        }

    @staticmethod
    def compute_laplacian_spectral(graph: nx.Graph) -> Dict[str, Any]:
        """
        Computes Adjacency Matrix, Degree Matrix, Laplacian Matrix L = D - A,
        and Algebraic Connectivity (Fiedler eigenvalue lambda_2).
        """
        nodes = sorted(list(graph.nodes()))
        adj_matrix = nx.to_numpy_array(graph, nodelist=nodes)
        degrees = [graph.degree(n) for n in nodes]
        deg_matrix = np.diag(degrees)
        laplacian = deg_matrix - adj_matrix

        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        algebraic_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        return {
            "nodes": nodes,
            "adjacency_matrix": adj_matrix.astype(int),
            "degree_matrix": deg_matrix.astype(int),
            "laplacian_matrix": laplacian.astype(int),
            "eigenvalues": [round(float(e), 3) for e in eigenvalues],
            "fiedler_value": round(algebraic_connectivity, 4),
            "connectivity_interpretation": (
                "Algebraic connectivity > 0 confirms the network is fully connected. "
                "Higher Fiedler values indicate high resilience against road bottlenecks and network partition."
            ),
        }

    @staticmethod
    def get_viva_questions() -> List[Dict[str, str]]:
        """10 Curated Viva Voce questions and model answers across all 5 Units."""
        return [
            {
                "question": "UNIT I: How do proposition formulas and inference rules model congestion?",
                "answer": (
                    "Traffic rules are formalized as propositional implications such as ((P ∧ Q) ∨ R) → S, "
                    "where P is high density, Q is long queue, R is emergency siren, and S is green extension. "
                    "Modus Ponens deduces green extension when telemetry verifies P and Q; Modus Tollens deduces "
                    "absence of emergency vehicles when standard signal transitions are observed."
                ),
            },
            {
                "question": "UNIT II: Why is the 'Same-Lane' relation an Equivalence Relation?",
                "answer": (
                    "The relation R where (u, v) ∈ R iff lane(u) = lane(v) is Reflexive (every vehicle is in its own lane), "
                    "Symmetric (u and v share a lane implies v and u share a lane), and Transitive (u and v share, v and w share implies u and w share). "
                    "By the Fundamental Theorem of Equivalence Relations, R partitions the vehicles into disjoint equivalence classes representing lane queues."
                ),
            },
            {
                "question": "UNIT II: How does Boolean Algebra simplify emergency traffic overrides?",
                "answer": (
                    "An emergency override logic rule (E ∧ P) ∨ (E ∧ ¬P) simplifies by the Distributive Law to E ∧ (P ∨ ¬P), "
                    "by the Complement Law to E ∧ 1, and by the Identity Law to E. This proves algebraically that emergency "
                    "preemption is universally active regardless of whether peak-hour congestion (P) is present."
                ),
            },
            {
                "question": "UNIT III: How does the Pigeonhole Principle guarantee lane congestion?",
                "answer": (
                    "If n vehicles are detected in an intersection zone with k lanes where n > k, the Pigeonhole Principle "
                    "guarantees that at least one lane contains ⌈n / k⌉ vehicles. If lane capacity is C < ⌈n / k⌉, "
                    "spillback is mathematically guaranteed without needing camera identification of specific lane lines."
                ),
            },
            {
                "question": "UNIT III: Why is the Principle of Inclusion-Exclusion essential for vehicle classification?",
                "answer": (
                    "When counting vehicles exhibiting overlapping violations (e.g. Speeding ∪ Heavy ∪ Off-Peak), naive summation "
                    "double-counts multi-infraction vehicles. Inclusion-Exclusion |A ∪ B ∪ C| = Σ|A| - Σ|A ∩ B| + |A ∩ B ∩ C| "
                    "removes intersections, preventing inaccurate tax/toll revenue calculation."
                ),
            },
            {
                "question": "UNIT IV: Why does cyclic signal phase shifting form an Abelian Group (Z₄, +₄)?",
                "answer": (
                    "Phase indices {0, 1, 2, 3} under addition mod 4 satisfy Closure ((a + b) mod 4 ∈ {0..3}), Associativity, "
                    "Identity (0 shifts nothing), Inverses ((4 - a) mod 4 returns to origin), and Commutativity (a + b = b + a). "
                    "It is therefore a finite cyclic Abelian group isomorphic to Z₄."
                ),
            },
            {
                "question": "UNIT IV: How do algebraic groups differ from ordinary traffic signal state automata?",
                "answer": (
                    "Signal state transitions (Red → Green → Yellow → Red) form a directed cyclic graph or Finite State Machine. "
                    "They do NOT form groups because physical transitions lack general time-reversible inverses (cars cannot "
                    "un-react to a yellow light), and timeout resets violate algebraic identity axioms."
                ),
            },
            {
                "question": "UNIT V: What is the significance of the Welsh-Powell Chromatic Number χ(G)?",
                "answer": (
                    "Vertices represent turning movements and edges represent intersecting collision paths. "
                    "A proper vertex coloring partitions movements into independent sets where each color represents a "
                    "concurrent safe green phase. The chromatic number χ(G) is the theoretical minimum number of phases required."
                ),
            },
            {
                "question": "UNIT V: How does Euler's Formula (V - E + F = 2) prove the need for flyovers?",
                "answer": (
                    "Planar graphs (which satisfy V - E + F = 2 and E ≤ 3V - 6) represent roads that can intersect at ground level. "
                    "When road density exceeds this bound, or contains K₅ or K₃,₃ Kuratowski subgraphs, planar geometry is impossible, "
                    "proving that grade-separated overpasses, bridges, or tunnels are mathematically mandatory."
                ),
            },
            {
                "question": "UNIT V: How do Eulerian vs Hamiltonian traversals differ in municipal operations?",
                "answer": (
                    "An Eulerian path visits every road link (edge) exactly once, optimized for police patrol cars and street-cleaning sweepers. "
                    "A Hamiltonian cycle visits every intersection (vertex) exactly once, optimized for signal hardware maintenance technicians "
                    "and camera calibration teams solving the Traveling Salesperson Problem."
                ),
            },
        ]
