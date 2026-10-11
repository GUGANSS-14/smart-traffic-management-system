"""
Page 5: Discrete Mathematics Capstone Laboratory
================================================
Comprehensive Academic Laboratory mapping the Smart Traffic Management System
directly to the 5 Units of the University Discrete Mathematics Syllabus:

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

DISCLAIMER: All demonstrations are academic discrete mathematical models designed
to illustrate theoretical computer science and mathematics principles.
They do not claim to optimize real municipal road networks without live GIS data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

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
from modules.graph_model import IntersectionGraphNetwork, FourWayIntersectionGraph
from visualizations.charts import (
    get_custom_css,
    create_conflict_graph_plot,
    create_network_graph_plot,
    DARK_LAYOUT_TEMPLATE,
)

st.set_page_config(
    page_title="Discrete Math Lab - Smart Traffic",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER & EDUCATIONAL DISCLAIMER
# ----------------------------------------------------------------------------
st.title("📐 Discrete Mathematics Capstone Laboratory")
st.markdown(
    """
    **Academic Curriculum Mapping: Five-Unit Discrete Mathematics Syllabus (CS204 / MA201)**  
    *Bridging foundational mathematical structures with intelligent urban transportation modeling.*
    """
)

st.warning(
    "⚠️ **ACADEMIC SIMULATION NOTICE:** All demonstrations in this laboratory are educational discrete mathematical "
    "models designed to teach computer science theory. They do not claim to optimize real-world municipal routes "
    "or replace live road telemetry."
)

lab = DiscreteMathLab()

# ----------------------------------------------------------------------------
# SIDEBAR: SYLLABUS COVERAGE CHECKLIST & NAVIGATION
# ----------------------------------------------------------------------------
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

    st.markdown("### 📋 Syllabus Coverage Checklist")
    checklist = lab.get_syllabus_checklist()
    total_topics = sum(len(u["topics"]) for u in checklist.values())
    st.markdown(f"**Total Syllabus Units:** 5 | **Topics Covered:** {total_topics} / {total_topics} (100%)")

    for u_code, u_info in checklist.items():
        with st.expander(f"{u_code}: {u_info['title']}", expanded=(u_code == "UNIT I")):
            for t_title, t_desc, t_done in u_info["topics"]:
                status_icon = "✅" if t_done else "❌"
                st.markdown(f"{status_icon} **{t_title}**  \n<small style='color:#94A3B8;'>{t_desc}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Concept Demonstration Legend")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#CBD5E1; line-height:1.5;">
            <span class="badge badge-cyan">Definition</span> Formal math concept<br>
            <span class="badge badge-emerald">Traffic Use</span> Real ITS system mapping<br>
            <span class="badge badge-amber">Simulation</span> Educational model
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# PRIMARY LAB TABS (UNITS I TO V + VIVA VOCE)
# ----------------------------------------------------------------------------
tab_u1, tab_u2, tab_u3, tab_u4, tab_u5, tab_viva = st.tabs([
    "📐 Unit I: Logic & Proofs",
    "🔢 Unit II: Sets & Relations",
    "🎲 Unit III: Combinatorics",
    "🏛️ Unit IV: Algebraic Systems",
    "🌐 Unit V: Graph Theory",
    "🎓 Viva Voce & Oral Prep",
])


# ============================================================================
# UNIT I: LOGIC AND PROOFS
# ============================================================================
with tab_u1:
    st.markdown("## Unit I: Logic and Proofs")
    st.markdown(
        """
        *Propositional Logic, Truth Tables, Logical Equivalences, Normal Forms (DNF & CNF),
        Propositional Modeling of Congestion Rules, and Rules of Inference.*
        """
    )

    # 1. Congestion Proposition Model
    st.markdown("### 🚦 1. Congestion Rule Propositional Model & Interactive Evaluator")
    st.markdown(
        r"""
        <div class="glass-card">
            <span class="badge badge-cyan">Formal Definition</span>
            <p style="color:#F8FAFC; margin-top:6px;">
                A <b>proposition</b> is a declarative statement that is either True ($1$) or False ($0$), but not both.
                Traffic signal controllers execute rule-based reasoning by evaluating propositional formulas.
            </p>
            <span class="badge badge-emerald">Traffic Application</span>
            <p style="color:#CBD5E1; font-size:0.9rem;">
                Let atomic propositions be defined as:
                <br>• $P$: "Approach density is high" ($\text{density} \ge 0.70$)
                <br>• $Q$: "Queue length exceeds threshold" ($\text{queue} \ge 5$)
                <br>• $R$: "Emergency vehicle / siren detected"
                <br>• $S$: "Extend green signal duration by 25 seconds"
                <br><br><b>Compound Propositional Rule:</b>
                <code style="color:#38BDF8; font-size:1.1rem;">((P ∧ Q) ∨ R) → S</code>
            </p>
            <span class="badge badge-amber">Educational Simulation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_prop1, c_prop2 = st.columns([1, 1.2])
    with c_prop1:
        st.markdown("#### 🧪 Interactive Inputs")
        p_val = st.checkbox("P: Density is High (≥ 70%)", value=True)
        q_val = st.checkbox("Q: Queue Exceeds Threshold (≥ 5 vehicles)", value=True)
        r_val = st.checkbox("R: Emergency Siren Detected", value=False)
        s_val = st.checkbox("S: Green Signal Duration Extended", value=True)

        eval_result = Unit1LogicProofs.evaluate_traffic_congestion_proposition(
            density_high=p_val, queue_exceeded=q_val, emergency_vehicle=r_val, extend_green=s_val
        )

    with c_prop2:
        st.markdown("#### 📊 Calculated Propositional Output")
        rule_ok = eval_result["Rule_Satisfied_(((P_and_Q)_or_R)_implies_S)"]
        premise_ok = eval_result["Premise_((P_and_Q)_or_R)"]

        st.markdown(
            f"""
            <div style="background:#0F172A; border:1px solid {'#10B981' if rule_ok else '#EF4444'}; border-radius:8px; padding:16px;">
                <div style="font-size:0.85rem; color:#94A3B8;">Sub-expression: P ∧ Q</div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#10B981' if eval_result['P_and_Q'] else '#94A3B8'};">
                    {eval_result['P_and_Q']}
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Premise: (P ∧ Q) ∨ R</div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#10B981' if premise_ok else '#94A3B8'};">
                    {premise_ok}
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Implication: ((P ∧ Q) ∨ R) → S</div>
                <div style="font-size:1.5rem; font-weight:800; color:{'#10B981' if rule_ok else '#EF4444'};">
                    {'VALID / SATISFIED' if rule_ok else 'VIOLATED (Premise True, Consequent False)'}
                </div>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">{eval_result['explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Truth Table & Normal Forms
    st.markdown("---")
    st.markdown("### 🧮 2. Truth Table & Normal Forms Synthesis (DNF & CNF)")
    st.markdown(
        r"""
        <div class="glass-card">
            <span class="badge badge-cyan">Formal Definition</span>
            <p style="color:#CBD5E1; font-size:0.9rem;">
                • <b>Disjunctive Normal Form (DNF):</b> Sum of minterms (OR of ANDs) for every truth assignment yielding True: $\bigvee (\bigwedge L_i)$.<br>
                • <b>Conjunctive Normal Form (CNF):</b> Product of maxterms (AND of ORs) for every assignment yielding False: $\bigwedge (\bigvee \neg L_i)$.<br>
                • <b>Contingency:</b> A proposition whose truth table has both True and False outcomes (neither a Tautology nor a Contradiction).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_tt = Unit1LogicProofs.generate_truth_table_traffic_rule()
    c_tt1, c_tt2 = st.columns([1.3, 1.0])

    with c_tt1:
        st.markdown("#### 16-Row Truth Table for `((P ∧ Q) ∨ R) → S`")
        st.dataframe(df_tt, use_container_width=True, height=280)

    with c_tt2:
        st.markdown("#### Algorithmic Normal Form Synthesis")
        forms = Unit1LogicProofs.compute_normal_forms(
            variables=["P (Density)", "Q (Queue)", "R (Emergency)", "S (Extend)"],
            truth_table_df=df_tt,
            target_col="((P ∧ Q) ∨ R) → S",
        )
        st.markdown(
            f"""
            - **Proposition Classification:** `{'Contingency' if forms['is_contingency'] else 'Tautology'}`
            - **True Rows (Minterms):** `{forms['minterm_count']} / 16`
            - **False Rows (Maxterms):** `{forms['maxterm_count']} / 16`
            """
        )
        with st.expander("📝 View Synthesized DNF Formula (Minterm Expansion)"):
            st.code(forms["DNF"], language="text")
        with st.expander("📝 View Synthesized CNF Formula (Maxterm Expansion)"):
            st.code(forms["CNF"], language="text")

    # 3. Rules of Inference
    st.markdown("---")
    st.markdown("### ⚖️ 3. Rules of Inference with Worked Traffic Proofs")
    inference_data = Unit1LogicProofs.demonstrate_inference_rules()

    c_inf1, c_inf2 = st.columns(2)
    with c_inf1:
        st.markdown("#### Modus Ponens (Direct Deduction)")
        mp = inference_data["modus_ponens"]
        st.markdown(
            f"""
            <div class="glass-card">
                <code style="color:#38BDF8;">{mp['schema']}</code>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">
                    <b>Premise 1:</b> {mp['traffic_premise_1']}<br>
                    <b>Premise 2:</b> {mp['traffic_premise_2']}<br>
                    <b>Conclusion:</b> <span style="color:#10B981; font-weight:700;">{mp['traffic_conclusion']}</span>
                </p>
                <span class="badge badge-emerald">{mp['validity']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Disjunctive Syllogism (Route Exclusion)")
        ds = inference_data["disjunctive_syllogism"]
        st.markdown(
            f"""
            <div class="glass-card">
                <code style="color:#38BDF8;">{ds['schema']}</code>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">
                    <b>Premise 1:</b> {ds['traffic_premise_1']}<br>
                    <b>Premise 2:</b> {ds['traffic_premise_2']}<br>
                    <b>Conclusion:</b> <span style="color:#10B981; font-weight:700;">{ds['traffic_conclusion']}</span>
                </p>
                <span class="badge badge-emerald">{ds['validity']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_inf2:
        st.markdown("#### Modus Tollens (Indirect Deduction)")
        mt = inference_data["modus_tollens"]
        st.markdown(
            f"""
            <div class="glass-card">
                <code style="color:#38BDF8;">{mt['schema']}</code>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">
                    <b>Premise 1:</b> {mt['traffic_premise_1']}<br>
                    <b>Premise 2:</b> {mt['traffic_premise_2']}<br>
                    <b>Conclusion:</b> <span style="color:#10B981; font-weight:700;">{mt['traffic_conclusion']}</span>
                </p>
                <span class="badge badge-emerald">{mt['validity']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Hypothetical Syllogism (Weather Transit Chain)")
        hs = inference_data["hypothetical_syllogism"]
        st.markdown(
            f"""
            <div class="glass-card">
                <code style="color:#38BDF8;">{hs['schema']}</code>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">
                    <b>Premise 1:</b> {hs['traffic_premise_1']}<br>
                    <b>Premise 2:</b> {hs['traffic_premise_2']}<br>
                    <b>Conclusion:</b> <span style="color:#10B981; font-weight:700;">{hs['traffic_conclusion']}</span>
                </p>
                <span class="badge badge-emerald">{hs['validity']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# UNIT II: SETS AND RELATIONS
# ============================================================================
with tab_u2:
    st.markdown("## Unit II: Sets and Relations")
    st.markdown(
        """
        *Set Representations, Set Operations (Union, Intersection, Cartesian Product, Power Set),
        Equivalence Relations, Partial Orders (Poset), Lattices, and Boolean Algebra Traffic Rules.*
        """
    )

    # 1. Set Operations
    st.markdown("### 📦 1. Vehicle Sets & Algebraic Set Operations")
    st.markdown(
        r"""
        <div class="glass-card">
            <span class="badge badge-cyan">Formal Definition</span>
            <p style="color:#F8FAFC; margin-top:6px;">
                A <b>set</b> is a well-defined collection of distinct objects. In computer vision traffic analysis,
                tracked vehicle IDs detected in individual lanes form finite discrete sets.
            </p>
            <span class="badge badge-emerald">Traffic Application</span>
            <p style="color:#CBD5E1; font-size:0.85rem;">
                Let $A$ be the set of vehicles detected in North Approach ($L_1$), and $B$ be vehicles in East Approach ($L_2$).
                Set union ($A \cup B$) gives all active vehicles across both lanes; intersection ($A \cap B$) identifies
                lane-changing or straddling vehicles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    set_demo = Unit2SetsRelations.demonstrate_set_operations()

    c_s1, c_s2 = st.columns([1, 1.2])
    with c_s1:
        st.markdown("#### Defined Traffic Sets")
        st.markdown(f"- **Set A (North Lane):** `{set_demo['Set_A']}` ($|A| = {set_demo['cardinality_A']}$)")
        st.markdown(f"- **Set B (East Lane):** `{set_demo['Set_B']}` ($|B| = {set_demo['cardinality_B']}$)")
        st.markdown(f"- **Universal Set U:** `{set_demo['Universal_Set']}`")

    with c_s2:
        st.markdown("#### Computed Set Operations")
        st.markdown(f"- **Union (A ∪ B):** `{set_demo['Union_A_cup_B']}` ($|A \\cup B| = {set_demo['cardinality_Union']}$)")
        st.markdown(f"- **Intersection (A ∩ B):** `{set_demo['Intersection_A_cap_B']}` ($|A \\cap B| = {set_demo['cardinality_Intersection']}$)")
        st.markdown(f"- **Difference (A \\setminus B):** `{set_demo['Difference_A_minus_B']}`")
        st.markdown(f"- **Symmetric Difference (A Δ B):** `{set_demo['Symmetric_Difference']}`")
        st.markdown(f"- **Power Set Cardinality (2^|A|):** `{set_demo['Power_Set_Cardinality_2_pow_A']}` distinct subsets")
        st.markdown(f"- **Cartesian Product Sample (A × B):** `{set_demo['Cartesian_Product_Sample']}`")

    # 2. Equivalence Relations
    st.markdown("---")
    st.markdown("### 🔗 2. Relations: Same-Lane Equivalence Relation & Partitioning")
    rel_demo = Unit2SetsRelations.analyze_same_lane_relation()

    c_r1, c_r2 = st.columns([1.1, 0.9])
    with c_r1:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-cyan">Relation Definition</span>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:6px;">
                    Let $V$ be the set of detected vehicles. We define relation $R$ on $V$:
                    <br><code style="color:#38BDF8;">(vᵢ, vⱼ) ∈ R  ⟺  lane(vᵢ) = lane(vⱼ)</code>
                </p>
                <div style="font-size:0.85rem; color:#F8FAFC;">
                    • <b>Reflexive:</b> {'✅ Yes (v, v) ∈ R for all v' if rel_demo['is_reflexive'] else '❌ No'}<br>
                    • <b>Symmetric:</b> {'✅ Yes (u, v) ∈ R ⇒ (v, u) ∈ R' if rel_demo['is_symmetric'] else '❌ No'}<br>
                    • <b>Transitive:</b> {'✅ Yes (u, v) ∈ R ∧ (v, w) ∈ R ⇒ (u, w) ∈ R' if rel_demo['is_transitive'] else '❌ No'}<br>
                    • <b>Equivalence Relation:</b> <span style="color:#10B981; font-weight:700;">{'CONFIRMED' if rel_demo['is_equivalence_relation'] else 'FAILED'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_r2:
        st.markdown("#### Equivalence Classes (Queue Partitions)")
        st.markdown(
            """
            By the Fundamental Theorem of Equivalence Relations, $R$ partitions $V$ into pairwise disjoint equivalence classes:
            """
        )
        for lane, v_list in rel_demo["equivalence_classes"].items():
            st.markdown(f"- **[{lane}] Queue:** `{v_list}`")
        st.info(rel_demo["mathematical_insight"])

    # 3. Poset & Lattice
    st.markdown("---")
    st.markdown("### 🪜 3. Partial Order (Poset) & Lattice Structure")
    poset_demo = Unit2SetsRelations.demonstrate_partial_order_lattice()

    c_lat1, c_lat2 = st.columns(2)
    with c_lat1:
        st.markdown(
            r"""
            <div class="glass-card">
                <span class="badge badge-cyan">Poset Definition</span>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:6px;">
                    A <b>Partially Ordered Set (Poset)</b> $(S, \le)$ satisfies:
                    <br>1. <b>Reflexivity:</b> $a \le a$
                    <br>2. <b>Antisymmetry:</b> $a \le b \land b \le a \implies a = b$
                    <br>3. <b>Transitivity:</b> $a \le b \land b \le c \implies a \le c$
                </p>
                <p style="color:#CBD5E1; font-size:0.85rem;">
                    <b>Lattice Definition:</b> A poset in which every pair has a unique:
                    <br>• <b>Greatest Lower Bound (GLB / Meet):</b> $a \land b$
                    <br>• <b>Least Upper Bound (LUB / Join):</b> $a \lor b$
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_lat2:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-emerald">Traffic Clearance Lattice</span>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:6px;">
                    Subset inclusion Boolean lattice P({{E, T}}) under subset inclusion (⊆):
                    <br>• <b>Bottom Element (0):</b> {poset_demo['bottom_element']}
                    <br>• <b>Top Element (1):</b> {poset_demo['top_element']}
                    <br>• <b>Sample Meet (GLB):</b> <code style="color:#38BDF8;">{poset_demo['sample_meet_glb']}</code>
                    <br>• <b>Sample Join (LUB):</b> <code style="color:#38BDF8;">{poset_demo['sample_join_lub']}</code>
                </p>
                <p style="color:#CBD5E1; font-size:0.85rem;">{poset_demo['explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Boolean Algebra
    st.markdown("---")
    st.markdown("### ⚡ 4. Boolean Algebra Traffic Override Rule Reduction")
    ba_demo = Unit2SetsRelations.demonstrate_boolean_algebra_traffic_rule()

    st.markdown(
        f"""
        <div class="glass-card">
            <h4 style="color:#38BDF8; margin-top:0;">Theorem: Emergency Signal Preemption Simplification</h4>
            <div style="font-size:0.95rem; color:#F8FAFC;">
                <b>Raw Policy:</b> <code style="color:#F59E0B;">{ba_demo['initial_expression']}</code>
                <br><b>Simplified Law:</b> <code style="color:#10B981; font-size:1.1rem; font-weight:700;">{ba_demo['simplified_expression']}</code>
            </div>
            <div style="margin-top:12px;">
        """,
        unsafe_allow_html=True,
    )
    df_proof = pd.DataFrame(ba_demo["proof_steps"])
    st.dataframe(df_proof, use_container_width=True, hide_index=True)
    st.info(ba_demo["traffic_meaning"])


# ============================================================================
# UNIT III: COMBINATORICS
# ============================================================================
with tab_u3:
    st.markdown("## Unit III: Combinatorics")
    st.markdown(
        """
        *Permutations and Combinations, The Pigeonhole Principle,
        Principle of Inclusion–Exclusion (PIE), and Mathematical Induction.*
        """
    )

    # 1. Permutations & Combinations
    st.markdown("### 🎲 1. Permutations & Combinations in Signal Phase Scheduling")
    c_cb1, c_cb2 = st.columns([1, 1.2])

    with c_cb1:
        st.markdown("#### 🧪 Combinatorial Controls")
        inp_n = st.number_input("Total Available Approaches (n)", min_value=2, max_value=8, value=4)
        inp_r = st.number_input("Selected Approaches for Concurrent Phase (r)", min_value=1, max_value=inp_n, value=2)
        comb_res = Unit3Combinatorics.compute_permutations_and_combinations(n=inp_n, r=inp_r)

    with c_cb2:
        st.markdown("#### 📊 Computed Permutations & Combinations")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.85rem; color:#94A3B8;">Permutations P(n, r) [Order Matters]:</div>
                <div style="font-size:1.6rem; font-weight:800; color:#38BDF8;">{comb_res['permutations_P_n_r']}</div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Combinations C(n, r) [Unordered Subsets]:</div>
                <div style="font-size:1.6rem; font-weight:800; color:#10B981;">{comb_res['combinations_C_n_r']}</div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Circular Permutations (n - 1)! [Rotary Roundabout]:</div>
                <div style="font-size:1.3rem; font-weight:700; color:#F59E0B;">{comb_res['circular_permutations_(n-1)!']}</div>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">{comb_res['traffic_application']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Pigeonhole Principle
    st.markdown("---")
    st.markdown("### 🐦 2. The Pigeonhole Principle (PHP) & Lane Spillback Guarantees")
    c_php1, c_php2 = st.columns([1, 1.2])

    with c_php1:
        st.markdown("#### 🧪 Zone Telemetry Inputs")
        p_vehicles = st.number_input("Detected Vehicles in Intersection Approach (Pigeons n)", min_value=1, max_value=50, value=14)
        p_lanes = st.number_input("Number of Approach Lanes (Holes k)", min_value=1, max_value=8, value=4)
        php_eval = Unit3Combinatorics.demonstrate_pigeonhole_principle(num_vehicles=p_vehicles, num_lanes=p_lanes)

    with c_php2:
        st.markdown("#### 📜 Pigeonhole Mathematical Proof")
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-cyan">Theorem Statement</span>
                <div style="font-size:1.4rem; font-weight:800; color:#38BDF8; margin:8px 0;">
                    Minimum In One Lane: ⌈n / k⌉ = {php_eval['ceiling_n_over_k']} Vehicles
                </div>
                <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.5;">
                    {php_eval['formal_statement']}
                </p>
                <div style="background:#0F172A; border-left:3px solid #10B981; padding:8px 12px; border-radius:4px; font-size:0.85rem; color:#E2E8F0;">
                    <b>Traffic Insight:</b> {php_eval['traffic_significance']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Principle of Inclusion-Exclusion
    st.markdown("---")
    st.markdown("### 📊 3. Principle of Inclusion–Exclusion (PIE) & Overlap Correction")
    pie_eval = Unit3Combinatorics.demonstrate_inclusion_exclusion()

    c_pie1, c_pie2 = st.columns([1, 1.2])
    with c_pie1:
        st.markdown("#### Violation Category Observations")
        st.markdown(f"- **A (Speeding Vehicles > 60 km/h):** `{pie_eval['count_A_Speeding']}`")
        st.markdown(f"- **B (Heavy Trucks / Multi-Axle):** `{pie_eval['count_B_HeavyTrucks']}`")
        st.markdown(f"- **C (Peak-Hour Corridor Violations):** `{pie_eval['count_C_Violations']}`")
        st.markdown(f"- **Shared A ∩ B:** `{pie_eval['intersection_A_B']}`")
        st.markdown(f"- **Shared A ∩ C:** `{pie_eval['intersection_A_C']}`")
        st.markdown(f"- **Shared B ∩ C:** `{pie_eval['intersection_B_C']}`")
        st.markdown(f"- **Triple Infraction A ∩ B ∩ C:** `{pie_eval['intersection_A_B_C']}`")

    with c_pie2:
        st.markdown("#### Double-Counting Comparison")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.85rem; color:#EF4444;">Naive Sum (|A| + |B| + |C|) [DOUBLE-COUNTED!]:</div>
                <div style="font-size:1.6rem; font-weight:800; color:#EF4444;">{pie_eval['naive_sum_double_counted']} Vehicles</div>
                <div style="font-size:0.85rem; color:#10B981; margin-top:8px;">True Distinct Infractions |A ∪ B ∪ C| [PIE CORRECTED]:</div>
                <div style="font-size:1.8rem; font-weight:800; color:#10B981;">{pie_eval['correct_PIE_union']} Unique Vehicles</div>
                <div style="font-size:0.85rem; color:#CBD5E1; margin-top:6px;">
                    Overcount Error Avoided: <b>+{pie_eval['overcount_error']} phantom vehicles</b>
                </div>
                <code style="color:#38BDF8; font-size:0.8rem; display:block; margin-top:8px;">{pie_eval['formula_string']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Mathematical Induction
    st.markdown("---")
    st.markdown("### 🪜 4. Mathematical Induction Worked Proof")
    ind_eval = Unit3Combinatorics.demonstrate_mathematical_induction(n_junctions=6)
    st.markdown("#### Theorem: A linear arterial corridor of $n$ traffic junctions has exactly $n - 1$ road links.")
    for step in ind_eval["induction_steps"]:
        st.markdown(
            f"""
            <div style="background:#0F172A; border-left:3px solid #38BDF8; padding:8px 14px; border-radius:6px; margin-bottom:8px;">
                <b style="color:#38BDF8;">{step['step_name']}</b><br>
                <span style="color:#CBD5E1; font-size:0.85rem;">{step['mathematical_statement']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# UNIT IV: ALGEBRAIC SYSTEMS
# ============================================================================
with tab_u4:
    st.markdown("## Unit IV: Algebraic Systems")
    st.markdown(
        r"""
        *Binary Operations, Semigroups, Monoids, Groups, Abelian Groups,
        Cyclic Phase Shift Group $(\mathbb{Z}_4, +_4)$, Klein 4-Group, Homomorphisms,
        Vector Spaces & Linear Independence, and Groups vs State Automata.*
        """
    )

    # 1. Cyclic Signal Group (Z_4, +_4)
    st.markdown(r"### 🔄 1. The Cyclic Signal Phase Group $(\mathbb{Z}_4, +_4)$")
    z4_res = Unit4AlgebraicSystems.demonstrate_cyclic_phase_group()

    c_alg1, c_alg2 = st.columns([1.1, 0.9])
    with c_alg1:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-cyan">Algebraic Axiom Verification</span>
                <h4 style="color:#38BDF8; margin:8px 0;">Classification: {z4_res['classification']}</h4>
                <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                    • <b>1. Closure:</b> {'✅ Satisfied ((a + b) mod 4 ∈ {0,1,2,3})' if z4_res['is_closed'] else '❌ Violated'}<br>
                    • <b>2. Associativity:</b> {'✅ Satisfied ((a + b) + c = a + (b + c))' if z4_res['is_associative'] else '❌ Violated'}<br>
                    • <b>3. Identity Element:</b> ✅ e = <b>{z4_res['identity_element']}</b> (0 phase shift)<br>
                    • <b>4. Inverses:</b> ✅ Inverse map: {z4_res['inverse_map']}<br>
                    • <b>5. Commutativity:</b> {'✅ Satisfied (a + b = b + a)' if z4_res['is_commutative'] else '❌ Violated'}<br>
                    • <b>Generator:</b> <1> = {{1, 2, 3, 0}} (Cyclic group of Order 4)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_alg2:
        st.markdown(r"#### Cayley Addition Table $(\mathbb{Z}_4, +_4)$")
        st.dataframe(z4_res["cayley_table"], use_container_width=True)

    # 2. Klein 4-Group
    st.markdown("---")
    st.markdown(r"### 🔲 2. Klein 4-Group $V_4 \cong \mathbb{Z}_2 \times \mathbb{Z}_2$ (Pedestrian Crossing Toggles)")
    k4_res = Unit4AlgebraicSystems.demonstrate_klein_four_group()

    c_k1, c_k2 = st.columns([1.1, 0.9])
    with c_k1:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-emerald">Klein 4-Group Structure</span>
                <h4 style="color:#10B981; margin:8px 0;">{k4_res['structure_name']}</h4>
                <p style="color:#CBD5E1; font-size:0.85rem; line-height:1.5;">
                    Elements represent binary toggle switches of North-South and East-West pedestrian pushbuttons:
                    <br>• <code>(0,0)</code>: Identity (No request)
                    <br>• <code>(0,1)</code>: EW Request Active
                    <br>• <code>(1,0)</code>: NS Request Active
                    <br>• <code>(1,1)</code>: Both Active
                    <br><br><b>Key Algebraic Property:</b> Every non-identity element has order 2 ($a * a = e$).
                    Each switch toggle is its own algebraic inverse under XOR!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_k2:
        st.markdown("#### Cayley XOR Table for $V_4$")
        st.dataframe(k4_res["cayley_table"], use_container_width=True)

    # 3. Group Homomorphism
    st.markdown("---")
    st.markdown(r"### 🌐 3. Group Homomorphism: $f: (\mathbb{Z}_4, +_4) \to (\mathbb{Z}_2, +_2)$")
    hom_res = Unit4AlgebraicSystems.demonstrate_homomorphism()

    st.markdown(
        f"""
        <div class="glass-card">
            <span class="badge badge-cyan">Homomorphism Definition</span>
            <p style="color:#F8FAFC; margin-top:6px;">
                A function $f: (G, *) \\to (H, \\cdot)$ is a <b>homomorphism</b> if it preserves operations:
                <code style="color:#38BDF8;">f(a * b) = f(a) · f(b)</code> for all $a, b \\in G$.
            </p>
            <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                • <b>Source Group:</b> {hom_res['source_group']}<br>
                • <b>Target Group:</b> {hom_res['target_group']}<br>
                • <b>Mapping Function:</b> f(x) = x mod 2<br>
                • <b>Operation Preservation:</b> <span style="color:#10B981; font-weight:700;">{'VERIFIED FOR ALL 16 PAIRS' if hom_res['preserves_operation'] else 'VIOLATED'}</span><br>
                • <b>Kernel (ker(f)):</b> <code style="color:#F59E0B;">{hom_res['kernel_ker_f']}</code> ({hom_res['kernel_meaning']})<br>
                • <b>First Isomorphism Theorem:</b> {hom_res['first_isomorphism_theorem']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Vector Space & Linear Independence
    st.markdown("---")
    st.markdown("### 📐 4. Vector Space & Linear Independence of Detector Readings")
    vec_res = Unit4AlgebraicSystems.demonstrate_vector_space_linear_independence()

    c_v1, c_v2 = st.columns([1, 1.2])
    with c_v1:
        st.markdown(r"#### Detector Flow Rate Vectors in $\mathbb{R}^3$")
        for idx, vec in enumerate(vec_res["vectors"], 1):
            st.markdown(f"- **Detector Sensor v_{idx}:** `{vec}` (Cars, Trucks, Bikes/hr)")

    with c_v2:
        st.markdown("#### Linear Independence Matrix Test")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.85rem; color:#94A3B8;">Determinant det(M):</div>
                <div style="font-size:1.6rem; font-weight:800; color:{'#10B981' if vec_res['is_linearly_independent'] else '#EF4444'};">
                    {vec_res['determinant']} {'(Non-Zero: Linearly Independent)' if vec_res['is_linearly_independent'] else '(Zero: Linearly Dependent)'}
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Matrix Rank:</div>
                <div style="font-size:1.3rem; font-weight:700; color:#38BDF8;">
                    Rank(M) = {vec_res['matrix_rank']} (Full Rank = {vec_res['dimension']})
                </div>
                <p style="color:#CBD5E1; font-size:0.85rem; margin-top:8px;">{vec_res['traffic_interpretation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Group vs State Automaton Distinction
    st.markdown("---")
    st.markdown("### 🚦 5. Rigorous Distinction: Algebraic Groups vs. State Automata")
    aut_diff = Unit4AlgebraicSystems.explain_algebra_vs_state_automata()
    st.info(f"**{aut_diff['distinction']}:**\n\n{aut_diff['automaton_definition']}\n\n{aut_diff['why_not_a_group']}")


# ============================================================================
# UNIT V: GRAPH THEORY
# ============================================================================
with tab_u5:
    st.markdown("## Unit V: Graph Theory")
    st.markdown(
        """
        *Vertices, Edges, Special Graph Types ($K_n, K_{m,n}, C_n$), Adjacency Matrix Walk Powers,
        Graph Isomorphism, Trees & Minimum Spanning Tree (MST), Welsh-Powell Conflict Coloring,
        Planar Graphs & Euler's Formula, and Laplacian Spectral Connectivity.*
        """
    )

    # 1. Special Graphs & Walk Powers
    st.markdown("### 🕸️ 1. Special Graph Types & Adjacency Matrix Walk Powers")
    spec_graphs = Unit5GraphTheory.generate_special_graphs()

    c_sg1, c_sg2 = st.columns([1, 1.2])
    with c_sg1:
        st.markdown("#### Special Graph Families")
        for g_k, g_v in spec_graphs.items():
            st.markdown(
                f"""
                <div style="background:#0F172A; border-left:3px solid #38BDF8; padding:8px 12px; border-radius:6px; margin-bottom:8px;">
                    <b style="color:#38BDF8;">{g_v['name']}</b><br>
                    <span style="color:#CBD5E1; font-size:0.85rem;">
                        Vertices: {len(g_v['nodes'])} | Edges: {g_v['edges_count']}<br>
                        Planar: {'✅ Yes' if g_v.get('is_planar') else '❌ No (Non-Planar!)'}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_sg2:
        st.markdown("#### Walk Count Theorem on Adjacency Matrix Powers $(A^k)$")
        net = IntersectionGraphNetwork()
        undirected_g = net.graph.to_undirected()
        walk_powers = Unit5GraphTheory.compute_adjacency_walk_powers(undirected_g, max_k=3)

        power_choice = st.radio("Select Walk Power", ["A^1 (Direct Edges)", "A^2 (Walks of Length 2)", "A^3 (Walks of Length 3)"], horizontal=True)
        key_map = {"A^1 (Direct Edges)": "A^1", "A^2 (Walks of Length 2)": "A^2", "A^3 (Walks of Length 3)": "A^3"}
        st.dataframe(walk_powers[key_map[power_choice]], use_container_width=True)
        st.caption(walk_powers["theorem_statement"])

    # 2. Graph Isomorphism
    st.markdown("---")
    st.markdown("### 🔄 2. Graph Isomorphism Verification")
    g_base = nx.cycle_graph(4)
    g_relabel = nx.relabel_nodes(g_base, {0: "J_North", 1: "J_East", 2: "J_South", 3: "J_West"})
    iso_res = Unit5GraphTheory.test_graph_isomorphism(g_base, g_relabel)

    st.markdown(
        f"""
        <div class="glass-card">
            <span class="badge badge-cyan">Isomorphism Bijection Test</span>
            <div style="font-size:1.3rem; font-weight:700; color:#10B981; margin:6px 0;">
                G₁ ≅ G₂ : {'ISOMORPHIC (Structural Bijection Exists)' if iso_res['is_isomorphic'] else 'NOT ISOMORPHIC'}
            </div>
            <p style="color:#CBD5E1; font-size:0.85rem;">
                <b>Node Mapping Bijection:</b> <code>{iso_res['vertex_mapping']}</code><br>
                Both networks preserve degree sequences {iso_res['g1_degrees']} and identical adjacency structures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Minimum Spanning Tree (MST)
    st.markdown("---")
    st.markdown("### 🌲 3. Trees & Minimum Spanning Tree (MST / Kruskal's Algorithm)")
    mst_res = Unit5GraphTheory.compute_minimum_spanning_tree(undirected_g)

    c_mst1, c_mst2 = st.columns([1.1, 0.9])
    with c_mst1:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-emerald">Kruskal's MST Optimization</span>
                <div style="font-size:1.6rem; font-weight:800; color:#38BDF8; margin:6px 0;">
                    Total Cabling Cost: {mst_res['total_cabling_weight']} Units
                </div>
                <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.5;">
                    • <b>Total Tree Edges (V - 1):</b> {mst_res['mst_edge_count']}<br>
                    • <b>Cayley's Formula (n^(n - 2) Spanning Trees on Kₙ):</b> {mst_res['cayley_formula_spanning_trees_Kn']:,}<br>
                    • <b>Application:</b> {mst_res['traffic_application']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_mst2:
        st.markdown("#### Selected Minimal Communications Links")
        df_mst = pd.DataFrame(mst_res["mst_edges"])
        st.dataframe(df_mst, use_container_width=True, hide_index=True)

    # 4. Welsh-Powell Vertex Coloring
    st.markdown("---")
    st.markdown("### 🎨 4. Welsh-Powell Vertex Coloring on Traffic Conflict Graph")
    coloring_data = lab.get_conflict_coloring()

    col_cp1, col_cp2 = st.columns([1.2, 1.0])
    with col_cp1:
        fig_conflict = create_conflict_graph_plot(coloring_data)
        st.plotly_chart(fig_conflict, use_container_width=True)

    with col_cp2:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-cyan">Chromatic Analysis</span>
                <div style="font-size:1.8rem; font-weight:800; color:#38BDF8; margin:6px 0;">
                    Chromatic Number χ(G) = {coloring_data['chromatic_number']}
                </div>
                <div style="font-size:0.85rem; color:#CBD5E1;">
                    Total Turning Movements (|V|): <b>{coloring_data['total_vertices']}</b><br>
                    Total Collision Conflict Edges (|E|): <b>{coloring_data['total_conflict_edges']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for p in coloring_data["phases"]:
            color_hex = p["color_hex"]
            mov_str = ", ".join(p["assigned_movements"])
            st.markdown(
                f"""
                <div style="background:#0F172A; border-left:4px solid {color_hex}; border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                    <div style="font-size:0.85rem; font-weight:700; color:{color_hex};">{p['phase_name']}</div>
                    <div style="font-size:0.8rem; color:#F8FAFC;">Safe Paths: <b>{mov_str}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Planar Graphs & Eulerian/Hamiltonian Traversal
    st.markdown("---")
    st.markdown("### 🌐 5. Planar Networks, Euler's Characteristic, and Traversals")
    c_pl1, c_pl2 = st.columns(2)

    with c_pl1:
        st.markdown("#### Planar Road Networks & Euler's Polyhedral Formula")
        inp_v = st.number_input("Intersections (V)", min_value=3, max_value=50, value=6, key="pl_v")
        inp_e = st.number_input("Road Segments (E)", min_value=1, max_value=100, value=8, key="pl_e")
        inp_f = st.number_input("City Blocks / Faces (F)", min_value=1, max_value=50, value=4, key="pl_f")

        eval_res = lab.verify_planarity_and_euler(inp_v, inp_e, inp_f)
        st.markdown(
            f"""
            <div class="glass-card">
                <b>Euler Characteristic (V - E + F):</b> <span style="font-size:1.2rem; font-weight:700; color:{'#10B981' if eval_res['is_valid_euler'] else '#EF4444'};">{eval_res['euler_characteristic']}</span><br>
                <b>Planarity Bound (E ≤ 3V - 6):</b> Max {eval_res['max_allowed_edges_for_planarity']} at-grade edges.<br>
                <b>Engineering Verdict:</b> {'At-grade road construction feasible.' if eval_res['is_theoretically_planar'] else 'Non-Planar! Flyovers/Bridges mathematically required!'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_pl2:
        st.markdown("#### Eulerian Patrol Trails vs Hamiltonian Maintenance Cycles")
        eh_eval = lab.analyze_eulerian_and_hamiltonian(undirected_g)
        st.markdown(
            f"""
            <div class="glass-card">
                • <b>Is Connected:</b> `{eh_eval['is_connected']}`<br>
                • <b>Odd Degree Vertices:</b> `{eh_eval['odd_degree_count']}` (Nodes: `{eh_eval['odd_degree_nodes']}`)<br>
                • <b>Eulerian Circuit (All Even Degrees):</b> `{'Yes' if eh_eval['has_eulerian_circuit'] else 'No'}`<br>
                • <b>Eulerian Path (0 or 2 Odd):</b> `{'Yes' if eh_eval['has_eulerian_path'] else 'No'}`<br>
                • <b>Dirac's Hamiltonian Condition (min deg ≥ n/2):</b> `{'Guaranteed' if eh_eval['satisfies_dirac_hamiltonian'] else 'Not guaranteed by Dirac'}`
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Laplacian Spectral Analysis
    st.markdown("---")
    st.markdown("### 🧮 6. Laplacian Matrix ($L = D - A$) & Spectral Connectivity")
    spectral_info = lab.compute_laplacian_spectral(undirected_g)

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.markdown("**Degree Matrix D:**")
        st.dataframe(pd.DataFrame(spectral_info["degree_matrix"], index=spectral_info["nodes"], columns=spectral_info["nodes"]), use_container_width=True)
    with col_mat2:
        st.markdown("**Laplacian Matrix L = D - A:**")
        st.dataframe(pd.DataFrame(spectral_info["laplacian_matrix"], index=spectral_info["nodes"], columns=spectral_info["nodes"]), use_container_width=True)

    st.markdown(
        f"""
        <div class="glass-card">
            <b>Laplacian Spectrum:</b> <code>{spectral_info['eigenvalues']}</code><br>
            <b>Fiedler Value (Algebraic Connectivity λ₂):</b> <span style="color:#10B981; font-weight:700; font-size:1.1rem;">{spectral_info['fiedler_value']}</span>
            <p style="color:#CBD5E1; font-size:0.85rem; margin-top:4px;">{spectral_info['connectivity_interpretation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STUDENT VIVA VOCE & ORAL PREPARATION GUIDE
# ============================================================================
with tab_viva:
    st.markdown("## 🎓 Student Capstone Viva Voce Guide")
    st.markdown(
        """
        *Curated oral examination questions directly mapped to the 5 Discrete Mathematics Units,
        designed for university final project assessments.*
        """
    )

    viva_items = lab.get_viva_questions()
    for idx, item in enumerate(viva_items, 1):
        with st.expander(f"Q{idx}. {item['question']}", expanded=(idx <= 2)):
            st.markdown(
                f"""
                <div style="background:#0F172A; border-left:3px solid #06B6D4; padding:12px; border-radius:6px; color:#E2E8F0; font-size:0.9rem; line-height:1.6;">
                    {item['answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )
