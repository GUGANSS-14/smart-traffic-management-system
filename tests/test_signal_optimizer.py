"""
Unit Tests for Signal Optimizer Module.
Validates:
- Empty traffic handling (fairness baseline)
- Equal lane demand (symmetric allocation)
- Heavily unequal demand (starvation prevention & clamping)
- Timing-boundary cases (min green, max green, cycle budget invariants, non-negativity)
- Plan comparison and divergence explanations
- Student simulation labeling
"""

import unittest
from modules.traffic_analyzer import LaneMetrics
from modules.signal_optimizer import SignalOptimizer, SignalPlan, SignalComparison, SIMULATION_DISCLAIMER


class TestSignalOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = SignalOptimizer(
            min_green=10,
            max_green=60,
            yellow_time=3,
            all_red=2,
            default_cycle=90,
        )

    def test_static_plan(self):
        """Verifies Webster 50/50 baseline split and clearance accounting."""
        plan = self.opt.compute_static_plan()
        self.assertEqual(plan.cycle_time_seconds, 90)
        self.assertEqual(len(plan.phases), 2)
        # Lost clearance = (3 + 2) * 2 = 10s => available green = 80s
        self.assertEqual(plan.total_lost_time, 10)
        self.assertEqual(plan.total_green_time, 80)
        self.assertEqual(plan.phases[0].green_time_seconds, 40)
        self.assertEqual(plan.phases[1].green_time_seconds, 40)
        self.assertIn("STUDENT SIMULATION", plan.simulation_disclaimer)

    def test_empty_traffic(self):
        """Feature Test: Empty traffic (0 vehicles / 0 PCU)."""
        empty_metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=0.0, total_vehicles=0),
            "South": LaneMetrics(lane_id="South", pcu_total=0.0, total_vehicles=0),
            "East": LaneMetrics(lane_id="East", pcu_total=0.0, total_vehicles=0),
            "West": LaneMetrics(lane_id="West", pcu_total=0.0, total_vehicles=0),
        }
        plan = self.opt.compute_adaptive_plan(empty_metrics)

        # Must satisfy non-negativity and min_green floor
        self.assertGreaterEqual(plan.phases[0].green_time_seconds, self.opt.min_green)
        self.assertGreaterEqual(plan.phases[1].green_time_seconds, self.opt.min_green)
        # Equal split for symmetric zero demand
        self.assertEqual(plan.phases[0].green_time_seconds, plan.phases[1].green_time_seconds)
        # Cycle budget invariant
        total_cycle = plan.total_green_time + plan.total_lost_time
        self.assertEqual(total_cycle, self.opt.default_cycle)

    def test_equal_lane_demand(self):
        """Feature Test: Equal lane demand on both corridors."""
        equal_metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=25.0, total_vehicles=20),
            "South": LaneMetrics(lane_id="South", pcu_total=25.0, total_vehicles=20),
            "East": LaneMetrics(lane_id="East", pcu_total=25.0, total_vehicles=20),
            "West": LaneMetrics(lane_id="West", pcu_total=25.0, total_vehicles=20),
        }
        comparison = self.opt.recommend_and_compare(equal_metrics)
        rec_plan = comparison.recommended_plan

        # Equal demand gives symmetric green durations
        self.assertEqual(rec_plan.phases[0].green_time_seconds, rec_plan.phases[1].green_time_seconds)
        self.assertEqual(rec_plan.phases[0].green_time_seconds, 40)
        self.assertEqual(rec_plan.phases[1].green_time_seconds, 40)
        # Comparison divergence notes symmetry
        self.assertIn("symmetric", comparison.divergence_explanation.lower())
        self.assertEqual(comparison.phase_deltas["Phase 1 (North-South)"], 0)
        self.assertEqual(comparison.phase_deltas["Phase 2 (East-West)"], 0)
        self.assertTrue(comparison.cycle_budget_maintained)

    def test_heavily_unequal_demand(self):
        """
        Feature Test: Heavily unequal demand.
        Tests that dominant corridor receives higher green (capped at max_green)
        and deserted corridor receives guaranteed min_green without starvation.
        """
        skewed_metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=80.0, total_vehicles=70),
            "South": LaneMetrics(lane_id="South", pcu_total=60.0, total_vehicles=55),
            "East": LaneMetrics(lane_id="East", pcu_total=0.0, total_vehicles=0),
            "West": LaneMetrics(lane_id="West", pcu_total=0.0, total_vehicles=0),
        }
        comparison = self.opt.recommend_and_compare(skewed_metrics)
        rec = comparison.recommended_plan

        # Phase 1 (NS) must receive more green than Phase 2 (EW)
        self.assertGreater(rec.phases[0].green_time_seconds, rec.phases[1].green_time_seconds)

        # Clamping check: Phase 1 green <= max_green (60s)
        self.assertLessEqual(rec.phases[0].green_time_seconds, self.opt.max_green)

        # Fairness safeguard check: Phase 2 (EW with 0 PCU) must receive at least min_green (10s)
        self.assertGreaterEqual(rec.phases[1].green_time_seconds, self.opt.min_green)

        # Cycle budget exact invariant
        self.assertEqual(rec.total_green_time + rec.total_lost_time, self.opt.default_cycle)

        # Verification of fairness notice and divergence explanation
        self.assertTrue(comparison.fairness_safeguard_active)
        self.assertIn("Fairness Safeguard Active", comparison.fairness_notice)
        self.assertIn("starvation", comparison.divergence_explanation.lower())

    def test_timing_boundary_cases_min_max_green(self):
        """Boundary Test: Strict min_green and max_green enforcement across extreme inputs."""
        # Extreme 1: Extremely high demand on EW corridor (10,000 PCU)
        extreme_metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=0.0),
            "South": LaneMetrics(lane_id="South", pcu_total=0.0),
            "East": LaneMetrics(lane_id="East", pcu_total=5000.0),
            "West": LaneMetrics(lane_id="West", pcu_total=5000.0),
        }
        plan = self.opt.compute_adaptive_plan(extreme_metrics)

        # EW capped at max_green
        self.assertLessEqual(plan.phases[1].green_time_seconds, self.opt.max_green)
        # NS floor protected at min_green
        self.assertGreaterEqual(plan.phases[0].green_time_seconds, self.opt.min_green)
        # Sum of greens matches available green
        self.assertEqual(plan.total_green_time, 80)

    def test_timing_boundary_cases_cycle_budget_invariance(self):
        """Boundary Test: Fixed cycle budget invariant for multiple configurations."""
        metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=30.0),
            "South": LaneMetrics(lane_id="South", pcu_total=10.0),
            "East": LaneMetrics(lane_id="East", pcu_total=5.0),
            "West": LaneMetrics(lane_id="West", pcu_total=15.0),
        }

        # Test various cycle budgets: 50s, 60s, 90s, 120s, 150s
        for c_budget in [50, 60, 90, 120, 150]:
            opt = SignalOptimizer(min_green=10, max_green=70, yellow_time=3, all_red=2, default_cycle=c_budget)
            plan = opt.compute_adaptive_plan(metrics, cycle_time=c_budget)
            total_time = plan.total_green_time + plan.total_lost_time
            self.assertEqual(total_time, c_budget, f"Failed cycle budget invariance for cycle={c_budget}")
            self.assertEqual(plan.cycle_time_seconds, c_budget)

    def test_timing_boundary_cases_non_negative_splits(self):
        """Boundary Test: Ensures all green times are strictly non-negative integers."""
        zero_metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=-5.0),  # Negative input resilience
            "South": LaneMetrics(lane_id="South", pcu_total=0.0),
            "East": LaneMetrics(lane_id="East", pcu_total=0.0),
            "West": LaneMetrics(lane_id="West", pcu_total=-10.0),
        }
        plan = self.opt.compute_adaptive_plan(zero_metrics)
        for phase in plan.phases:
            self.assertGreaterEqual(phase.green_time_seconds, 0)
            self.assertGreaterEqual(phase.yellow_time_seconds, 0)
            self.assertGreaterEqual(phase.red_time_seconds, 0)

    def test_emergency_preemption(self):
        """Verifies emergency override forces maximum safe green on incident corridor."""
        metrics = {
            "North": LaneMetrics(lane_id="North", pcu_total=10.0),
            "South": LaneMetrics(lane_id="South", pcu_total=10.0),
            "East": LaneMetrics(lane_id="East", pcu_total=10.0),
            "West": LaneMetrics(lane_id="West", pcu_total=10.0),
        }
        plan = self.opt.compute_adaptive_plan(metrics, emergency_lane="North")
        self.assertEqual(plan.phases[0].priority_level, "Emergency Override")
        self.assertEqual(plan.phases[0].green_time_seconds, self.opt.max_green)
        # Cross phase retains min_green fairness floor
        self.assertGreaterEqual(plan.phases[1].green_time_seconds, self.opt.min_green)

    def test_student_simulation_disclaimer(self):
        """Verifies explicit student simulation label is present on all plans and comparisons."""
        plan = self.opt.compute_static_plan()
        self.assertTrue(plan.is_simulation_recommendation)
        self.assertEqual(plan.simulation_disclaimer, SIMULATION_DISCLAIMER)

        metrics = {"North": LaneMetrics("North", 10.0)}
        comp = self.opt.recommend_and_compare(metrics)
        self.assertEqual(comp.simulation_disclaimer, SIMULATION_DISCLAIMER)

    def test_various_cycle_budget_invariants(self):
        """Verifies cycle budget preservation across various total cycle durations."""
        for cycle_len in [60, 90, 120, 150]:
            custom_opt = SignalOptimizer(min_green=10, max_green=70, yellow_time=3, all_red=2, default_cycle=cycle_len)
            metrics = {
                "North": LaneMetrics("North", pcu_total=40.0),
                "South": LaneMetrics("South", pcu_total=30.0),
                "East": LaneMetrics("East", pcu_total=20.0),
                "West": LaneMetrics("West", pcu_total=10.0),
            }
            plan = custom_opt.compute_adaptive_plan(metrics)
            total = plan.total_green_time + plan.total_lost_time
            self.assertEqual(total, cycle_len)

    def test_starvation_prevention_on_low_demand(self):
        """Verifies minor corridor with 1 PCU vs 150 PCU still receives min_green floor."""
        metrics = {
            "North": LaneMetrics("North", pcu_total=150.0),
            "South": LaneMetrics("South", pcu_total=100.0),
            "East": LaneMetrics("East", pcu_total=1.0),
            "West": LaneMetrics("West", pcu_total=0.5),
        }
        plan = self.opt.compute_adaptive_plan(metrics)
        phase2 = plan.phases[1]
        self.assertGreaterEqual(phase2.green_time_seconds, self.opt.min_green)


if __name__ == "__main__":
    unittest.main()

