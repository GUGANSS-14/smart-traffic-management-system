"""
Unit Tests for Traffic Analyzer Module.
Validates:
- Empty frame detections (zero initializations)
- PCU conversion weighting
- Documented density proxy calculation (vehicles / normalized area)
- Spatial occupancy ratio calculation
- Flow rate estimation from line crossings per minute
- Configurable 3-level traffic classification (Low / Moderate / High)
- Vehicle count aggregation by time interval
- Metric assumptions & limitations documentation
- KPI summary calculations
"""

import unittest
import pandas as pd
from modules.vehicle_detector import Detection
from modules.traffic_analyzer import TrafficAnalyzer, LaneMetrics


class TestTrafficAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrafficAnalyzer(
            lane_capacity_pcu=30.0,
            low_threshold=35.0,
            high_threshold=70.0,
        )

    def test_empty_detections(self):
        """Validates behavior when no vehicles are detected."""
        metrics = self.analyzer.analyze_frame_detections([])
        self.assertEqual(len(metrics), 4)
        for lane in ["North", "South", "East", "West"]:
            self.assertIn(lane, metrics)
            self.assertEqual(metrics[lane].total_vehicles, 0)
            self.assertEqual(metrics[lane].pcu_total, 0.0)
            self.assertEqual(metrics[lane].density_proxy, 0.0)
            self.assertEqual(metrics[lane].flow_rate_per_min, 0.0)
            self.assertEqual(metrics[lane].traffic_state, "Low")
            self.assertEqual(metrics[lane].congestion_level, "Free Flow")

    def test_pcu_weighting(self):
        """Tests Passenger Car Unit weighting: Car=1.0, Bus=3.0, Truck=3.0, Motorcycle=0.5."""
        dets = [
            Detection(bbox=(0, 0, 10, 10), class_name="car", confidence=0.9, center=(5, 5), lane_id="North"),
            Detection(bbox=(0, 0, 10, 10), class_name="bus", confidence=0.85, center=(5, 5), lane_id="North"),
            Detection(bbox=(0, 0, 10, 10), class_name="truck", confidence=0.88, center=(5, 5), lane_id="North"),
            Detection(bbox=(0, 0, 10, 10), class_name="motorcycle", confidence=0.95, center=(5, 5), lane_id="North"),
        ]
        metrics = self.analyzer.analyze_frame_detections(dets)
        north = metrics["North"]

        self.assertEqual(north.cars, 1)
        self.assertEqual(north.buses, 1)
        self.assertEqual(north.trucks, 1)
        self.assertEqual(north.motorcycles, 1)
        self.assertEqual(north.total_vehicles, 4)
        # Expected PCU: 1.0 + 3.0 + 3.0 + 0.5 = 7.5
        self.assertAlmostEqual(north.pcu_total, 7.5, places=1)

    def test_density_proxy_calculation(self):
        """Feature Test: Documented density proxy = vehicle_count / normalized_area."""
        # Full frame (normalized area = 1.0)
        dp_full = TrafficAnalyzer.calculate_density_proxy(vehicle_count=10, normalized_region_area=1.0)
        self.assertEqual(dp_full, 10.0)

        # Region of Interest half-frame (normalized area = 0.5) -> density proxy doubles
        dp_half = TrafficAnalyzer.calculate_density_proxy(vehicle_count=10, normalized_region_area=0.5)
        self.assertEqual(dp_half, 20.0)

        # Degenerate zero/negative area fallback to 1.0
        dp_fallback = TrafficAnalyzer.calculate_density_proxy(vehicle_count=8, normalized_region_area=0.0)
        self.assertEqual(dp_fallback, 8.0)

    def test_occupancy_ratio_calculation(self):
        """Feature Test: Documented spatial occupancy ratio clamped to [0.0, 1.0]."""
        # 2 boxes of 100x100 = 20,000 px in an ROI of 40,000 px -> ratio = 0.5
        ratio = TrafficAnalyzer.calculate_occupancy_ratio(
            bbox_areas=[10000.0, 10000.0],
            roi_area_px=40000.0,
        )
        self.assertAlmostEqual(ratio, 0.5, places=3)

        # Empty bboxes
        self.assertEqual(TrafficAnalyzer.calculate_occupancy_ratio([], 40000.0), 0.0)

        # Overflow capping at 1.0
        self.assertEqual(TrafficAnalyzer.calculate_occupancy_ratio([50000.0], 40000.0), 1.0)

    def test_flow_rate_estimation(self):
        """Feature Test: Estimate flow rate (vehicles per minute) from line crossings."""
        # 15 vehicles crossed line in 60 seconds -> 15.0 veh/min
        flow = TrafficAnalyzer.estimate_traffic_flow(crossing_count=15, duration_seconds=60.0)
        self.assertEqual(flow, 15.0)

        # 30 vehicles in 120 seconds -> 15.0 veh/min
        flow_2 = TrafficAnalyzer.estimate_traffic_flow(crossing_count=30, duration_seconds=120.0)
        self.assertEqual(flow_2, 15.0)

        # 0 seconds duration edge case
        self.assertEqual(TrafficAnalyzer.estimate_traffic_flow(crossing_count=10, duration_seconds=0.0), 0.0)

    def test_traffic_classification_configurable_thresholds(self):
        """Feature Test: Classify traffic as Low, Moderate, or High using configurable thresholds."""
        custom_analyzer = TrafficAnalyzer(
            lane_capacity_pcu=30.0,
            low_threshold=25.0,
            high_threshold=65.0,
        )

        # Below low threshold -> Low
        self.assertEqual(custom_analyzer.classify_traffic(15.0), "Low")
        # At / Above low threshold, below high threshold -> Moderate
        self.assertEqual(custom_analyzer.classify_traffic(25.0), "Moderate")
        self.assertEqual(custom_analyzer.classify_traffic(50.0), "Moderate")
        # At / Above high threshold -> High
        self.assertEqual(custom_analyzer.classify_traffic(65.0), "High")
        self.assertEqual(custom_analyzer.classify_traffic(90.0), "High")

        # Explicit parameter override test
        self.assertEqual(custom_analyzer.classify_traffic(30.0, low_thresh=40.0), "Low")
        self.assertEqual(custom_analyzer.classify_traffic(40.0, high_thresh=35.0), "High")

    def test_interval_aggregation(self):
        """Feature Test: Vehicle counts aggregated by class, lane, and interval."""
        records = [
            {"timestamp": "2026-10-10 08:00:00", "approach_lane": "North", "cars": 10, "buses": 1, "trucks": 1, "motorcycles": 5, "total_vehicles": 17, "pcu_total": 18.5},
            {"timestamp": "2026-10-10 08:05:00", "approach_lane": "North", "cars": 12, "buses": 2, "trucks": 0, "motorcycles": 4, "total_vehicles": 18, "pcu_total": 20.0},
            {"timestamp": "2026-10-10 08:20:00", "approach_lane": "North", "cars": 20, "buses": 3, "trucks": 2, "motorcycles": 8, "total_vehicles": 33, "pcu_total": 39.0},
        ]
        df = pd.DataFrame(records)
        df_15m = TrafficAnalyzer.calculate_vehicle_counts_by_interval(df, interval="15min")

        self.assertFalse(df_15m.empty)
        # First 15min bucket (08:00:00) should sum cars = 10 + 12 = 22
        first_row = df_15m.iloc[0]
        self.assertEqual(first_row["cars"], 22)
        self.assertEqual(first_row["total_vehicles"], 35)

    def test_metric_assumptions_documentation(self):
        """Feature Test: Explains assumptions and limitations for each metric."""
        docs = TrafficAnalyzer.explain_metric_assumptions()
        expected_keys = [
            "vehicle_counts",
            "density_proxy",
            "occupancy_ratio",
            "traffic_flow",
            "pcu_conversion",
            "traffic_classification",
        ]
        for k in expected_keys:
            self.assertIn(k, docs)
            self.assertIn("name", docs[k])
            self.assertIn("formula", docs[k])
            self.assertIn("assumptions", docs[k])
            self.assertIn("limitations", docs[k])
            self.assertGreater(len(docs[k]["assumptions"]), 10)
            self.assertGreater(len(docs[k]["limitations"]), 10)

    def test_kpi_summary(self):
        """Verifies calculation of KPI aggregates."""
        sample_df = pd.DataFrame([
            {
                "timestamp": "2026-10-10 08:00:00",
                "total_vehicles": 100,
                "pcu_total": 95.0,
                "delay_seconds_static": 50.0,
                "delay_seconds_adaptive": 25.0,
                "density_proxy": 12.5,
            }
        ])
        kpis = TrafficAnalyzer.compute_kpi_summary(sample_df)
        self.assertEqual(kpis["total_vehicles_processed"], 100)
        self.assertEqual(kpis["delay_saved_pct"], 50.0)
        self.assertGreater(kpis["co2_saved_kg"], 0.0)
        self.assertEqual(kpis["avg_density_proxy"], 12.5)

    def test_empty_dataframe_kpi_summary(self):
        """Verifies graceful handling of empty DataFrame for KPI summaries."""
        empty_df = pd.DataFrame()
        kpis = TrafficAnalyzer.compute_kpi_summary(empty_df)
        self.assertEqual(kpis["total_vehicles_processed"], 0)
        self.assertEqual(kpis["delay_saved_pct"], 0.0)
        self.assertEqual(kpis["co2_saved_kg"], 0.0)

    def test_traffic_classification_thresholds(self):
        """Verifies 3-level traffic classification thresholds (Low, Moderate, High)."""
        # Under low threshold (< 35%)
        self.assertEqual(self.analyzer.classify_traffic(20.0), "Low")
        # Between low and high threshold (35% to 70%)
        self.assertEqual(self.analyzer.classify_traffic(50.0), "Moderate")
        # Above high threshold (>= 70%)
        self.assertEqual(self.analyzer.classify_traffic(85.0), "High")


if __name__ == "__main__":
    unittest.main()

