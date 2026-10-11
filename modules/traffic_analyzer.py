"""
Traffic Analyzer Module
Computes lane-level vehicle counts, Passenger Car Units (PCU),
spatial density proxy, spatial occupancy ratio, flow rate estimation,
configurable traffic classification (Low / Moderate / High), and KPI benchmarking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .vehicle_detector import Detection


@dataclass
class LaneMetrics:
    """Metrics aggregated for a single intersection approach lane."""
    lane_id: str
    cars: int = 0
    buses: int = 0
    trucks: int = 0
    motorcycles: int = 0
    total_vehicles: int = 0
    pcu_total: float = 0.0
    density_proxy: float = 0.0         # Vehicles per normalized unit area
    occupancy_ratio: float = 0.0       # Spatial bbox occupancy [0.0 - 1.0]
    flow_rate_per_min: float = 0.0     # Vehicles crossing line per minute
    traffic_state: str = "Low"         # Configurable 3-level: "Low", "Moderate", "High"
    congestion_index: float = 0.0      # 0 to 100% capacity utilization
    congestion_level: str = "Free Flow"
    queue_length_m: float = 0.0
    avg_speed_kmh: float = 45.0
    capacity_pcu: float = 30.0        # Max capacity threshold for approach lane
    class_counts: Dict[str, int] = field(default_factory=dict)


class TrafficAnalyzer:
    """
    Transparent, rule-based traffic analytics engine.
    Analyzes detected vehicles, estimates flow rates, evaluates spatial density proxies,
    classifies congestion states, and documents engineering assumptions.
    """
    LANE_NAMES = ["North", "South", "East", "West"]

    PCU_WEIGHTS = {
        "car": 1.0,
        "bus": 3.0,
        "truck": 3.0,
        "motorcycle": 0.5,
    }

    VEHICLE_LENGTHS_M = {
        "car": 4.5,
        "bus": 12.0,
        "truck": 10.0,
        "motorcycle": 2.0,
    }

    def __init__(
        self,
        lane_capacity_pcu: float = 25.0,
        low_threshold: float = 35.0,
        high_threshold: float = 70.0,
    ):
        """
        Initializes the traffic analyzer.

        Args:
            lane_capacity_pcu: Baseline capacity in Passenger Car Units per approach lane.
            low_threshold: Cutoff percentage threshold between Low and Moderate traffic.
            high_threshold: Cutoff percentage threshold between Moderate and High traffic.
        """
        self.lane_capacity_pcu = max(1.0, float(lane_capacity_pcu))
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.history: List[Dict[str, Any]] = []

    def classify_traffic(
        self,
        metric_value: float,
        low_thresh: Optional[float] = None,
        high_thresh: Optional[float] = None,
    ) -> str:
        """
        Classifies traffic intensity into 'Low', 'Moderate', or 'High'
        using configurable thresholds.

        Args:
            metric_value: Congestion index (0-100), density proxy, or PCU value.
            low_thresh: Custom low threshold override.
            high_thresh: Custom high threshold override.

        Returns:
            'Low', 'Moderate', or 'High'
        """
        low = self.low_threshold if low_thresh is None else float(low_thresh)
        high = self.high_threshold if high_thresh is None else float(high_thresh)
        val = float(metric_value)

        if val < low:
            return "Low"
        elif val < high:
            return "Moderate"
        else:
            return "High"

    @staticmethod
    def calculate_density_proxy(
        vehicle_count: float,
        normalized_region_area: float = 1.0,
    ) -> float:
        """
        Calculates a documented spatial density proxy.

        Formula:
            density_proxy = vehicle_count / max(0.001, normalized_region_area)

        Definition & Assumptions:
            True physical traffic density (vehicles/km/lane) requires extrinsic camera
            calibration and bird's-eye inverse perspective mapping.
            The density proxy represents vehicle concentration per normalized unit area
            within an observation Region of Interest (ROI) (normalized_area in (0, 1]).
            For full-frame analysis, normalized_region_area is 1.0.

        Args:
            vehicle_count: Number of vehicles (or PCU) detected within the region.
            normalized_region_area: Region area normalized relative to total camera view (0 to 1].

        Returns:
            Vehicles per normalized area unit, rounded to 2 decimal places.
        """
        if normalized_region_area <= 0.0:
            normalized_region_area = 1.0
        return round(float(vehicle_count) / max(0.001, normalized_region_area), 2)

    @staticmethod
    def calculate_occupancy_ratio(
        bbox_areas: List[float],
        roi_area_px: float,
    ) -> float:
        """
        Calculates the spatial occupancy ratio of a defined Region of Interest (ROI).

        Formula:
            occupancy_ratio = sum(bbox_areas) / roi_area_px

        Args:
            bbox_areas: List of pixel areas for vehicle bounding boxes inside the ROI.
            roi_area_px: Total area in pixels of the defined ROI (> 0).

        Returns:
            Occupancy fraction clamped to [0.0, 1.0], rounded to 4 decimal places.
        """
        if roi_area_px <= 0.0 or not bbox_areas:
            return 0.0
        total_veh_area = sum(max(0.0, float(a)) for a in bbox_areas)
        ratio = total_veh_area / roi_area_px
        return round(min(1.0, max(0.0, ratio)), 4)

    @staticmethod
    def estimate_traffic_flow(
        crossing_count: int,
        duration_seconds: float,
    ) -> float:
        """
        Estimates traffic flow (vehicles per minute) from tracked vehicles crossing a counting line.

        Formula:
            flow_per_min = (crossing_count / duration_seconds) * 60.0

        Args:
            crossing_count: Number of unique tracked vehicles that traversed the counting line.
            duration_seconds: Elapsed observation duration in seconds (> 0).

        Returns:
            Flow rate in vehicles per minute, rounded to 2 decimal places.
        """
        if duration_seconds <= 0.0:
            return 0.0
        return round((max(0, crossing_count) / duration_seconds) * 60.0, 2)

    @staticmethod
    def calculate_vehicle_counts_by_interval(
        df: pd.DataFrame,
        interval: str = "15min",
        time_column: str = "timestamp",
        lane_column: str = "approach_lane",
    ) -> pd.DataFrame:
        """
        Aggregates vehicle counts by class, approach lane, and time interval.

        Args:
            df: DataFrame containing traffic records with timestamps and vehicle class columns.
            interval: Resampling interval string (e.g. '5min', '15min', '1h').
            time_column: Column name for timestamps.
            lane_column: Column name for approach lanes.

        Returns:
            Aggregated DataFrame with resampled sums.
        """
        if df.empty or time_column not in df.columns:
            return pd.DataFrame()

        work_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(work_df[time_column]):
            work_df[time_column] = pd.to_datetime(work_df[time_column])

        metric_cols = [
            c for c in ["cars", "buses", "trucks", "motorcycles", "total_vehicles", "pcu_total"]
            if c in work_df.columns
        ]

        if not metric_cols:
            return pd.DataFrame()

        if lane_column in work_df.columns:
            grouped = (
                work_df.set_index(time_column)
                .groupby([pd.Grouper(freq=interval), lane_column])[metric_cols]
                .sum()
                .reset_index()
            )
        else:
            grouped = (
                work_df.set_index(time_column)
                .resample(interval)[metric_cols]
                .sum()
                .reset_index()
            )
        return grouped

    @staticmethod
    def explain_metric_assumptions() -> Dict[str, Dict[str, str]]:
        """
        Returns structured documentation detailing the mathematical formulation,
        civil engineering assumptions, and practical limitations of each traffic metric.
        """
        return {
            "vehicle_counts": {
                "name": "Vehicle Counts by Class & Lane",
                "formula": "N_class = count(detections where class == c and lane == l)",
                "assumptions": (
                    "Assumes the object detection model reliably identifies road vehicles "
                    "in the camera view without excessive false positives or missed detections."
                ),
                "limitations": (
                    "Dense bumper-to-bumper traffic causes occlusion where two-wheelers and compact "
                    "cars behind large trucks/buses are obscured. Night glare, shadows, and rain also "
                    "temporarily degrade inference confidence."
                ),
            },
            "density_proxy": {
                "name": "Spatial Density Proxy (rho_proxy)",
                "formula": "rho_proxy = N_vehicles / Area_normalized",
                "assumptions": (
                    "Assumes vehicular concentration inside a 2D camera Region of Interest (ROI) "
                    "serves as a linear proxy for physical roadway density."
                ),
                "limitations": (
                    "Camera perspective compresses the distance in the upper visual field (perspective foreshortening). "
                    "Without 3D inverse homography calibration, distant vehicles occupy far fewer pixels "
                    "than foreground vehicles."
                ),
            },
            "occupancy_ratio": {
                "name": "Spatial Occupancy Ratio",
                "formula": "Occupancy = sum(BoundingBox_Area_i) / ROI_Area",
                "assumptions": (
                    "Assumes bounding box pixel area reflects physical road occupancy. "
                    "Emulates the operational principle of physical inductive loop detectors."
                ),
                "limitations": (
                    "Rectangular bounding boxes enclose empty roadway around angled vehicles. "
                    "Elevated camera viewing angles artificially exaggerate the visual area of high-profile "
                    "transit buses and commercial trucks."
                ),
            },
            "traffic_flow": {
                "name": "Estimated Traffic Flow Rate (q)",
                "formula": "q = (N_crossed / Delta_t_minutes)",
                "assumptions": (
                    "Assumes tracked vehicle centroid trajectories cross the counting line exactly once "
                    "with stable tracking IDs."
                ),
                "limitations": (
                    "Short observation windows (< 5 minutes) exhibit high Poisson arrival variance. "
                    "Vehicles stopped on top of the counting line during red signals can cause tracking "
                    "jitter and double counting."
                ),
            },
            "pcu_conversion": {
                "name": "Passenger Car Units (PCU / PCE)",
                "formula": "PCU = 1.0 * Cars + 3.0 * Buses + 3.0 * Trucks + 0.5 * Motorcycles",
                "assumptions": (
                    "Based on Indian Highway Capacity Manual (Indo-HCM) / US HCM standards where heavy vehicles "
                    "consume greater headway space and exhibit slower acceleration compared to standard passenger cars."
                ),
                "limitations": (
                    "PCU factors assume uniform flat terrain. On steep grades, truck PCUs can exceed 4.5. "
                    "Filtering two-wheelers weaving between stationary queues violate strict lane-based flow rules."
                ),
            },
            "traffic_classification": {
                "name": "Traffic State Classification (Low / Moderate / High)",
                "formula": "State = 'Low' if Index < Thresh_Low else ('Moderate' if Index < Thresh_High else 'High')",
                "assumptions": (
                    "Assumes predetermined threshold cutoffs divide Level of Service (LOS) bands "
                    "proportionally to approach lane design capacity."
                ),
                "limitations": (
                    "Fixed cutoffs enforce discrete boundaries whereas real traffic dynamics transition "
                    "continuously along the fundamental Greenshields speed-density curve."
                ),
            },
        }

    def analyze_frame_detections(
        self,
        detections: List[Detection],
        normalized_region_area: float = 1.0,
        elapsed_seconds: float = 60.0,
    ) -> Dict[str, LaneMetrics]:
        """
        Groups detections by approach lane and computes lane-level traffic metrics
        including counts, PCU, density proxy, queue length, and 3-level traffic classification.

        Args:
            detections: List of Detection instances from the vehicle detector.
            normalized_region_area: Area fraction of the observation region (0 < area <= 1.0).
            elapsed_seconds: Duration in seconds for flow rate calculation.

        Returns:
            Dictionary mapping approach lane names to LaneMetrics dataclasses.
        """
        metrics: Dict[str, LaneMetrics] = {
            lane: LaneMetrics(
                lane_id=lane,
                capacity_pcu=self.lane_capacity_pcu,
                class_counts={"car": 0, "bus": 0, "truck": 0, "motorcycle": 0},
            )
            for lane in self.LANE_NAMES
        }

        # Tabulate counts
        for det in detections:
            lane = det.lane_id if det.lane_id in metrics else "North"
            m = metrics[lane]
            cname = det.class_name.lower()

            if cname == "car":
                m.cars += 1
            elif cname == "bus":
                m.buses += 1
            elif cname == "truck":
                m.trucks += 1
            elif cname == "motorcycle":
                m.motorcycles += 1
            else:
                m.cars += 1

            m.class_counts[cname] = m.class_counts.get(cname, 0) + 1
            m.total_vehicles += 1
            m.pcu_total += self.PCU_WEIGHTS.get(cname, 1.0)
            m.queue_length_m += self.VEHICLE_LENGTHS_M.get(cname, 4.5) + 1.5  # +1.5m safety gap

        # Compute congestion indices, density proxy, and classifications
        for lane, m in metrics.items():
            if m.capacity_pcu > 0:
                raw_ci = (m.pcu_total / m.capacity_pcu) * 100.0
                m.congestion_index = round(min(100.0, raw_ci), 1)
            else:
                m.congestion_index = 0.0

            # Density proxy (vehicles per normalized unit area)
            m.density_proxy = self.calculate_density_proxy(m.total_vehicles, normalized_region_area)

            # Flow rate estimate from instantaneous frame vehicle count over elapsed seconds
            m.flow_rate_per_min = self.estimate_traffic_flow(m.total_vehicles, elapsed_seconds)

            # 3-level traffic state classification
            m.traffic_state = self.classify_traffic(m.congestion_index)

            # Detailed Level of Service (LOS) classification
            if m.congestion_index < self.low_threshold:
                m.congestion_level = "Free Flow"
                m.avg_speed_kmh = round(45.0 - (m.congestion_index * 0.15), 1)
            elif m.congestion_index < self.high_threshold:
                m.congestion_level = "Moderate"
                m.avg_speed_kmh = round(35.0 - ((m.congestion_index - self.low_threshold) * 0.3), 1)
            elif m.congestion_index < 88.0:
                m.congestion_level = "High"
                m.avg_speed_kmh = round(20.0 - ((m.congestion_index - self.high_threshold) * 0.4), 1)
            else:
                m.congestion_level = "Severe Gridlock"
                m.avg_speed_kmh = max(5.0, round(10.0 - ((m.congestion_index - 88.0) * 0.3), 1))

            m.pcu_total = round(m.pcu_total, 1)
            m.queue_length_m = round(m.queue_length_m, 1)

        return metrics

    def record_snapshot(
        self,
        timestamp: str,
        intersection_id: str,
        lane_metrics: Dict[str, LaneMetrics],
    ) -> None:
        """Records a timestamped snapshot of all lane metrics for historical analytics."""
        for lane, m in lane_metrics.items():
            self.history.append({
                "timestamp": timestamp,
                "intersection_id": intersection_id,
                "approach_lane": lane,
                "cars": m.cars,
                "buses": m.buses,
                "trucks": m.trucks,
                "motorcycles": m.motorcycles,
                "total_vehicles": m.total_vehicles,
                "pcu_total": m.pcu_total,
                "density_proxy": m.density_proxy,
                "flow_rate_per_min": m.flow_rate_per_min,
                "traffic_state": m.traffic_state,
                "congestion_index": m.congestion_index,
                "congestion_level": m.congestion_level,
                "queue_length_m": m.queue_length_m,
                "average_speed_kmh": m.avg_speed_kmh,
            })

    def get_history_df(self) -> pd.DataFrame:
        """Returns recorded history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    @staticmethod
    def load_sample_dataset(csv_path: str) -> pd.DataFrame:
        """Loads and formats the sample traffic dataset."""
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def compute_kpi_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates global KPI summaries from traffic records."""
        if df.empty:
            return {
                "total_vehicles_processed": 0,
                "avg_pcu": 0.0,
                "avg_static_delay": 0.0,
                "avg_adaptive_delay": 0.0,
                "delay_saved_pct": 0.0,
                "co2_saved_kg": 0.0,
                "avg_density_proxy": 0.0,
                "peak_density_proxy": 0.0,
            }

        total_vehicles = int(df["total_vehicles"].sum()) if "total_vehicles" in df.columns else 0
        avg_pcu = float(df["pcu_total"].mean()) if "pcu_total" in df.columns else 0.0

        avg_static_delay = float(df["delay_seconds_static"].mean()) if "delay_seconds_static" in df.columns else 45.0
        avg_adaptive_delay = float(df["delay_seconds_adaptive"].mean()) if "delay_seconds_adaptive" in df.columns else 26.0

        if avg_static_delay > 0:
            delay_saved_pct = round(((avg_static_delay - avg_adaptive_delay) / avg_static_delay) * 100.0, 1)
        else:
            delay_saved_pct = 0.0

        # Environmental metric: ~0.00035 kg CO2 per idling vehicle-second
        idle_time_saved = max(0.0, (avg_static_delay - avg_adaptive_delay) * total_vehicles)
        co2_saved_kg = round(idle_time_saved * 0.00035, 2)

        # Density proxy KPI
        if "density_proxy" in df.columns:
            avg_density = round(float(df["density_proxy"].mean()), 2)
            peak_density = round(float(df["density_proxy"].max()), 2)
        else:
            avg_density = round(avg_pcu / 1.0, 2)
            peak_density = round(float(df["pcu_total"].max()) if "pcu_total" in df.columns else 0.0, 2)

        return {
            "total_vehicles_processed": total_vehicles,
            "avg_pcu": round(avg_pcu, 1),
            "avg_static_delay": round(avg_static_delay, 1),
            "avg_adaptive_delay": round(avg_adaptive_delay, 1),
            "delay_saved_pct": delay_saved_pct,
            "co2_saved_kg": co2_saved_kg,
            "avg_density_proxy": avg_density,
            "peak_density_proxy": peak_density,
        }
