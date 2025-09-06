# cost_reliability/cost_optimizer.py
# -------------------------------------------------------
# Section 3: Cost & Reliability Optimization Engine
# Multi-dimensional cost optimization with business impact
# Python 3.13 compatible
# -------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingMethod(Enum):
    HORIZONTAL_PODS = "horizontal_pods"
    VERTICAL_SCALE = "vertical_scale"
    CACHE_BOOST = "cache_boost"
    DB_REPLICAS = "db_replicas"
    CDN_ACTIVATION = "cdn_activation"

class BusinessTier(Enum):
    CRITICAL = "critical"      # Payroll, Regulatory
    HIGH = "high"             # Customer transactions
    STANDARD = "standard"     # Analytics, reporting

class SLATier(Enum):
    PEAK_EVENT = {"availability": 99.99, "latency_ms": 100}
    BUSINESS_HOURS = {"availability": 99.9, "latency_ms": 200}
    OFF_HOURS = {"availability": 99.5, "latency_ms": 500}

@dataclass
class CostModel:
    # Cost per unit per hour in JPY
    horizontal_pods: float = 12.0      # ¥/pod/hour
    vertical_scale: float = 8.0        # ¥/GB/hour
    cache_boost: float = 15.0          # ¥/GB cache/hour
    db_replicas: float = 20.0          # ¥/replica/hour
    cdn_activation: float = 5.0        # ¥/GB transferred/hour
    
    # Business impact multipliers
    downtime_cost_per_minute: float = 50000.0  # ¥50,000/min
    regulatory_penalty_risk: float = 2000000.0  # ¥2M potential penalty
    reputation_impact_daily: float = 10000000.0  # ¥10M daily reputation cost

@dataclass
class ScalingStrategy:
    method: ScalingMethod
    units_required: float
    total_cost_per_hour: float
    reliability_score: float
    business_alignment: float
    carbon_footprint: float
    deployment_time_seconds: int

class CostReliabilityOptimizer:
    def __init__(self, config_path: Optional[str] = None):
        self.cost_model = CostModel()
        self.load_config(config_path)
        
    def load_config(self, config_path: Optional[str]):
        """Load configuration from JSON file if provided"""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.cost_model = CostModel(**config.get('cost_model', {}))
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
    
    def calculate_required_units(self, method: ScalingMethod, predicted_load: float, 
                               current_capacity: float) -> float:
        """Calculate units needed for each scaling method"""
        capacity_gap = max(0, predicted_load - current_capacity)
        
        if method == ScalingMethod.HORIZONTAL_PODS:
            # Assume each pod handles 100 concurrent users
            return np.ceil(capacity_gap / 100)
        elif method == ScalingMethod.VERTICAL_SCALE:
            # Scale memory/CPU in GB
            return np.ceil(capacity_gap * 0.01)  # 1GB per 100 users
        elif method == ScalingMethod.CACHE_BOOST:
            # Cache scaling based on read load
            return np.ceil(capacity_gap * 0.005)  # 0.5GB cache per 100 users
        elif method == ScalingMethod.DB_REPLICAS:
            # DB replicas for read-heavy loads
            return min(3, np.ceil(capacity_gap / 500))  # Max 3 replicas
        elif method == ScalingMethod.CDN_ACTIVATION:
            # CDN for static content
            return capacity_gap * 0.1  # 0.1GB transfer per user
        
        return 0
    
    def calculate_reliability_score(self, method: ScalingMethod, units: float, 
                                  sla_tier: SLATier) -> float:
        """Calculate reliability score for scaling method"""
        base_scores = {
            ScalingMethod.HORIZONTAL_PODS: 0.95,
            ScalingMethod.VERTICAL_SCALE: 0.85,
            ScalingMethod.CACHE_BOOST: 0.90,
            ScalingMethod.DB_REPLICAS: 0.88,
            ScalingMethod.CDN_ACTIVATION: 0.92
        }
        
        base_score = base_scores[method]
        
        # Adjust for SLA requirements
        if sla_tier == SLATier.PEAK_EVENT:
            reliability_multiplier = 1.0
        elif sla_tier == SLATier.BUSINESS_HOURS:
            reliability_multiplier = 0.95
        else:
            reliability_multiplier = 0.90
        
        # Scale with units (diminishing returns)
        unit_factor = min(1.0, 0.8 + 0.2 * np.log(units + 1))
        
        return base_score * reliability_multiplier * unit_factor
    
    def calculate_carbon_footprint(self, method: ScalingMethod, units: float) -> float:
        """Calculate carbon footprint in kg CO2 equivalent"""
        carbon_factors = {
            ScalingMethod.HORIZONTAL_PODS: 0.5,    # kg CO2/pod/hour
            ScalingMethod.VERTICAL_SCALE: 0.3,     # kg CO2/GB/hour
            ScalingMethod.CACHE_BOOST: 0.2,        # kg CO2/GB/hour
            ScalingMethod.DB_REPLICAS: 0.8,        # kg CO2/replica/hour
            ScalingMethod.CDN_ACTIVATION: 0.1      # kg CO2/GB/hour
        }
        
        return carbon_factors[method] * units
    
    def get_deployment_time(self, method: ScalingMethod) -> int:
        """Get deployment time in seconds"""
        deployment_times = {
            ScalingMethod.HORIZONTAL_PODS: 120,    # 2 minutes
            ScalingMethod.VERTICAL_SCALE: 300,     # 5 minutes
            ScalingMethod.CACHE_BOOST: 60,         # 1 minute
            ScalingMethod.DB_REPLICAS: 180,        # 3 minutes
            ScalingMethod.CDN_ACTIVATION: 30       # 30 seconds
        }
        
        return deployment_times[method]
    
    def calculate_business_alignment(self, method: ScalingMethod, 
                                   business_tier: BusinessTier,
                                   event_type: str) -> float:
        """Calculate how well scaling method aligns with business needs"""
        
        # Base alignment scores
        alignment_matrix = {
            BusinessTier.CRITICAL: {
                ScalingMethod.HORIZONTAL_PODS: 1.0,
                ScalingMethod.VERTICAL_SCALE: 0.8,
                ScalingMethod.CACHE_BOOST: 0.6,
                ScalingMethod.DB_REPLICAS: 0.9,
                ScalingMethod.CDN_ACTIVATION: 0.5
            },
            BusinessTier.HIGH: {
                ScalingMethod.HORIZONTAL_PODS: 0.9,
                ScalingMethod.VERTICAL_SCALE: 0.9,
                ScalingMethod.CACHE_BOOST: 0.8,
                ScalingMethod.DB_REPLICAS: 0.7,
                ScalingMethod.CDN_ACTIVATION: 0.8
            },
            BusinessTier.STANDARD: {
                ScalingMethod.HORIZONTAL_PODS: 0.7,
                ScalingMethod.VERTICAL_SCALE: 0.8,
                ScalingMethod.CACHE_BOOST: 0.9,
                ScalingMethod.DB_REPLICAS: 0.6,
                ScalingMethod.CDN_ACTIVATION: 0.9
            }
        }
        
        base_score = alignment_matrix[business_tier][method]
        
        # Event-specific adjustments
        if event_type in ['payroll', 'tax_filing']:
            if method in [ScalingMethod.HORIZONTAL_PODS, ScalingMethod.DB_REPLICAS]:
                base_score *= 1.2
        elif event_type == 'month_end_reporting':
            if method in [ScalingMethod.CACHE_BOOST, ScalingMethod.CDN_ACTIVATION]:
                base_score *= 1.1
                
        return min(1.0, base_score)
    
    def optimize_scaling_strategy(self, predicted_load: float, current_capacity: float,
                                business_tier: BusinessTier, sla_tier: SLATier,
                                event_type: str = "") -> List[ScalingStrategy]:
        """Main optimization function - returns ranked strategies"""
        
        strategies = []
        
        for method in ScalingMethod:
            units = self.calculate_required_units(method, predicted_load, current_capacity)
            
            if units <= 0:
                continue
                
            # Calculate costs
            cost_per_unit = getattr(self.cost_model, method.value)
            total_cost = cost_per_unit * units
            
            # Calculate other metrics
            reliability = self.calculate_reliability_score(method, units, sla_tier)
            business_alignment = self.calculate_business_alignment(method, business_tier, event_type)
            carbon_footprint = self.calculate_carbon_footprint(method, units)
            deployment_time = self.get_deployment_time(method)
            
            strategy = ScalingStrategy(
                method=method,
                units_required=units,
                total_cost_per_hour=total_cost,
                reliability_score=reliability,
                business_alignment=business_alignment,
                carbon_footprint=carbon_footprint,
                deployment_time_seconds=deployment_time
            )
            
            strategies.append(strategy)
        
        # Rank strategies by composite score
        for strategy in strategies:
            strategy.composite_score = self._calculate_composite_score(
                strategy, business_tier
            )
        
        # Sort by composite score (higher is better)
        strategies.sort(key=lambda x: x.composite_score, reverse=True)
        
        return strategies
    
    def _calculate_composite_score(self, strategy: ScalingStrategy, 
                                 business_tier: BusinessTier) -> float:
        """Calculate composite score for strategy ranking"""
        
        # Normalize costs (lower is better, so invert)
        cost_score = 1.0 / (1.0 + strategy.total_cost_per_hour / 100.0)
        
        # Reliability score (higher is better)
        reliability_score = strategy.reliability_score
        
        # Business alignment (higher is better)
        business_score = strategy.business_alignment
        
        # Carbon footprint (lower is better, so invert)
        carbon_score = 1.0 / (1.0 + strategy.carbon_footprint / 10.0)
        
        # Deployment time (lower is better, so invert)
        deployment_score = 1.0 / (1.0 + strategy.deployment_time_seconds / 300.0)
        
        # Weight based on business tier
        if business_tier == BusinessTier.CRITICAL:
            weights = {
                'reliability': 0.4, 'business': 0.3, 'deployment': 0.2, 
                'cost': 0.1, 'carbon': 0.0
            }
        elif business_tier == BusinessTier.HIGH:
            weights = {
                'reliability': 0.3, 'business': 0.25, 'deployment': 0.15,
                'cost': 0.25, 'carbon': 0.05
            }
        else:
            weights = {
                'reliability': 0.2, 'business': 0.2, 'deployment': 0.1,
                'cost': 0.4, 'carbon': 0.1
            }
        
        composite = (
            weights['reliability'] * reliability_score +
            weights['business'] * business_score +
            weights['deployment'] * deployment_score +
            weights['cost'] * cost_score +
            weights['carbon'] * carbon_score
        )
        
        return composite

def main():
    """Test the cost optimizer"""
    optimizer = CostReliabilityOptimizer()
    
    # Test scenario: Payroll processing peak load
    strategies = optimizer.optimize_scaling_strategy(
        predicted_load=2000,      # 2000 concurrent users
        current_capacity=1000,    # Current capacity
        business_tier=BusinessTier.CRITICAL,
        sla_tier=SLATier.PEAK_EVENT,
        event_type="payroll"
    )
    
    print("\n=== Cost & Reliability Optimization Results ===")
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy.method.value}")
        print(f"   Units Required: {strategy.units_required}")
        print(f"   Cost/Hour: ¥{strategy.total_cost_per_hour:,.2f}")
        print(f"   Reliability: {strategy.reliability_score:.3f}")
        print(f"   Business Fit: {strategy.business_alignment:.3f}")
        print(f"   Carbon: {strategy.carbon_footprint:.2f} kg CO2/hr")
        print(f"   Deploy Time: {strategy.deployment_time_seconds}s")
        print(f"   Score: {strategy.composite_score:.3f}")

if __name__ == "__main__":
    main()
