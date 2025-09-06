# cost_reliability/ml_integration.py
# -------------------------------------------------------
# Debugged ML integration with proper path handling and fallbacks
# Python 3.13 compatible - Windows path fix
# -------------------------------------------------------

import sys
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

# Fix path resolution for Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, 'models')

# Try different possible model locations
possible_model_paths = [
    os.path.join(models_dir, 'xgb_scaling_next1h.pkl'),
    os.path.join(parent_dir, 'models', 'xgb_scaling_next1h.pkl'),
    '../models/xgb_scaling_next1h.pkl',
    'models/xgb_scaling_next1h.pkl',
    './models/xgb_scaling_next1h.pkl'
]

from cost_optimizer import CostReliabilityOptimizer, BusinessTier, SLATier, ScalingStrategy
from traffic_manager import intelligent_traffic_shaping, SystemLoad

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_model_file():
    """Find the ML model file in possible locations"""
    for path in possible_model_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            logger.info(f"Found model file at: {abs_path}")
            return abs_path
    
    logger.warning("No ML model file found in any of these locations:")
    for path in possible_model_paths:
        logger.warning(f"  - {os.path.abspath(path)}")
    return None

class MockMLModel:
    """Mock ML model for when the real model isn't available"""
    
    def __init__(self):
        self.features = [
            'Hour', 'Weekday', 'WeekOfMonth', 'Is_Business_Hours',
            'Active_Users', 'CPU_Usage (%)', 'Memory_Usage (%)',
            'Has_Event', 'Event_Payroll', 'Event_Tax', 'Event_EoM',
            'Event_Risk_Score', 'Event_Day_Proximity_Hours', 'Saturation_Max',
            'Active_Users_Lag1', 'CPU_Usage (%)_Lag1', 'Memory_Usage (%)_Lag1',
            'Active_Users_Trend1', 'CPU_Usage (%)_Trend1', 'Memory_Usage (%)_Trend1',
            'Active_Users_MA3', 'CPU_Usage (%)_MA3', 'Memory_Usage (%)_MA3'
        ]
        self.threshold = 0.5
        logger.info("Initialized Mock ML Model with heuristic rules")
    
    def predict_proba(self, X):
        """Mock prediction using heuristic rules"""
        probabilities = []
        
        for _, row in X.iterrows():
            # Extract key features
            cpu = row.get('CPU_Usage (%)', 0)
            memory = row.get('Memory_Usage (%)', 0)
            users = row.get('Active_Users', 0)
            has_event = row.get('Has_Event', 0)
            is_business_hours = row.get('Is_Business_Hours', 0)
            saturation = row.get('Saturation_Max', 0)
            
            # Heuristic scoring
            score = 0.0
            
            # High utilization increases probability
            if cpu > 85:
                score += 0.4
            elif cpu > 70:
                score += 0.2
            
            if memory > 80:
                score += 0.3
            elif memory > 60:
                score += 0.1
            
            # High user load
            if users > 2000:
                score += 0.3
            elif users > 1500:
                score += 0.15
            
            # Business events increase probability
            if has_event:
                score += 0.25
            
            # Business hours multiplier
            if is_business_hours:
                score *= 1.2
            
            # Saturation penalty
            if saturation > 90:
                score += 0.2
            
            # Normalize to probability
            probability = min(0.95, max(0.05, score))
            probabilities.append([1 - probability, probability])
        
        return np.array(probabilities)

class MLCostIntegration:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or find_model_file()
        self.ml_model = None
        self.features = []
        self.threshold = 0.5
        self.is_mock = False
        
        self.load_ml_model()
        self.cost_optimizer = CostReliabilityOptimizer()
        self.prediction_history = []
        
    def load_ml_model(self):
        """Load the trained XGBoost model with fallback to mock"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Attempting to load ML model from: {self.model_path}")
                bundle = joblib.load(self.model_path)
                self.ml_model = bundle['model']
                self.features = bundle['features']
                self.threshold = float(bundle.get('chosen_threshold', 0.5))
                self.is_mock = False
                logger.info(f"✅ Successfully loaded real ML model with {len(self.features)} features")
                logger.info(f"   Threshold: {self.threshold:.3f}")
                return
            except Exception as e:
                logger.error(f"Failed to load real ML model: {e}")
        
        # Fallback to mock model
        logger.warning("⚠️ Using Mock ML Model (heuristic-based predictions)")
        mock_model = MockMLModel()
        self.ml_model = mock_model
        self.features = mock_model.features
        self.threshold = mock_model.threshold
        self.is_mock = True
    
    def predict_scaling_need(self, current_data: Dict) -> Tuple[bool, float]:
        """Predict if scaling is needed and probability"""
        try:
            # Prepare features for ML model
            feature_row = self._prepare_features(current_data)
            
            # Make prediction
            X = pd.DataFrame([feature_row], columns=self.features)
            proba = float(self.ml_model.predict_proba(X)[:, 1][0])
            needs_scaling = proba >= self.threshold
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'probability': proba,
                'prediction': needs_scaling,
                'features': current_data,
                'is_mock': self.is_mock
            })
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return needs_scaling, proba
            
        except Exception as e:
            logger.error(f"Error in predict_scaling_need: {e}")
            # Emergency fallback
            cpu = current_data.get('cpu_usage', 0)
            memory = current_data.get('memory_usage', 0)
            emergency_prob = min(0.9, (cpu + memory) / 200.0)
            return emergency_prob > 0.7, emergency_prob
    
    def _prepare_features(self, data: Dict) -> Dict:
        """Prepare features for ML model from current system data"""
        feature_row = {}
        
        try:
            # Time features
            now = datetime.now()
            feature_row['Hour'] = now.hour
            feature_row['Weekday'] = now.weekday()
            feature_row['WeekOfMonth'] = ((now.day - 1) // 7) + 1
            feature_row['Is_Business_Hours'] = int(9 <= now.hour <= 19)
            
            # System metrics
            feature_row['Active_Users'] = int(data.get('active_users', 0))
            feature_row['CPU_Usage (%)'] = float(data.get('cpu_usage', 0))
            feature_row['Memory_Usage (%)'] = float(data.get('memory_usage', 0))
            
            # Event features
            feature_row['Has_Event'] = int(data.get('has_event', 0))
            feature_row['Event_Payroll'] = int(data.get('event_payroll', 0))
            feature_row['Event_Tax'] = int(data.get('event_tax', 0))
            feature_row['Event_EoM'] = int(data.get('event_eom', 0))
            feature_row['Event_Risk_Score'] = float(data.get('event_risk_score', 0))
            feature_row['Event_Day_Proximity_Hours'] = int(data.get('event_proximity', 72))
            
            # Derived metrics
            feature_row['Saturation_Max'] = max(
                feature_row['CPU_Usage (%)'],
                feature_row['Memory_Usage (%)']
            )
            
            # Lag features (from history or defaults)
            feature_row['Active_Users_Lag1'] = int(data.get('active_users_lag1', feature_row['Active_Users'] * 0.9))
            feature_row['CPU_Usage (%)_Lag1'] = float(data.get('cpu_lag1', feature_row['CPU_Usage (%)'] * 0.9))
            feature_row['Memory_Usage (%)_Lag1'] = float(data.get('memory_lag1', feature_row['Memory_Usage (%)'] * 0.9))
            
            # Trend features
            feature_row['Active_Users_Trend1'] = feature_row['Active_Users'] - feature_row['Active_Users_Lag1']
            feature_row['CPU_Usage (%)_Trend1'] = feature_row['CPU_Usage (%)'] - feature_row['CPU_Usage (%)_Lag1']
            feature_row['Memory_Usage (%)_Trend1'] = feature_row['Memory_Usage (%)'] - feature_row['Memory_Usage (%)_Lag1']
            
            # Moving averages (simplified - in real implementation would use historical data)
            feature_row['Active_Users_MA3'] = feature_row['Active_Users']
            feature_row['CPU_Usage (%)_MA3'] = feature_row['CPU_Usage (%)']
            feature_row['Memory_Usage (%)_MA3'] = feature_row['Memory_Usage (%)']
            
            return feature_row
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return minimal feature set
            return {feature: 0 for feature in self.features}
    
    def get_business_context(self, data: Dict) -> Tuple[BusinessTier, SLATier, str]:
        """Determine business context from current data"""
        try:
            # Determine business tier
            if data.get('event_payroll') or data.get('event_tax'):
                business_tier = BusinessTier.CRITICAL
            elif data.get('is_business_hours', False) and data.get('active_users', 0) > 1000:
                business_tier = BusinessTier.HIGH
            else:
                business_tier = BusinessTier.STANDARD
                
            # Determine SLA tier
            current_hour = datetime.now().hour
            has_event = data.get('has_event', False)
            
            if has_event and business_tier == BusinessTier.CRITICAL:
                sla_tier = SLATier.PEAK_EVENT
            elif 9 <= current_hour <= 19:
                sla_tier = SLATier.BUSINESS_HOURS
            else:
                sla_tier = SLATier.OFF_HOURS
                
            # Event type
            event_type = ""
            if data.get('event_payroll'):
                event_type = "payroll"
            elif data.get('event_tax'):
                event_type = "tax_filing"
            elif data.get('event_eom'):
                event_type = "month_end_reporting"
                
            return business_tier, sla_tier, event_type
            
        except Exception as e:
            logger.error(f"Error getting business context: {e}")
            return BusinessTier.STANDARD, SLATier.BUSINESS_HOURS, ""
    
    def make_scaling_recommendation(self, current_data: Dict) -> Dict:
        """Complete scaling recommendation with cost optimization"""
        try:
            # Get ML prediction
            needs_scaling, probability = self.predict_scaling_need(current_data)
            
            if not needs_scaling:
                return {
                    'recommendation': 'no_scaling',
                    'probability': probability,
                    'ml_probability': probability,
                    'model_type': 'mock' if self.is_mock else 'xgboost',
                    'reasoning': f'{"Heuristic" if self.is_mock else "ML"} model predicts no scaling needed (prob: {probability:.3f})',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get business context
            business_tier, sla_tier, event_type = self.get_business_context(current_data)
            
            # Calculate predicted load
            current_capacity = current_data.get('current_capacity', 1000)
            predicted_load = current_capacity * (1 + probability * 0.5)  # Conservative load estimation
            
            # Get cost-optimized strategies
            strategies = self.cost_optimizer.optimize_scaling_strategy(
                predicted_load=predicted_load,
                current_capacity=current_capacity,
                business_tier=business_tier,
                sla_tier=sla_tier,
                event_type=event_type
            )
            
            # Prepare traffic shaping if needed
            system_load = SystemLoad(
                cpu_usage=current_data.get('cpu_usage', 0) / 100.0,
                memory_usage=current_data.get('memory_usage', 0) / 100.0,
                active_users=current_data.get('active_users', 0),
                queue_depth=current_data.get('queue_depth', 0),
                timestamp=datetime.now()
            )
            
            traffic_config = intelligent_traffic_shaping(system_load, [])
            
            # Format recommendation
            top_strategy = strategies[0] if strategies else None
            
            recommendation = {
                'recommendation': 'scale',
                'ml_probability': probability,
                'model_type': 'mock' if self.is_mock else 'xgboost',
                'predicted_load': predicted_load,
                'business_context': {
                    'tier': business_tier.value,
                    'sla': sla_tier.name,
                    'event_type': event_type
                },
                'top_strategy': {
                    'method': top_strategy.method.value,
                    'units_required': top_strategy.units_required,
                    'cost_per_hour': top_strategy.total_cost_per_hour,
                    'reliability_score': top_strategy.reliability_score,
                    'deployment_time': top_strategy.deployment_time_seconds
                } if top_strategy else None,
                'alternative_strategies': [
                    {
                        'method': s.method.value,
                        'cost_per_hour': s.total_cost_per_hour,
                        'reliability_score': s.reliability_score
                    } for s in strategies[1:4]  # Top 3 alternatives
                ],
                'traffic_management': traffic_config,
                'reasoning': f'{"Heuristic" if self.is_mock else "ML"} predicts scaling needed (prob: {probability:.3f}). '
                            f'Recommended: {top_strategy.method.value if top_strategy else "None"} '
                            f'for {business_tier.value} tier workload.',
                'timestamp': datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in make_scaling_recommendation: {e}")
            return {
                'recommendation': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'mock' if self.is_mock else 'xgboost',
            'model_path': self.model_path,
            'features_count': len(self.features),
            'threshold': self.threshold,
            'predictions_made': len(self.prediction_history)
        }

def demo_integration():
    """Demo the ML-Cost integration with comprehensive testing"""
    logger.info("Starting ML-Cost Integration Demo")
    
    try:
        integrator = MLCostIntegration()
        
        # Display model info
        model_info = integrator.get_model_info()
        print("\n" + "="*60)
        print("🤖 ML MODEL INFORMATION")
        print("="*60)
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Test scenario 1: Normal load
        print("\n" + "="*60)
        print("📊 SCENARIO 1: NORMAL BUSINESS LOAD")
        print("="*60)
        
        normal_data = {
            'active_users': 800,
            'cpu_usage': 45,
            'memory_usage': 38,
            'has_event': 0,
            'event_payroll': 0,
            'event_tax': 0,
            'event_eom': 0,
            'is_business_hours': True,
            'current_capacity': 1000
        }
        
        recommendation = integrator.make_scaling_recommendation(normal_data)
        print(f"Recommendation: {recommendation['recommendation'].upper()}")
        print(f"Model Type: {recommendation.get('model_type', 'unknown')}")
        print(f"Probability: {recommendation.get('ml_probability', recommendation.get('probability', 0)):.3f}")
        
        if recommendation.get('top_strategy'):
            print(f"Method: {recommendation['top_strategy']['method']}")
            print(f"Cost: ¥{recommendation['top_strategy']['cost_per_hour']:,.2f}/hour")
        
        # Test scenario 2: High load with payroll event
        print("\n" + "="*60)
        print("🚨 SCENARIO 2: HIGH LOAD + PAYROLL EVENT")
        print("="*60)
        
        peak_data = {
            'active_users': 2200,
            'cpu_usage': 87,
            'memory_usage': 82,
            'has_event': 1,
            'event_payroll': 1,
            'event_tax': 0,
            'event_eom': 0,
            'is_business_hours': True,
            'current_capacity': 1500
        }
        
        recommendation = integrator.make_scaling_recommendation(peak_data)
        print(f"Recommendation: {recommendation['recommendation'].upper()}")
        print(f"Model Type: {recommendation.get('model_type', 'unknown')}")
        print(f"Probability: {recommendation.get('ml_probability', recommendation.get('probability', 0)):.3f}")
        print(f"Business Tier: {recommendation.get('business_context', {}).get('tier', 'unknown')}")
        
        if recommendation.get('top_strategy'):
            print(f"Method: {recommendation['top_strategy']['method']}")
            print(f"Cost: ¥{recommendation['top_strategy']['cost_per_hour']:,.2f}/hour")
            print(f"Reliability: {recommendation['top_strategy']['reliability_score']:.3f}")
        
        # Test scenario 3: Edge case - very high load
        print("\n" + "="*60)
        print("🔥 SCENARIO 3: EXTREME LOAD EDGE CASE")
        print("="*60)
        
        extreme_data = {
            'active_users': 4500,
            'cpu_usage': 95,
            'memory_usage': 93,
            'has_event': 1,
            'event_payroll': 1,
            'event_tax': 1,
            'event_eom': 0,
            'is_business_hours': True,
            'current_capacity': 2000
        }
        
        recommendation = integrator.make_scaling_recommendation(extreme_data)
        print(f"Recommendation: {recommendation['recommendation'].upper()}")
        print(f"Model Type: {recommendation.get('model_type', 'unknown')}")
        print(f"Probability: {recommendation.get('ml_probability', recommendation.get('probability', 0)):.3f}")
        
        if recommendation.get('top_strategy'):
            print(f"Method: {recommendation['top_strategy']['method']}")
            print(f"Cost: ¥{recommendation['top_strategy']['cost_per_hour']:,.2f}/hour")
            print(f"Units: {recommendation['top_strategy']['units_required']}")
        
        print("\n" + "="*60)
        print("✅ ML INTEGRATION DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_paths():
    """Test different model path configurations"""
    print("\n" + "="*60)
    print("🔍 MODEL PATH DIAGNOSTICS")
    print("="*60)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {os.path.exists(models_dir)}")
    
    print("\nChecking possible model locations:")
    for i, path in enumerate(possible_model_paths, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"  {i}. {path}")
        print(f"     → {abs_path}")
        print(f"     → {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    found_path = find_model_file()
    if found_path:
        print(f"\n🎯 Model found at: {found_path}")
    else:
        print(f"\n⚠️ No model file found. Will use heuristic fallback.")

def main():
    """Main function with comprehensive testing"""
    print("🤖 MUFG ML-COST INTEGRATION TESTING")
    print("=" * 80)
    
    # Test model paths first
    test_model_paths()
    
    # Run integration demo
    demo_integration()

if __name__ == "__main__":
    main()
