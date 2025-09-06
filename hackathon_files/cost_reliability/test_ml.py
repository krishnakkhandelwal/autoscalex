# cost_reliability/test_ml.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml_integration import MLCostIntegration

def test_ml():
    print("Running ML Integration test...")
    ml = MLCostIntegration()
    sample_data = {
        'active_users': 1200,
        'cpu_usage': 60,
        'memory_usage': 50,
        'has_event': 0,
        'event_payroll': 0,
        'event_tax': 0,
        'event_eom': 0,
        'is_business_hours': True,
        'current_capacity': 1000
    }
    decision, prob = ml.predict_scaling_need(sample_data)
    print(f"Prediction: {'Scale' if decision else 'No Scale'} with probability {prob:.3f}")

if __name__ == "__main__":
    test_ml()
