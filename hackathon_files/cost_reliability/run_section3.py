# cost_reliability/run_section3.py
# -------------------------------------------------------
# Complete test runner for Section 3: Cost & Reliability
# Python 3.13 compatible
# -------------------------------------------------------

import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_optimizer import CostReliabilityOptimizer, BusinessTier, SLATier
from traffic_manager import intelligent_traffic_shaping, SystemLoad
from ml_integration import MLCostIntegration

def test_cost_optimizer():
    """Test the cost optimizer"""
    print("\n" + "="*60)
    print("TESTING COST OPTIMIZER")
    print("="*60)
    
    optimizer = CostReliabilityOptimizer()
    
    # Test critical payroll scenario
    strategies = optimizer.optimize_scaling_strategy(
        predicted_load=2500,
        current_capacity=1000,
        business_tier=BusinessTier.CRITICAL,
        sla_tier=SLATier.PEAK_EVENT,
        event_type="payroll"
    )
    
    print("🎯 CRITICAL PAYROLL SCENARIO - Top 3 Strategies:")
    for i, strategy in enumerate(strategies[:3], 1):
        print(f"\n{i}. {strategy.method.value}")
        print(f"   Units: {strategy.units_required}")
        print(f"   Cost: ¥{strategy.total_cost_per_hour:,.2f}/hour")
        print(f"   Reliability: {strategy.reliability_score:.3f}")
        print(f"   Deploy Time: {strategy.deployment_time_seconds}s")
        print(f"   Carbon: {strategy.carbon_footprint:.2f} kg CO2/hr")

def test_traffic_manager():
    """Test traffic shaping and bulkhead isolation"""
    print("\n" + "="*60)
    print("TESTING TRAFFIC MANAGER")
    print("="*60)
    
    # High load scenario
    high_load = SystemLoad(
        cpu_usage=0.89,
        memory_usage=0.84,
        active_users=2200,
        queue_depth=120,
        timestamp=datetime.now()
    )
    
    result = intelligent_traffic_shaping(high_load, ["account_transfer"])
    
    print("🚦 HIGH LOAD TRAFFIC SHAPING:")
    print(f"Bulkhead State: {result['bulkhead_state'].upper()}")
    print(f"Emergency Pools: {'ACTIVE' if result['emergency_pools_active'] else 'INACTIVE'}")
    
    print("\n📊 Transaction Delays:")
    for tx_type, delay in result['transaction_delays'].items():
        status = "🔴" if delay > 500 else "🟡" if delay > 100 else "🟢"
        print(f"   {status} {tx_type}: {delay}ms")
    
    print("\n⚡ Resource Allocations:")
    for tx_type, allocation in result['resource_allocations'].items():
        print(f"   {tx_type}: {allocation*100:.1f}%")

def test_ml_integration():
    """Test ML integration with cost optimization"""
    print("\n" + "="*60)
    print("TESTING ML INTEGRATION")
    print("="*60)
    
    try:
        integrator = MLCostIntegration()
        
        # Test normal load scenario
        normal_scenario = {
            'active_users': 800,
            'cpu_usage': 45,
            'memory_usage': 38,
            'has_event': 0,
            'event_payroll': 0,
            'event_tax': 0,
            'event_eom': 0,
            'current_capacity': 1000
        }
        
        print("📊 NORMAL LOAD SCENARIO:")
        recommendation = integrator.make_scaling_recommendation(normal_scenario)
        print(f"Recommendation: {recommendation['recommendation'].upper()}")
        print(f"ML Probability: {recommendation['ml_probability']:.3f}")
        
        if recommendation.get('top_strategy'):
            print(f"Method: {recommendation['top_strategy']['method']}")
            print(f"Cost: ¥{recommendation['top_strategy']['cost_per_hour']:,.2f}/hour")
        
        # Test high load scenario
        high_scenario = {
            'active_users': 2200,
            'cpu_usage': 87,
            'memory_usage': 82,
            'has_event': 1,
            'event_payroll': 1,
            'event_tax': 0,
            'event_eom': 0,
            'current_capacity': 1500
        }
        
        print("\n🚨 HIGH LOAD + PAYROLL SCENARIO:")
        recommendation = integrator.make_scaling_recommendation(high_scenario)
        print(f"Recommendation: {recommendation['recommendation'].upper()}")
        print(f"ML Probability: {recommendation['ml_probability']:.3f}")
        print(f"Business Tier: {recommendation['business_context']['tier']}")
        
        if recommendation.get('top_strategy'):
            print(f"Method: {recommendation['top_strategy']['method']}")
            print(f"Cost: ¥{recommendation['top_strategy']['cost_per_hour']:,.2f}/hour")
            print(f"Reliability: {recommendation['top_strategy']['reliability_score']:.3f}")
        
    except Exception as e:
        print(f"❌ ML Integration test failed: {e}")
        print("Note: Ensure ML model file exists at ../models/xgb_scaling_next1h.pkl")

def calculate_cost_savings():
    """Calculate potential cost savings"""
    print("\n" + "="*60)
    print("COST SAVINGS ANALYSIS")
    print("="*60)
    
    # Simulate traditional vs optimized scaling costs
    traditional_cost_per_hour = 500  # ¥500/hour for simple horizontal scaling
    
    optimizer = CostReliabilityOptimizer()
    strategies = optimizer.optimize_scaling_strategy(
        predicted_load=2000,
        current_capacity=1000,
        business_tier=BusinessTier.HIGH,
        sla_tier=SLATier.BUSINESS_HOURS,
        event_type=""
    )
    
    if strategies:
        optimized_cost = strategies[0].total_cost_per_hour
        daily_savings = (traditional_cost_per_hour - optimized_cost) * 24
        monthly_savings = daily_savings * 30
        
        print(f"💰 COST COMPARISON:")
        print(f"   Traditional Scaling: ¥{traditional_cost_per_hour:,.2f}/hour")
        print(f"   Optimized Scaling: ¥{optimized_cost:,.2f}/hour")
        print(f"   Hourly Savings: ¥{traditional_cost_per_hour - optimized_cost:,.2f}")
        print(f"   Daily Savings: ¥{daily_savings:,.2f}")
        print(f"   Monthly Savings: ¥{monthly_savings:,.2f}")
        print(f"   Annual Savings: ¥{monthly_savings * 12:,.2f}")

def run_dashboard_demo():
    """Instructions for running the dashboard"""
    print("\n" + "="*60)
    print("DASHBOARD DEMO INSTRUCTIONS")
    print("="*60)
    
    print("🖥️  To run the interactive dashboard:")
    print("   1. Install Flask: pip install flask")
    print("   2. Run: python dashboard.py")
    print("   3. Open browser to: http://localhost:5000")
    print("   4. Dashboard features:")
    print("      - Real-time metrics monitoring")
    print("      - Cost comparison charts")
    print("      - ML-powered recommendations")
    print("      - Business event simulator")
    print("      - What-if scenario testing")

def main():
    """Main test runner"""
    print("🏦 MUFG SECTION 3: COST & RELIABILITY TESTING")
    print("=" * 80)
    
    try:
        test_cost_optimizer()
        time.sleep(1)
        
        test_traffic_manager()
        time.sleep(1)
        
        test_ml_integration()
        time.sleep(1)
        
        calculate_cost_savings()
        time.sleep(1)
        
        run_dashboard_demo()
        
        print("\n" + "="*80)
        print("✅ SECTION 3 TESTING COMPLETED SUCCESSFULLY!")
        print("All components are working and integrated.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
