# cost_reliability/traffic_manager.py
# -------------------------------------------------------
# Traffic shaping and bulkhead isolation for constrained periods
# Python 3.13 compatible
# -------------------------------------------------------

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import queue
import threading

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    CRITICAL_BANKING = "critical_banking"      # Customer transactions
    PAYROLL = "payroll"                        # Payroll processing
    REGULATORY = "regulatory"                  # Compliance reporting
    ANALYTICS = "analytics"                    # Data analytics
    MAINTENANCE = "maintenance"                # System maintenance

class BulkheadState(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded" 
    EMERGENCY = "emergency"

@dataclass
class TrafficRule:
    transaction_type: TransactionType
    priority_weight: float
    max_delay_ms: int
    resource_allocation: float  # 0.0 to 1.0

@dataclass
class SystemLoad:
    cpu_usage: float
    memory_usage: float
    active_users: int
    queue_depth: int
    timestamp: datetime

class TrafficShaper:
    def __init__(self):
        self.traffic_rules = self._initialize_traffic_rules()
        self.current_load = SystemLoad(0, 0, 0, 0, datetime.now())
        self.bulkhead_state = BulkheadState.NORMAL
        self.request_queues = {t: queue.PriorityQueue() for t in TransactionType}
        self.is_running = False
        
    def _initialize_traffic_rules(self) -> Dict[TransactionType, TrafficRule]:
        """Initialize traffic shaping rules"""
        return {
            TransactionType.CRITICAL_BANKING: TrafficRule(
                TransactionType.CRITICAL_BANKING, 1.0, 100, 0.6
            ),
            TransactionType.PAYROLL: TrafficRule(
                TransactionType.PAYROLL, 0.9, 200, 0.25
            ),
            TransactionType.REGULATORY: TrafficRule(
                TransactionType.REGULATORY, 0.8, 500, 0.1
            ),
            TransactionType.ANALYTICS: TrafficRule(
                TransactionType.ANALYTICS, 0.3, 2000, 0.05
            ),
            TransactionType.MAINTENANCE: TrafficRule(
                TransactionType.MAINTENANCE, 0.1, 5000, 0.0
            )
        }
    
    def update_system_load(self, cpu: float, memory: float, 
                          active_users: int, queue_depth: int):
        """Update current system load metrics"""
        self.current_load = SystemLoad(
            cpu, memory, active_users, queue_depth, datetime.now()
        )
        self._evaluate_bulkhead_state()
    
    def _evaluate_bulkhead_state(self):
        """Determine current bulkhead state based on system load"""
        cpu_threshold_emergency = 0.95
        cpu_threshold_degraded = 0.85
        memory_threshold_emergency = 0.90
        memory_threshold_degraded = 0.75
        
        if (self.current_load.cpu_usage >= cpu_threshold_emergency or 
            self.current_load.memory_usage >= memory_threshold_emergency):
            self.bulkhead_state = BulkheadState.EMERGENCY
        elif (self.current_load.cpu_usage >= cpu_threshold_degraded or
              self.current_load.memory_usage >= memory_threshold_degraded):
            self.bulkhead_state = BulkheadState.DEGRADED
        else:
            self.bulkhead_state = BulkheadState.NORMAL
            
        logger.info(f"Bulkhead state: {self.bulkhead_state.value}")
    
    def calculate_delay(self, transaction_type: TransactionType) -> int:
        """Calculate delay for transaction type based on current load"""
        rule = self.traffic_rules[transaction_type]
        base_delay = rule.max_delay_ms
        
        if self.bulkhead_state == BulkheadState.NORMAL:
            return 0
        elif self.bulkhead_state == BulkheadState.DEGRADED:
            # Apply proportional delay based on priority
            load_factor = (self.current_load.cpu_usage + self.current_load.memory_usage) / 2
            delay_multiplier = (1 - rule.priority_weight) * load_factor
            return int(base_delay * delay_multiplier * 0.5)
        else:  # EMERGENCY
            # Critical transactions get minimal delay, others get maximum
            if rule.priority_weight >= 0.8:
                return int(base_delay * 0.1)
            else:
                return base_delay
    
    def get_resource_allocation(self, transaction_type: TransactionType) -> float:
        """Get current resource allocation for transaction type"""
        rule = self.traffic_rules[transaction_type]
        
        if self.bulkhead_state == BulkheadState.EMERGENCY:
            # In emergency, redistribute resources to critical transactions
            if rule.priority_weight >= 0.8:
                return min(1.0, rule.resource_allocation * 1.5)
            else:
                return rule.resource_allocation * 0.3
        elif self.bulkhead_state == BulkheadState.DEGRADED:
            # In degraded state, slightly favor critical transactions
            if rule.priority_weight >= 0.8:
                return min(1.0, rule.resource_allocation * 1.2)
            else:
                return rule.resource_allocation * 0.7
        else:
            return rule.resource_allocation

class BulkheadManager:
    def __init__(self):
        self.circuit_breakers = {}
        self.isolation_pools = {}
        self.health_checks = {}
        
    def create_isolation_pool(self, pool_name: str, max_connections: int,
                            transaction_types: List[TransactionType]):
        """Create isolated resource pool for specific transaction types"""
        self.isolation_pools[pool_name] = {
            'max_connections': max_connections,
            'current_connections': 0,
            'transaction_types': transaction_types,
            'created_at': datetime.now()
        }
        logger.info(f"Created isolation pool: {pool_name}")
    
    def activate_emergency_bulkheads(self):
        """Activate emergency isolation measures"""
        # Create emergency pools for critical transactions
        self.create_isolation_pool(
            'critical_banking_emergency',
            max_connections=100,
            transaction_types=[TransactionType.CRITICAL_BANKING]
        )
        
        self.create_isolation_pool(
            'payroll_emergency', 
            max_connections=50,
            transaction_types=[TransactionType.PAYROLL]
        )
        
        logger.warning("Emergency bulkheads activated")
    
    def get_pool_for_transaction(self, transaction_type: TransactionType) -> Optional[str]:
        """Get appropriate pool for transaction type"""
        for pool_name, pool_info in self.isolation_pools.items():
            if transaction_type in pool_info['transaction_types']:
                if pool_info['current_connections'] < pool_info['max_connections']:
                    return pool_name
        return None

def intelligent_traffic_shaping(current_load: SystemLoad, 
                              critical_transactions: List[str]) -> Dict:
    """Main traffic shaping function"""
    shaper = TrafficShaper()
    bulkhead = BulkheadManager()
    
    shaper.update_system_load(
        current_load.cpu_usage,
        current_load.memory_usage, 
        current_load.active_users,
        current_load.queue_depth
    )
    
    # Calculate delays for each transaction type
    delays = {}
    allocations = {}
    
    for tx_type in TransactionType:
        delays[tx_type.value] = shaper.calculate_delay(tx_type)
        allocations[tx_type.value] = shaper.get_resource_allocation(tx_type)
    
    # Activate emergency measures if needed
    if shaper.bulkhead_state == BulkheadState.EMERGENCY:
        bulkhead.activate_emergency_bulkheads()
    
    return {
        'bulkhead_state': shaper.bulkhead_state.value,
        'transaction_delays': delays,
        'resource_allocations': allocations,
        'emergency_pools_active': len(bulkhead.isolation_pools) > 0,
        'timestamp': current_load.timestamp.isoformat()
    }

def main():
    """Test traffic shaping and bulkhead isolation"""
    # Simulate high load scenario
    high_load = SystemLoad(
        cpu_usage=0.92,
        memory_usage=0.87,
        active_users=2500,
        queue_depth=150,
        timestamp=datetime.now()
    )
    
    critical_tx = ["account_transfer", "balance_inquiry"]
    
    result = intelligent_traffic_shaping(high_load, critical_tx)
    
    print("\n=== Traffic Shaping Results ===")
    print(f"Bulkhead State: {result['bulkhead_state']}")
    print(f"Emergency Pools Active: {result['emergency_pools_active']}")
    print("\nTransaction Delays (ms):")
    for tx_type, delay in result['transaction_delays'].items():
        print(f"  {tx_type}: {delay}ms")
    print("\nResource Allocations:")
    for tx_type, allocation in result['resource_allocations'].items():
        print(f"  {tx_type}: {allocation*100:.1f}%")

if __name__ == "__main__":
    main()
