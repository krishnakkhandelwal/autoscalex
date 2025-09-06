# cost_reliability/terminal_dashboard.py
# Python 3.13+ terminal dashboard (no HTML) for Section 3

import os
import time
import random
from datetime import datetime

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.layout import Layout
from rich.text import Text
from rich import box

from ml_integration import MLCostIntegration
from traffic_manager import intelligent_traffic_shaping, SystemLoad

console = Console()

def simulate_metrics():
    now = datetime.now()
    hour = now.hour
    base_users = 600
    if 9 <= hour <= 17:
        base_users *= 2
    if hour in (10, 14):
        base_users = int(base_users * 1.5)

    users = max(100, int(random.gauss(base_users, base_users * 0.2)))
    cpu = max(10.0, min(98.0, 30 + users * 0.02 + random.gauss(0, 8)))
    mem = max(10.0, min(98.0, 25 + users * 0.015 + random.gauss(0, 6)))

    has_event = random.random() < 0.12
    event_payroll = has_event and random.random() < 0.35
    event_tax = has_event and random.random() < 0.30
    event_eom = has_event and random.random() < 0.35

    return {
        "active_users": users,
        "cpu_usage": round(cpu, 1),
        "memory_usage": round(mem, 1),
        "has_event": int(has_event),
        "event_payroll": int(event_payroll),
        "event_tax": int(event_tax),
        "event_eom": int(event_eom),
        "event_risk_score": 0.7 if has_event else 0.0,
        "is_business_hours": int(9 <= hour <= 19),
        "current_capacity": 1000,
        "queue_depth": int(users * 0.05),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

def build_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout

def header_text():
    return Panel(Text("MUFG Cost & Reliability - Terminal Dashboard", style="bold white"), style="on blue", padding=(0, 2))

def footer_text(model_mode):
    msg = f"Press Ctrl+C to exit | Model: {model_mode}"
    return Panel(Text(msg, style="bold"), style="on grey23")

def make_metrics_table(data):
    tbl = Table(title="Live Metrics", box=box.SIMPLE_HEAVY, expand=True)
    tbl.add_column("Metric", justify="left")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Active Users", str(data["active_users"]))
    tbl.add_row("CPU Usage (%)", f'{data["cpu_usage"]:.1f}')
    tbl.add_row("Memory Usage (%)", f'{data["memory_usage"]:.1f}')
    tbl.add_row("Has Event", str(data["has_event"]))
    tbl.add_row("Payroll/Tax/EoM", f'{data["event_payroll"]}/{data["event_tax"]}/{data["event_eom"]}')
    tbl.add_row("Queue Depth", str(data["queue_depth"]))
    tbl.add_row("Timestamp", data["timestamp"])
    return tbl

def make_recommendation_panel(rec):
    if rec.get("recommendation") == "scale":
        method = rec["top_strategy"]["method"] if rec.get("top_strategy") else "N/A"
        cost = rec["top_strategy"]["cost_per_hour"] if rec.get("top_strategy") else 0.0
        reliability = rec["top_strategy"]["reliability_score"] if rec.get("top_strategy") else 0.0
        prob = rec.get("ml_probability", rec.get("probability", 0.0))
        tier = rec.get("business_context", {}).get("tier", "unknown")
        sla = rec.get("business_context", {}).get("sla", "unknown")
        txt = (
            f"[yellow bold]Scaling Recommended[/yellow bold]\n"
            f"Method: {method}\n"
            f"Cost/Hour: ¥{cost:,.2f}\n"
            f"Reliability: {reliability:.3f}\n"
            f"Probability: {prob:.3f}\n"
            f"Business Tier: {tier} | SLA: {sla}"
        )
        return Panel(Text(txt), title="Recommendation", border_style="yellow")
    else:
        prob = rec.get("ml_probability", rec.get("probability", 0.0))
        txt = (
            f"[green bold]No Scaling Needed[/green bold]\n"
            f"Probability: {prob:.3f}\n"
            f"Reason: {rec.get('reasoning','')}"
        )
        return Panel(Text(txt), title="Recommendation", border_style="green")

def make_bulkhead_panel(traffic_cfg):
    state = traffic_cfg.get("bulkhead_state", "unknown").upper()
    pools = "ACTIVE" if traffic_cfg.get("emergency_pools_active") else "INACTIVE"
    delays = traffic_cfg.get("transaction_delays", {})
    allocs = traffic_cfg.get("resource_allocations", {})

    tbl = Table(box=box.MINIMAL, expand=True)
    tbl.add_column("Txn Type", justify="left")
    tbl.add_column("Delay (ms)", justify="right")
    tbl.add_column("Alloc (%)", justify="right")

    for t in sorted(delays.keys()):
        d = delays[t]
        a = allocs.get(t, 0.0)
        tbl.add_row(t, str(d), f"{a*100:.1f}")

    title = f"Bulkhead: {state} | Emergency Pools: {pools}"
    return Panel(tbl, title=title, border_style="magenta")

def run_terminal_dashboard():
    ml = MLCostIntegration()
    model_mode = "mock" if getattr(ml, "is_mock", False) else "xgboost"
    layout = build_layout()
    try:
        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                data = simulate_metrics()
                recommendation = ml.make_scaling_recommendation(data)
                system_load = SystemLoad(
                    cpu_usage=data["cpu_usage"] / 100.0,
                    memory_usage=data["memory_usage"] / 100.0,
                    active_users=data["active_users"],
                    queue_depth=data["queue_depth"],
                    timestamp=datetime.now(),
                )
                traffic_cfg = intelligent_traffic_shaping(system_load, [])

                layout["header"].update(header_text())
                layout["left"].update(make_metrics_table(data))
                layout["right"].update(make_recommendation_panel(recommendation))
                layout["footer"].update(footer_text(model_mode))
                console.print(make_bulkhead_panel(traffic_cfg))
                time.sleep(2)
    except KeyboardInterrupt:
        console.clear()
        console.print("\n[bold cyan]Stopped Terminal Dashboard.[/bold cyan]")

if __name__ == "__main__":
    run_terminal_dashboard()
