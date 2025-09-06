# visualizations/plot_scaling_overview.py
# -------------------------------------------------------
# Visual overview of Scaling_Next_1h (0/1) across time and key drivers.
# Reads: Data/genAI_featured_trimmed.csv
# Saves plots to: outputs/section2_plots
# Includes extra debug prints and guards to prevent common runtime errors.
# -------------------------------------------------------

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

INPUT_CSV = 'Data/genAI_featured_trimmed.csv'
OUT_DIR = 'outputs/section2_plots'

def debug(msg):
    print(f"[DEBUG] {msg}", flush=True)

def load_data():
    debug(f"Loading data from: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])
    debug(f"Loaded shape: {df.shape}")
    df = df.sort_values('Datetime').reset_index(drop=True)
    expected = [
        'Datetime','Hour','Weekday','WeekOfMonth','Is_Business_Hours',
        'Active_Users','CPU_Usage (%)','Memory_Usage (%)',
        'Has_Event','Event_Payroll','Event_Tax','Event_EoM','Event_Risk_Score','Event_Day_Proximity_Hours',
        'Saturation_Max',
        'Active_Users_Lag1','CPU_Usage (%)_Lag1','Memory_Usage (%)_Lag1',
        'Active_Users_Trend1','CPU_Usage (%)_Trend1','Memory_Usage (%)_Trend1',
        'Active_Users_MA3','CPU_Usage (%)_MA3','Memory_Usage (%)_MA3',
        'Scaling_Action','Scaling_Next_1h'
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Ensure binary int for scaling
    df['Scaling_Next_1h'] = df['Scaling_Next_1h'].astype(int)
    return df

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)
    debug(f"Output directory ready: {OUT_DIR}")

def plot_scaling_timeline(df):
    debug("Plotting scaling_timeline.png")
    plt.figure(figsize=(12,4))
    times = df['Datetime']
    y = df['Scaling_Next_1h'].astype(int)
    plt.plot(times, y, drawstyle='steps-post', color='tab:blue', linewidth=1.2, label='Scaling_Next_1h')

    # Overlay event windows
    ev_mask = df['Has_Event'] == 1
    if ev_mask.any():
        plt.scatter(df.loc[ev_mask, 'Datetime'],
                    df.loc[ev_mask, 'Scaling_Next_1h'] + 0.02,
                    s=18, color='tab:orange', alpha=0.7, label='Has_Event')

    # Overlay high CPU points
    hi_cpu = df['CPU_Usage (%)'] >= 85
    if hi_cpu.any():
        offsets = np.where(df.loc[hi_cpu, 'Scaling_Next_1h'].values == 1, 1.05, -0.05)
        plt.scatter(df.loc[hi_cpu, 'Datetime'], offsets,
                    s=10, color='tab:red', alpha=0.6, label='CPU≥85')

    plt.ylim(-0.1, 1.15)
    plt.title('Scaling_Next_1h over time')
    plt.xlabel('Datetime'); plt.ylabel('Scaling_Next_1h')
    plt.legend(loc='upper right')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'scaling_timeline.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

def plot_hourly_heatmap(df):
    debug("Plotting hourly_heatmap.png")
    pivot = df.pivot_table(values='Scaling_Next_1h', index='Weekday', columns='Hour',
                           aggfunc='mean', fill_value=0.0)
    plt.figure(figsize=(12,4))
    sns.heatmap(pivot, cmap='YlGnBu', vmin=0, vmax=1, cbar_kws={'label':'Scaling rate'})
    plt.title('Scaling rate by Weekday vs Hour')
    plt.ylabel('Weekday (Mon=0..Sun=6)'); plt.xlabel('Hour')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'hourly_heatmap.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

def plot_cpu_memory_scatter(df):
    debug("Plotting cpu_memory_scatter.png")
    plt.figure(figsize=(7,6))
    colors = df['Scaling_Next_1h'].map({0: 'tab:gray', 1: 'tab:green'})
    plt.scatter(df['CPU_Usage (%)'], df['Memory_Usage (%)'],
                c=colors, s=16, alpha=0.7, edgecolors='none')

    # Threshold guide lines
    plt.axvline(85, color='tab:red', linestyle='--', linewidth=1, alpha=0.6)
    plt.axhline(70, color='tab:red', linestyle='--', linewidth=1, alpha=0.6)

    plt.xlabel('CPU_Usage (%)'); plt.ylabel('Memory_Usage (%)')
    plt.title('CPU vs Memory colored by Scaling_Next_1h')

    # Robust legend handles (Line2D requires xdata and ydata)
    legend_elems = [
    Patch(facecolor='tab:green', label='Scaling=1'),
    Patch(facecolor='tab:gray', label='Scaling=0'),
    Line2D([], [], color='tab:red', linestyle='--', label='Heuristic thresholds')
]
    plt.legend(handles=legend_elems, loc='lower right')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'cpu_memory_scatter.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

def plot_saturation_timeline(df):
    debug("Plotting saturation_timeline.png")
    plt.figure(figsize=(12,4))
    plt.plot(df['Datetime'], df['Saturation_Max'], color='tab:purple', linewidth=1.2, label='Saturation_Max')

    # Scaling markers
    mask = df['Scaling_Next_1h'] == 1
    if mask.any():
        plt.scatter(df.loc[mask, 'Datetime'], df.loc[mask, 'Saturation_Max'],
                    s=18, color='tab:green', label='Scaling=1', zorder=3)

    # CPU trend overlay (offset for visibility)
    cpu_trend = df['CPU_Usage (%)_Trend1'].clip(-50, 50)
    plt.plot(df['Datetime'], cpu_trend + 60, color='tab:orange', alpha=0.6, linewidth=1,
             label='CPU_Trend1 (offset +60)')

    # Visual threshold for saturation
    plt.axhline(85, color='tab:red', linestyle='--', linewidth=1, alpha=0.6)

    plt.title('Saturation vs CPU trend with Scaling markers')
    plt.ylabel('Value'); plt.xlabel('Datetime')
    plt.legend(loc='upper right')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'saturation_timeline.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

def plot_event_lift(df):
    debug("Plotting event_lift_bars.png")
    groups = {
        'NoEvent': (df['Has_Event'] == 0),
        'Payroll': (df['Event_Payroll'] == 1),
        'Tax': (df['Event_Tax'] == 1),
        'EoM': (df['Event_EoM'] == 1)
    }
    names, rates, counts = [], [], []
    for name, m in groups.items():
        sub = df.loc[m, 'Scaling_Next_1h']
        names.append(name)
        if len(sub) == 0:
            rates.append(0.0); counts.append(0)
        else:
            rates.append(float(sub.mean())); counts.append(int(len(sub)))

    plt.figure(figsize=(7,4))
    ax = sns.barplot(x=names, y=rates, color='tab:blue')
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r + 0.01, f"{r:.2f}\n(n={n})", ha='center', va='bottom', fontsize=9)
    plt.ylim(0, 1)
    plt.title('Scaling rate by Event category')
    plt.ylabel('Scaling rate'); plt.xlabel('Event category')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'event_lift_bars.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

def main():
    print("Plotting Scaling_Next_1h dashboards...", flush=True)
    ensure_outdir()
    df = load_data()
    debug(f"Datetime range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    debug(f"Scaling=1 count: {int((df['Scaling_Next_1h']==1).sum())} of {len(df)} rows")

    plot_scaling_timeline(df)
    plot_hourly_heatmap(df)
    plot_cpu_memory_scatter(df)
    plot_saturation_timeline(df)
    plot_event_lift(df)

    print("All plots saved to:", OUT_DIR)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("[ERROR]", str(e))
        raise
