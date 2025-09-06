# -----------------------------------------------
# Section 1: Data Preparation + Visualizations
# -----------------------------------------------
# Prerequisites (run in terminal if needed):
#   pip install pandas numpy seaborn matplotlib
# -----------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fast matplotlib settings for high-DPI screens
plt.rcParams["figure.dpi"] = 130
sns.set(context="notebook", style="whitegrid")

# -----------------------------------------------
# 0) Load dataset
# -----------------------------------------------
INPUT_CSV = "hackathon_dataset.csv"
OUTPUT_IMG_DIR = "outputs"
OUTPUT_DATA_DIR = "Data"
OUTPUT_FEATURE_FILE = os.path.join(OUTPUT_DATA_DIR, "genAI_featured.csv")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# -----------------------------------------------
# 1) EDA + Data Quality Checks (lightweight)
# -----------------------------------------------
print("Data Summary:")
print(df.info())

print("\nMissing Values Per Column:")
print(df.isnull().sum())

print("\nRows with Negative or >100% CPU Usage:")
print(df[(df['CPU_Usage (%)'] < 0) | (df['CPU_Usage (%)'] > 100)])

print("\nRows with Negative or >100% Memory Usage:")
print(df[(df['Memory_Usage (%)'] < 0) | (df['Memory_Usage (%)'] > 100)])

print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# -----------------------------------------------
# 2) Quick Visualizations (raw)
# -----------------------------------------------
# Scatter: Active Users vs CPU by Business Event
plt.figure(figsize=(9,4))
sns.scatterplot(data=df, x='Active_Users', y='CPU_Usage (%)', hue='Business_Event', s=26, palette='husl')
plt.title('Active Users vs CPU Usage by Business Event')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'users_vs_cpu.png'))
plt.close()

# Scatter: Active Users vs Memory by Business Event
plt.figure(figsize=(9,4))
sns.scatterplot(data=df, x='Active_Users', y='Memory_Usage (%)', hue='Business_Event', s=26, palette='husl')
plt.title('Active Users vs Memory Usage by Business Event')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'users_vs_memory.png'))
plt.close()

# Scatter: CPU vs Memory by Scaling Action
plt.figure(figsize=(9,4))
sns.scatterplot(data=df, x='CPU_Usage (%)', y='Memory_Usage (%)', hue='Scaling_Action', s=26, palette='Set1')
plt.title('CPU Usage vs Memory Usage by Scaling Action')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'cpu_vs_memory_scaling.png'))
plt.close()

print(f"EDA visualizations saved to {OUTPUT_IMG_DIR}/")

# -----------------------------------------------
# 3) Cleaning
# -----------------------------------------------
# Parse timestamp to datetime (coerce for safety)
df['Datetime'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%y %H:%M', errors='coerce')

# Clip invalid percentage values
for col in ['CPU_Usage (%)', 'Memory_Usage (%)']:
    df[col] = df[col].clip(lower=0, upper=100)

# Normalize Business_Event: treat "None" or "" as NaN
df['Business_Event'] = df['Business_Event'].replace({'None': np.nan, '': np.nan})

# -----------------------------------------------
# 4) High-impact Feature Engineering
# -----------------------------------------------
# Temporal features
df['Hour'] = df['Datetime'].dt.hour
df['Weekday'] = df['Datetime'].dt.weekday  # Mon=0..Sun=6
df['DayOfMonth'] = df['Datetime'].dt.day
df['WeekOfMonth'] = ((df['DayOfMonth'] - 1) // 7) + 1
df['Is_Business_Hours'] = df['Hour'].between(9, 19).astype(int)

# Event features
df['Has_Event'] = df['Business_Event'].notna().astype(int)
df['Event_Payroll'] = (df['Business_Event'] == 'Payroll Processing').astype(int)
df['Event_Tax'] = (df['Business_Event'] == 'Tax Filing').astype(int)
df['Event_EoM'] = (df['Business_Event'] == 'Month-End Reporting').astype(int)

# Risk scoring (tunable for demo)
df['Event_Risk_Score'] = 0.9*df['Event_Payroll'] + 0.8*df['Event_Tax'] + 0.7*df['Event_EoM']

# Event day proximity (hours to next {5,10,15,20,25,30} near peak windows)
def hours_to_next_event_day(ts):
    if pd.isna(ts):
        return np.nan
    day = ts.day
    candidates = [5,10,15,20,25,30]
    future_days = [d for d in candidates if d >= day]
    if not future_days:
        # Approximate wrap to next month 5th
        delta_days = (30 - day) + 5
    else:
        delta_days = min(future_days) - day
    # Peak operating hours near event windows
    peak_hours = np.array([10,12,14,16,18,19])
    hour = ts.hour
    hour_dist = int(np.min(np.abs(peak_hours - hour)))
    return int(delta_days*24 + hour_dist)

df['Event_Day_Proximity_Hours'] = df['Datetime'].apply(hours_to_next_event_day)

# Lags, trends, rolling stats
for c in ['Active_Users','CPU_Usage (%)','Memory_Usage (%)']:
    df[f'{c}_Lag1'] = df[c].shift(1)
    df[f'{c}_Trend1'] = df[c] - df[f'{c}_Lag1']
    df[f'{c}_MA3'] = df[c].rolling(3).mean()
    df[f'{c}_MA6'] = df[c].rolling(6).mean()
    df[f'{c}_PctChange1'] = (
        df[c].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .clip(-1, 1)
    )

# Saturation indices (simultaneous pressure on CPU+Memory)
eps = 1e-6
df['Saturation_Max'] = df[['CPU_Usage (%)','Memory_Usage (%)']].max(axis=1)
df['Saturation_HMean'] = 2 * (df['CPU_Usage (%)'] * df['Memory_Usage (%)'] + eps) / (df['CPU_Usage (%)'] + df['Memory_Usage (%)'] + eps)

# Queue Pressure proxy using rolling P90 on last 2 rows (approx. 2h in your hourly data)
def rolling_quantile(series, window, q):
    return series.rolling(window=window, min_periods=window).quantile(q)

df['Users_P90_2h'] = rolling_quantile(df['Active_Users'], 2, 0.90)
df['CPU_Trend1_P90_2h'] = rolling_quantile(df['CPU_Usage (%)'].diff(1), 2, 0.90)
df['Mem_Trend1_P90_2h'] = rolling_quantile(df['Memory_Usage (%)'].diff(1), 2, 0.90)

df['Queue_Pressure'] = (
    (df['Active_Users'] > df['Users_P90_2h']) &
    (df['CPU_Usage (%)'].diff(1) > df['CPU_Trend1_P90_2h']) &
    (df['Memory_Usage (%)'].diff(1) > df['Mem_Trend1_P90_2h'])
).astype(int).fillna(0)

# Lead target (predictive) for pre-scaling
df['Scaling_Next_1h'] = df['Scaling_Action'].shift(-1).fillna(0).astype(int)

# -----------------------------------------------
# 5) Post-engineering Visuals (feature insight)
# -----------------------------------------------
# Heatmap (correlations of key numeric features)
corr_cols = [
    'Active_Users','CPU_Usage (%)','Memory_Usage (%)',
    'Saturation_Max','Saturation_HMean','Event_Risk_Score',
    'Is_Business_Hours','Has_Event','Scaling_Action'
]
corr_df = df[corr_cols].copy()
plt.figure(figsize=(9,6))
sns.heatmap(corr_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=False)
plt.title('Correlation Heatmap (Selected Features)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'corr_heatmap_selected.png'))
plt.close()

# Time series strip: mark scaling events on CPU timeline (first ~200 points for readability)
preview = df.head(200).copy()
plt.figure(figsize=(10,4))
plt.plot(preview['Datetime'], preview['CPU_Usage (%)'], label='CPU%')
scale_ts = preview.loc[preview['Scaling_Action'] == 1, 'Datetime']
scale_vals = preview.loc[preview['Scaling_Action'] == 1, 'CPU_Usage (%)']
plt.scatter(scale_ts, scale_vals, color='red', s=28, label='Scaling Action=1')
plt.title('CPU Usage with Scaling Events (Preview)')
plt.xlabel('Time'); plt.ylabel('CPU%'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'cpu_timeline_scaling_preview.png'))
plt.close()

print(f"Feature insight visuals saved to {OUTPUT_IMG_DIR}/")

# -----------------------------------------------
# 6) Finalize dataset: drop redundant/noisy columns
# -----------------------------------------------
drop_cols = [
    'Timestamp',          # raw string
    'Day',                # redundant to Weekday
    'Business_Event',     # replaced by flags/score
    'Users_P90_2h', 'CPU_Trend1_P90_2h', 'Mem_Trend1_P90_2h'  # intermediate helpers
]
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# Drop rows with NA due to rolling/lag at the start
df_clean = df.dropna().reset_index(drop=True)

# Reorder columns: targets at the end
target_cols = ['Scaling_Action', 'Scaling_Next_1h']
feature_cols = [c for c in df_clean.columns if c not in target_cols]
df_final = df_clean[feature_cols + target_cols]

# -----------------------------------------------
# 7) Save model-ready file
# -----------------------------------------------
df_final.to_csv(OUTPUT_FEATURE_FILE, index=False)
print(f"Feature engineering complete. Saved model-ready file: {OUTPUT_FEATURE_FILE}")


# -----------------------------------------------
# Trim to lean, high-signal feature set (26 cols)
# -----------------------------------------------
keep_cols = [
    'Datetime','Hour','Weekday','WeekOfMonth','Is_Business_Hours',
    'Active_Users','CPU_Usage (%)','Memory_Usage (%)',
    'Has_Event','Event_Payroll','Event_Tax','Event_EoM','Event_Risk_Score','Event_Day_Proximity_Hours',
    'Saturation_Max',
    'Active_Users_Lag1','CPU_Usage (%)_Lag1','Memory_Usage (%)_Lag1',
    'Active_Users_Trend1','CPU_Usage (%)_Trend1','Memory_Usage (%)_Trend1',
    'Active_Users_MA3','CPU_Usage (%)_MA3','Memory_Usage (%)_MA3',
    'Scaling_Action','Scaling_Next_1h'
]
missing = [c for c in keep_cols if c not in df_final.columns]
if missing:
    raise ValueError(f"Missing expected columns in df_final: {missing}")

df_trim = df_final[keep_cols].copy()
df_trim.to_csv('Data/genAI_featured_trimmed.csv', index=False)
print('Saved trimmed feature file to Data/genAI_featured_trimmed.csv')

# -----------------------------------------------
# 8) Quick summary of resulting columns
# -----------------------------------------------
print("\nFinal Columns:")
print(list(df_final.columns))
print(f"\nRows: {len(df_final)}  |  Columns: {len(df_final.columns)}")
print("Section 1 complete.")