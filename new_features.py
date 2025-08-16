import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('genAI.csv')


print("Data Summary:")
print(df.info())

print("\nMissing Values Per Column:")
print(df.isnull().sum())


print("\nRows with Negative or >100% CPU/Memory Usage:")
print(df[(df['CPU_Usage (%)'] < 0) | (df['CPU_Usage (%)'] > 100)])
print(df[(df['Memory_Usage (%)'] < 0) | (df['Memory_Usage (%)'] > 100)])


print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='Active_Users', y='CPU_Usage (%)', hue='Business_Event')
plt.title('Active Users vs CPU Usage by Business Event')
plt.savefig('users_vs_cpu.png')

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='Active_Users', y='Memory_Usage (%)', hue='Business_Event')
plt.title('Active Users vs Memory Usage by Business Event')
plt.savefig('users_vs_memory.png')

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='CPU_Usage (%)', y='Memory_Usage (%)', hue='Scaling_Action')
plt.title('CPU Usage vs Memory Usage by Scaling Action')
plt.savefig('cpu_vs_memory_scaling.png')

print("EDA complete and visualizations saved as PNG files.")

import numpy as np


df['Memory_Usage (%)'] = df['Memory_Usage (%)'].apply(lambda x: max(x, 0))


df['Business_Event'] = df['Business_Event'].fillna('None')


df = pd.get_dummies(df, columns=['Business_Event'])

# Add lag features
for col in ['CPU_Usage (%)', 'Memory_Usage (%)', 'Active_Users']:
    df[f'{col}_Lag1'] = df[col].shift(1)
    df[f'{col}_Lag2'] = df[col].shift(2)

# Add rolling averages
df['CPU_Usage_Roll3'] = df['CPU_Usage (%)'].rolling(window=3).mean()
df['Memory_Usage_Roll3'] = df['Memory_Usage (%)'].rolling(window=3).mean()
df['Active_Users_Roll3'] = df['Active_Users'].rolling(window=3).mean()

# Cyclical encoding for Hour and Day
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)

# Upcoming Event flag (if any business event in next 2 hours)
event_cols = [col for col in df.columns if col.startswith('Business_Event_') and col != 'Business_Event_None']
df['Upcoming_Event'] = False
for idx in range(len(df) - 2):
    if df.loc[idx + 1: idx + 2, event_cols].any(axis=1).any():
        df.at[idx, 'Upcoming_Event'] = True

# Drop rows with NaNs from lag/rolling calculations
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop Timestamp, Day, Hour
df.drop(['Timestamp', 'Day', 'Hour'], axis=1, inplace=True)

# Save to CSV
df.to_csv('genAI_feature_engineer.csv', index=False)


