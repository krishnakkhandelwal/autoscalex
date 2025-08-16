import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/Hackathon/Data/genAI.csv')


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


df['Memory_Usage (%)'] = df['Memory_Usage (%)'].apply(lambda x: max(x, 0))

df['Business_Event'] = df['Business_Event'].fillna('None')

df = pd.get_dummies(df, columns=['Business_Event'])

df.drop(['Timestamp'], axis=1, inplace=True)

df.to_csv('D:/Hackathon/Data/genAI_minimal.csv', index=False)

