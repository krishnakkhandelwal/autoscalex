import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_path = 'Data/genAI_feature_engineered.csv'
df = pd.read_csv(data_path)

target_col = 'Scaling_Action'
feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols].values
y = df[target_col].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled[target_col] = y
df_scaled.to_csv('D:/Hackathon/Data/genAI_rnn_scaled.csv', index=False)

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 3
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

np.save('Data/X_seq.npy', X_seq)
np.save('Data/y_seq.npy', y_seq)

print("Saved")

