import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
            "relative_humidity_2m_mean", "surface_pressure_mean",
            "doy_sin", "doy_cos", "is_monsoon", "precipitation_sum"]

data = df[features].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

SEQ_LEN = 14
target_idx = features.index("precipitation_sum")

X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i, target_idx])
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

split = int(len(X) * 0.8)
X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
y_train, y_test = torch.tensor(y[:split]).unsqueeze(1), torch.tensor(y[split:]).unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

model = LSTMModel(input_size=len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30, Loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).squeeze().numpy()

dummy = np.zeros((len(pred_scaled), len(features)))
dummy[:, target_idx] = pred_scaled
pred = scaler.inverse_transform(dummy)[:, target_idx]
pred = np.clip(pred, 0, None)

y_test_np = y_test.squeeze().numpy()
dummy_true = np.zeros((len(y_test_np), len(features)))
dummy_true[:, target_idx] = y_test_np
y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

mae = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)
print(f"\nLSTM (PyTorch): MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")