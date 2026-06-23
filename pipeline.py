import requests
import pandas as pd

API_KEY = "h8DQH3SQszpIbRExapPen1mWWwfiqNu6FHVKlgyE"
URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

params = {
    "api_key": API_KEY,
    "frequency": "hourly",
    "data[]": "value",
    "facets[respondent][]": "CAL",
    "facets[type][]": "D",  # D = demand
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "length": 168,
}

r = requests.get(URL, params=params)
data = r.json()["response"]["data"]
df = pd.DataFrame(data)
df["period"] = pd.to_datetime(df["period"])
df["value"] = pd.to_numeric(df["value"])
df = df.sort_values("period").reset_index(drop=True)
print(df.head())

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np

df["hour"] = df["period"].dt.hour
df["dayofweek"] = df["period"].dt.dayofweek
df["rolling_mean"] = df["value"].rolling(window=24, min_periods=1).mean()
df["rolling_std"] = df["value"].rolling(window=24, min_periods=1).std().fillna(0)
df["deviation"] = df["value"] - df["rolling_mean"]
features = df[["value", "hour", "dayofweek", "rolling_mean", "deviation"]].fillna(0)
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(features)
df["is_anomaly"] = df["anomaly"] == -1
anomalies = df[df["is_anomaly"]]
print(f"Anomalies found: {len(anomalies)}")
print(anomalies[["period", "value", "deviation"]].to_string())
plt.figure(figsize=(12, 4))
plt.plot(df["period"], df["value"], label="Demand")
plt.scatter(anomalies["period"], anomalies["value"], color="red", zorder=5, label="Anomaly")
df.to_csv("/Users/ruchi/Desktop/eia_demand_raw.csv", index=False)
print("CSV saved to Desktop")