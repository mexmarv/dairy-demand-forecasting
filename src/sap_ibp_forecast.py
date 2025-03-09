import pandas as pd
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load Data
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dairy_forecast_data.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Ensure SARIMA Captures Seasonality Correctly
sarima_model = SARIMAX(
    df["y"],
    order=(2,1,2),  # Fine-tuned for better short-term fluctuation detection
    seasonal_order=(1,1,2,12),  # Strengthen seasonal component for yearly cycles
    trend="t",  # Add a trend component
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

# Save Model
MODEL_PATH = os.path.join(BASE_DIR, "models", "sap_ibp_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(sarima_model, f)

print("✅ SAP IBP Model (SARIMA) Trained & Saved!")
