import pandas as pd
import pickle
import os
from prophet import Prophet

# Get Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dairy_forecast_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")

# ✅ Ensure Data File Exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de datos: {DATA_PATH}")

# ✅ Load Data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Handle Missing Values
df = df.dropna()  # Remove any missing data

# ✅ Train Prophet Model
print("🔄 Entrenando modelo Prophet...")
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, changepoint_prior_scale=0.05)
model.fit(df)

# ✅ Ensure Models Directory Exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ✅ Save Model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo Prophet entrenado y guardado correctamente.")
