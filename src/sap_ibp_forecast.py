import pandas as pd
import os
import pickle
import gzip
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 📌 Cargar Datos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = "data/dairy_forecast_data.csv"
MODEL_PATH = "models/sap_ibp_model.pkl.gz"  # 📌 Guardar comprimido

df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# 📌 Entrenar SARIMA con Parámetros Optimizados
sarima_model = SARIMAX(
    df["y"],
    order=(1,1,1),  # 📌 Parámetros reducidos
    seasonal_order=(1,1,1,12),  # 📌 Menos complejidad en estacionalidad
    trend="c",  # 📌 Se mantiene solo tendencia constante
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)  # 📌 No guardar logs innecesarios

# 📌 Guardar Modelo Comprimido (GZIP)
os.makedirs("models", exist_ok=True)
with gzip.open(MODEL_PATH, "wb") as f:
    pickle.dump(sarima_model, f)

print("✅ Modelo SAP IBP (SARIMA) reducido y guardado como `.pkl.gz`")
