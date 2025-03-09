import pandas as pd
import os
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 📌 Cargar Datos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = "data/dairy_forecast_data.csv"
MODEL_PATH = "models/sap_ibp_model.pkl"  # 📌 Modelo optimizado

df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# 📌 Crear Variables de Tiempo
df["year"] = df["ds"].dt.year
df["month"] = df["ds"].dt.month
df["day"] = df["ds"].dt.day
df["day_of_week"] = df["ds"].dt.dayofweek

# 📌 Separar Datos en Entrenamiento y Prueba
X = df[["year", "month", "day", "day_of_week"]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Entrenar LightGBM en lugar de SARIMA
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
lgb_model.fit(X_train, y_train)

# 📌 Evaluar Precisión del Modelo
y_pred = lgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 MAE del modelo LightGBM (SAP IBP optimizado): {mae}")

# 📌 Guardar Modelo en Formato Ligero
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(lgb_model, f)

print("✅ Modelo SAP IBP Optimizado con LightGBM Guardado")
