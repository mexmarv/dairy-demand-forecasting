import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ✅ Cargar Datos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dairy_forecast_data.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Generar Features para XGBoost
df["year"] = df["ds"].dt.year
df["month"] = df["ds"].dt.month
df["day"] = df["ds"].dt.day
df["day_of_week"] = df["ds"].dt.dayofweek

X = df[["year", "month", "day", "day_of_week"]]
y = df["y"]

# ✅ Separar Datos en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Entrenar Modelo XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)

# ✅ Evaluar el Modelo
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 MAE del modelo XGBoost (Oracle SCM Simulado): {mae}")

# ✅ Guardar Modelo
MODEL_PATH = os.path.join(BASE_DIR, "models", "oracle_scm_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ Modelo Oracle SCM (XGBoost) Entrenado y Guardado")
