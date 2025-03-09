import pickle
import pandas as pd

# Cargar modelo entrenado
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Generar predicciones para los próximos 90 días
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Guardar predicciones
forecast.to_csv("dairy_forecast_predictions.csv", index=False)
print("✅ Pronóstico generado y guardado.")
