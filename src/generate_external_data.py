import pandas as pd
import numpy as np
import os

np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=730, freq="D")

temperature = np.random.normal(loc=22, scale=5, size=len(dates))
prices = np.random.uniform(18, 25, size=len(dates))

df = pd.DataFrame({"ds": dates, "Temperature": temperature, "Price": prices})  # ✅ `ds` corregido

df.to_csv("../data/external_factors.csv", index=False)

print("✅ Datos de clima y precios generados correctamente.")
