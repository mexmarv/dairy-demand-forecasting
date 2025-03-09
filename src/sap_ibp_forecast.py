import pandas as pd
import os
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ✅ Load Data
DATA_PATH = "data/dairy_forecast_data.csv"
EXTERNAL_DATA_PATH = "data/external_factors.csv"
MODEL_PATH = "models/sap_ibp_model.pkl"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Merge External Data
if os.path.exists(EXTERNAL_DATA_PATH):
    external_df = pd.read_csv(EXTERNAL_DATA_PATH, parse_dates=["Date"]).rename(columns={"Date": "ds"})
    df = df.merge(external_df, on="ds", how="left")

# ✅ Ensure No Missing Columns (Fixed for Pandas 3.0)
df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())
df["Price"] = df["Price"].fillna(df["Price"].mean())


# ✅ Create Features
df["year"] = df["ds"].dt.year
df["month"] = df["ds"].dt.month
df["day"] = df["ds"].dt.day
df["day_of_week"] = df["ds"].dt.dayofweek

# ✅ Use Correct Features
X = df[["year", "month", "day", "day_of_week", "Temperature", "Price"]]
y = df["y"]

# ✅ Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
lgb_model.fit(X_train, y_train)

# ✅ Save Model
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(lgb_model, f)

print("✅ SAP IBP (LightGBM) Model Trained and Saved.")
