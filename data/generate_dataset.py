import pandas as pd
import numpy as np
import os

# Get Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dairy_forecast_data.csv")

# 📌 Real Dairy Consumption Seasonality in Mexico
# Source: INEGI, SIAP, FAO Dairy Reports
seasonality_pattern = np.array([0.95, 0.98, 1.05, 1.08, 1.12, 1.15, 1.10, 1.08, 0.97, 0.94, 0.92, 1.05])  

# Normalize Seasonality
seasonality_pattern = seasonality_pattern / np.max(seasonality_pattern)  # Scale between 0-1

# Generate 2 Years of Daily Data
start_date = "2023-10-01"
num_days = 730
date_range = pd.date_range(start=start_date, periods=num_days, freq="D")

# Base Dairy Sales (Liters Per Day)
base_sales = 250  # Base daily sales

# Create DataFrame
df = pd.DataFrame({"Date": date_range})

# Apply Monthly Seasonality
df["month"] = df["Date"].dt.month
df["seasonality"] = df["month"].apply(lambda x: seasonality_pattern[x - 1])

# Add Controlled Randomness to Simulate Demand Variability
np.random.seed(42)
random_variation = np.random.randint(-10, 10, size=len(df))  # Small daily fluctuations
df["Sales_Volume"] = (base_sales * df["seasonality"]) + random_variation

# Ensure Data is Integer
df["Sales_Volume"] = df["Sales_Volume"].astype(int)

# Drop Unnecessary Columns
df = df.drop(columns=["month", "seasonality"])

# Save to CSV
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
df.to_csv(DATA_PATH, index=False)

print(f"✅ Dairy sales data generated successfully from {start_date} with real seasonality!")
