import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import subprocess
import sys

# ✅ Configuración inicial (Debe ser la primera línea)
st.set_page_config(page_title="Pronóstico Demanda Leche", page_icon="🥛", layout="wide")

# ✅ Sidebar: Theme Selection
st.sidebar.title("⚙Configuración del Pronóstico")
theme_choice = st.sidebar.radio("Modo de Visualización", ["🌙 Oscuro", "☀️ Claro"])

# ✅ Apply Theme Dynamically
selected_theme = "plotly_dark" if theme_choice == "🌙 Oscuro" else "plotly_white"
st.markdown(f"<style>body {{ background-color: {'#1e1e1e' if theme_choice == '🌙 Oscuro' else 'white'}; color: {'white' if theme_choice == '🌙 Oscuro' else 'black'}; }}</style>", unsafe_allow_html=True)

# ✅ Load Data
DATA_PATH = "data/dairy_forecast_data.csv"
SEASONALITY_PATH = "data/dairy_seasonality.csv"
EXTERNAL_DATA_PATH = "data/external_factors.csv"

uploaded_file = st.sidebar.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
df = pd.read_csv(uploaded_file if uploaded_file else DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Load External Factors (Fix `parse_dates` issue)
if os.path.exists(EXTERNAL_DATA_PATH):
    external_df = pd.read_csv(EXTERNAL_DATA_PATH)

    # ✅ Ensure correct column names
    if "Date" in external_df.columns:
        external_df.rename(columns={"Date": "ds"}, inplace=True)

    # ✅ Convert 'ds' to datetime in both DataFrames before merging
    external_df["ds"] = pd.to_datetime(external_df["ds"])
    df["ds"] = pd.to_datetime(df["ds"])

    if {"ds", "Temperature", "Price"}.issubset(external_df.columns):
        df = df.merge(external_df, on="ds", how="left")
    else:
        st.warning("⚠ El archivo de clima/precios no tiene las columnas correctas. Se usarán valores por defecto.")
else:
    st.warning("⚠ No se encontraron datos de clima y precios. Se usarán valores por defecto.")

# ✅ Default values if missing
df["Temperature"] = df["Temperature"].fillna(22)
df["Price"] = df["Price"].fillna(20)

# ✅ Sidebar: Model Selection
model_choice = st.sidebar.selectbox("📡 Modelo de Pronóstico:", ["Facebook Prophet", "SAP IBP (LightGBM)", "Oracle SCM (XGBoost)"])
days = st.sidebar.slider("⏳ Días a predecir:", 30, 365, 90)

# 📌 Train Model Button
if st.sidebar.button("Entrenar Modelo"):
    TRAIN_SCRIPTS = {
        "Facebook Prophet": "src/train_model.py",
        "SAP IBP (LightGBM)": "src/sap_ibp_forecast.py",
        "Oracle SCM (XGBoost)": "src/oracle_scm_forecast.py"
    }
    st.info("🔄 Instalando dependencias y entrenando modelo...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    process = subprocess.Popen([sys.executable, TRAIN_SCRIPTS[model_choice]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    training_logs = st.empty()
    log_text = ""
    for line in process.stdout:
        log_text += line + "\n"
        training_logs.text_area("📜 Registro de Entrenamiento:", log_text, height=200)

    process.wait()
    if process.returncode == 0:
        st.success(f"✅ {model_choice} entrenado correctamente.")
    else:
        st.error(f"❌ Error en el entrenamiento.")

# 📌 Load Model
MODEL_PATHS = {
    "Facebook Prophet": "models/trained_model.pkl",
    "SAP IBP (LightGBM)": "models/sap_ibp_model.pkl",
    "Oracle SCM (XGBoost)": "models/oracle_scm_model.pkl"
}
model_path = MODEL_PATHS[model_choice]

with open(model_path, "rb") as f:
    model = pickle.load(f)

# 📌 Generate Predictions
future_dates = pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=days, freq="D")

if model_choice == "Facebook Prophet":
    future = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
else:
    future_features = pd.DataFrame({
        "year": future_dates.year,
        "month": future_dates.month,
        "day": future_dates.day,
        "day_of_week": future_dates.dayofweek,
        "Temperature": df["Temperature"].mean(),
        "Price": df["Price"].mean()
    })
    forecast_values = model.predict(future_features)
    forecast = pd.DataFrame({"ds": future_dates, "yhat": forecast_values, "yhat_lower": forecast_values - 10, "yhat_upper": forecast_values + 10})

# ✅ Restaurar Icono de Alpura y Título
st.title("Pronóstico de la Demanda - Leche Alpura Deslactosada 🥛")
st.text("Sin gastar en Oracle SCM o SAP IBP. Por Marvin Nahmias ©2025.")

# 📌 Display Seasonality Graph
if os.path.exists(SEASONALITY_PATH):
    seasonality_df = pd.read_csv(SEASONALITY_PATH)
    fig_seasonality = go.Figure()
    fig_seasonality.add_trace(go.Scatter(x=seasonality_df["Month"], y=seasonality_df["Seasonality"], mode='lines+markers', name="Estacionalidad", line=dict(color="blue")))
    fig_seasonality.update_layout(title="Estacionalidad del Consumo de Lácteos en México", xaxis_title="Mes", yaxis_title="Índice de Consumo", template=selected_theme)
    st.plotly_chart(fig_seasonality, use_container_width=True)

# 📌 Display Graph for Temperature and Price Trends
fig_external = go.Figure()
fig_external.add_trace(go.Scatter(x=df["ds"], y=df["Temperature"], mode='lines', name="Temperatura", line=dict(color="red")))
fig_external.add_trace(go.Scatter(x=df["ds"], y=df["Price"], mode='lines', name="Precio", line=dict(color="green")))
fig_external.update_layout(title="Variación de Temperatura y Precio", xaxis_title="Fecha", template=selected_theme)
st.plotly_chart(fig_external, use_container_width=True)

# 📌 Display Prediction Graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name="Ventas Históricas", marker=dict(color='#0f4c75', size=5)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"Pronóstico ({model_choice})", line=dict(color='#3282b8', width=3)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name="Límite Superior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name="Límite Inferior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)'), fill='tonexty'))

fig.update_layout(title="📈 Pronóstico de Ventas (Pasado + Futuro Sin Cortes)", xaxis_title="Fecha", yaxis_title="Ventas (litros)", template=selected_theme, xaxis=dict(rangeslider=dict(visible=True), type="date"), yaxis=dict(fixedrange=False))

st.plotly_chart(fig, use_container_width=True)
