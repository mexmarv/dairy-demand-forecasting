import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import subprocess
import sys  # ✅ IMPORTAR sys para evitar el NameError

# ✅ Configuración inicial (Debe ser la primera línea)
st.set_page_config(page_title="Pronóstico Demanda de Leche", page_icon="🥛", layout="wide")

# ✅ Sidebar: Selección de Tema
st.sidebar.title("Configuración del Pronóstico")
st.sidebar.subheader("by Marvin Nahmias ©2025")
theme_choice = st.sidebar.radio("Modo de Visualización", ["🌙 Oscuro", "☀️ Claro"])

# ✅ Aplicar Tema Dinámicamente
selected_theme = "plotly_dark" if theme_choice == "🌙 Oscuro" else "plotly_white"
st.markdown(f"<style>body {{ background-color: {'#1e1e1e' if theme_choice == '🌙 Oscuro' else 'white'}; color: {'white' if theme_choice == '🌙 Oscuro' else 'black'}; }}</style>", unsafe_allow_html=True)

# ✅ Restaurar Icono de Alpura y Título
st.title("Pronóstico de Ventas - Leche Deslactosada en México 🥛")
st.subheader("Sin gastar en SAP IBP u Oracle SCM :) ")

# 📌 Opción para subir un archivo CSV o usar el predeterminado
uploaded_file = st.sidebar.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
DATA_PATH = "data/dairy_forecast_data.csv"
SEASONALITY_PATH = "data/dairy_seasonality.csv"

df = pd.read_csv(uploaded_file if uploaded_file else DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})
st.sidebar.success("✅ Archivo cargado correctamente." if uploaded_file else "⚠ Usando dataset predeterminado.")

# 📌 Mostrar Estacionalidad
seasonality_df = pd.read_csv(SEASONALITY_PATH)
fig_seasonality = go.Figure()
fig_seasonality.add_trace(go.Scatter(x=seasonality_df["Month"], y=seasonality_df["Seasonality"], mode='lines+markers', name="Estacionalidad Real", line=dict(color="blue")))
fig_seasonality.update_layout(title="📅 Estacionalidad del Consumo de Lácteos en México", xaxis_title="Mes", yaxis_title="Índice de Consumo", template=selected_theme, width=500, height=300)
st.plotly_chart(fig_seasonality, use_container_width=False)

# 📌 Sidebar: Selección de Modelo
model_choice = st.sidebar.selectbox("Modelo de Pronóstico:", ["Facebook Prophet", "SAP IBP (LightGBM)", "Oracle SCM (XGBoost)"])
days = st.sidebar.slider("Días a predecir:", 30, 365, 90)

# 📌 Entrenar Modelos con `sys.executable` para evitar errores en Streamlit Cloud
if st.sidebar.button("Entrenar Modelo"):
    st.info("🔄 Instalando dependencias y entrenando modelo...")

    # ✅ Instalar dependencias en el entorno de Streamlit Cloud antes de entrenar
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])

    # ✅ Ejecutar el script de entrenamiento
    TRAIN_SCRIPTS = {
        "Facebook Prophet": "src/train_model.py",
        "SAP IBP (LightGBM)": "src/sap_ibp_forecast.py",
        "Oracle SCM (XGBoost)": "src/oracle_scm_forecast.py"
    }
    script_path = TRAIN_SCRIPTS[model_choice]

    training_logs = st.empty()
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    log_text = ""
    for line in process.stdout:
        log_text += line + "\n"
        training_logs.text_area("📜 Registro de Entrenamiento:", log_text, height=200)

    process.wait()

    if process.returncode == 0:
        st.success(f"✅ {model_choice} entrenado correctamente.")
    else:
        st.error(f"❌ Error en el entrenamiento. Revisa los logs.")


# 📌 Cargar Modelo
MODEL_PATHS = {"Facebook Prophet": "models/trained_model.pkl", "SAP IBP (LightGBM)": "models/sap_ibp_model.pkl", "Oracle SCM (XGBoost)": "models/oracle_scm_model.pkl"}
model_path = MODEL_PATHS[model_choice]
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 📌 Generar Predicciones
if model_choice == "Facebook Prophet":
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
elif model_choice == "SAP IBP (LightGBM)":
    future_dates = pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=days, freq="D")

    # ✅ Asegurar que las features sean las mismas que en el entrenamiento
    future_features = pd.DataFrame({
        "year": future_dates.year,
        "month": future_dates.month,
        "day": future_dates.day,
        "day_of_week": future_dates.dayofweek
    })

    forecast_values = model.predict(future_features)

    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast_values,
        "yhat_lower": forecast_values - 10,  # Estimación simple
        "yhat_upper": forecast_values + 10
    })
else:
    future_dates = pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    future_features = pd.DataFrame({"year": future_dates.year, "month": future_dates.month, "day": future_dates.day, "day_of_week": future_dates.dayofweek})
    forecast_values = model.predict(future_features)
    forecast = pd.DataFrame({"ds": future_dates, "yhat": forecast_values, "yhat_lower": forecast_values - 10, "yhat_upper": forecast_values + 10})

# 📌 Restaurar Gráfica Completa con Intervalos de Confianza y Zoom
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name="Ventas Históricas", marker=dict(color='#0f4c75', size=5)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"Pronóstico ({model_choice})", line=dict(color='#3282b8', width=3)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name="Límite Superior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name="Límite Inferior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)'), fill='tonexty'))

fig.update_layout(
    title="📈 Pronóstico de Ventas (Pasado + Futuro Sin Cortes)",
    xaxis_title="Fecha",
    yaxis_title="Ventas (litros)",
    template=selected_theme,
    xaxis=dict(rangeslider=dict(visible=True), type="date"),  # ✅ Restaurado Zoom
    yaxis=dict(fixedrange=False)
)

st.plotly_chart(fig, use_container_width=True)
