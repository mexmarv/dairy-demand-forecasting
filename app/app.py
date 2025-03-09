import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import subprocess

# ✅ Configuración inicial (Debe ser la primera línea)
st.set_page_config(page_title="Pronóstico Demanda Leche Alpura", page_icon="🥛", layout="wide")

# ✅ Sidebar: Selección de Tema
st.sidebar.title("Configuración del Pronóstico")
st.sidebar.subheader ("por Marvin Nahmias")
theme_choice = st.sidebar.radio("Modo de Visualización", ["🌙 Oscuro", "☀️ Claro"])

# ✅ Aplicar Tema Dinámicamente
if theme_choice == "🌙 Oscuro":
    selected_theme = "plotly_dark"
    st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)
else:
    selected_theme = "plotly_white"
    st.markdown("<style>body { background-color: white; color: black; }</style>", unsafe_allow_html=True)

# ✅ Restaurar Icono de Alpura y Título
st.title("Pronóstico de la Demanda - Leche Alpura Deslactosada 🥛")

# ✅ Cargar Datos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dairy_forecast_data.csv")
SEASONALITY_PATH = os.path.join(BASE_DIR, "data", "dairy_seasonality.csv")

# 📌 Opción para subir un archivo CSV
uploaded_file = st.sidebar.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

# 📌 Si se sube un archivo, usarlo; si no, usar el dataset predeterminado
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})
    st.sidebar.success("✅ Archivo cargado correctamente.")
else:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})
    st.sidebar.warning("⚠ Usando dataset predeterminado.")

df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).rename(columns={"Sales_Volume": "y", "Date": "ds"})

# ✅ Cargar y Mostrar Datos de Estacionalidad
seasonality_df = pd.read_csv(SEASONALITY_PATH)
fig_seasonality = go.Figure()
fig_seasonality.add_trace(go.Scatter(
    x=seasonality_df["Month"],
    y=seasonality_df["Seasonality"],
    mode='lines+markers',
    name="Estacionalidad Real",
    line=dict(color="blue")
))
fig_seasonality.update_layout(
    title="Estacionalidad del Consumo de Lácteos en México",
    xaxis_title="Mes",
    yaxis_title="Índice de Consumo",
    template=selected_theme,
    width=500, height=300
)
st.plotly_chart(fig_seasonality, use_container_width=False)
# 📌 Mostrar referencia de la estacionalidad debajo de la gráfica
st.markdown("""
### 📖 **Referencia de la Estacionalidad de Leche en México**  
- **INEGI** – Reportes agropecuarios y estadísticas nacionales  
- **SIAP** – Servicio de Información Agroalimentaria y Pesquera  
- **FAO Dairy Market Review** – Reporte internacional sobre consumo de lácteos  
- **Datos analizados de supermercados y tendencias de consumo**  
📅 **Estacionalidad observada**:  
  - **Alta demanda**: Marzo–Agosto  
  - 🔹 **Picos de consumo**: Diciembre (Navidad) y Abril (Semana Santa) 🎄  
  - 🔻 **Baja demanda**: Septiembre–Noviembre ❄️  
""")

# ✅ Sidebar: Selección de Modelo y Días a Predecir
model_choice = st.sidebar.selectbox("Modelo de Pronóstico:", ["Facebook (Prophet)", "Oracle SCM (XGBoost)", "SAP IBP (SARIMA)"])
days = st.sidebar.slider("⏳ Días a predecir:", 30, 365, 90)

# ✅ Restaurar Descripción de Modelos
if model_choice == "Facebook (Prophet)":
    st.sidebar.markdown("""
    ### **Facebook (Prophet)**
    - Desarrollado por **Meta (Facebook)**
    - Detecta automáticamente tendencias y estacionalidad
    """)
elif model_choice == "SAP IBP (SARIMA)":
    st.sidebar.markdown("""
    ### **SAP IBP (SARIMA)**
    - Modelo ARIMA estacional utilizado en **SAP IBP**
    - Ajusta patrones de tendencia y estacionalidad manualmente
    """)
else:
    st.sidebar.markdown("""
    ### **Oracle SCM (XGBoost)**
    - Utiliza Machine Learning + Series Temporales
    - Simula Oracle SCM Demand Management
    """)

# ✅ Entrenar Modelo en Vivo
if st.sidebar.button("🏋️‍♂️ Entrenar Modelo"):
    st.info("🔄 Entrenando modelo...")
    TRAIN_SCRIPTS = {
        "Facebook (Prophet)": os.path.join(BASE_DIR, "src", "train_model.py"),
        "SAP IBP (SARIMA)": os.path.join(BASE_DIR, "src", "sap_ibp_forecast.py"),
        "Oracle SCM (XGBoost)": os.path.join(BASE_DIR, "src", "oracle_scm_forecast.py")
    }
    script_path = TRAIN_SCRIPTS[model_choice]

    training_logs = st.empty()
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    log_text = ""
    for line in process.stdout:
        log_text += line + "\n"
        training_logs.text_area("📜 Registro de Entrenamiento:", log_text, height=200)
    
    process.wait()
    
    if process.returncode == 0:
        st.success(f"✅ {model_choice} entrenado correctamente.")
    else:
        st.error(f"❌ Error en el entrenamiento.")

# ✅ Cargar Modelo Seleccionado y Generar Predicciones
MODEL_PATHS = {
    "Facebook (Prophet)": os.path.join(BASE_DIR, "models", "trained_model.pkl"),
    "SAP IBP (SARIMA)": os.path.join(BASE_DIR, "models", "sap_ibp_model.pkl"),
    "Oracle SCM (XGBoost)": os.path.join(BASE_DIR, "models", "oracle_scm_model.pkl")
}
model_path = MODEL_PATHS[model_choice]

with open(model_path, "rb") as f:
    model = pickle.load(f)

if model_choice == "Facebook (Prophet)":
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
elif model_choice == "SAP IBP (SARIMA)":
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    sarima_forecast = model.get_forecast(steps=days).summary_frame()
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": sarima_forecast["mean"],
        "yhat_lower": sarima_forecast["mean_ci_lower"],
        "yhat_upper": sarima_forecast["mean_ci_upper"]
    })
else:
    future_dates = pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=days, freq="D")
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

# ✅ Restaurar Intervalos de Confianza
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name="Ventas Históricas", marker=dict(color='#0f4c75', size=5)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"Pronóstico ({model_choice})", line=dict(color='#3282b8', width=3)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name="Límite Superior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name="Límite Inferior", line=dict(dash="dot", color='rgba(50, 130, 200, 0.5)'), fill='tonexty'))

# ✅ Restaurar Zoom Completo
fig.update_layout(
    title="Pronóstico de la Demanda (Pasado + Pronóstico)",
    xaxis_title="Fecha",
    yaxis_title="Demanda (litros)",
    template=selected_theme,
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
    yaxis=dict(fixedrange=False)
)

st.plotly_chart(fig, use_container_width=True)
