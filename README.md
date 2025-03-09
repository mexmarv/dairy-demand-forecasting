# Planeación de la Demanda de Leche - Dairy Demand Forecasting 🥛 

**Pronóstico de ventas de leche usando modelos avanzados de Machine Learning y Series Temporales.**  

Este proyecto permite estimar la demanda futura de productos lácteos utilizando **Facebook Prophet, SAP IBP (SARIMA) y Oracle SCM (XGBoost)**.  

**Contacto:** [mexmarv@gmail.com](mailto:mexmarv@gmail.com)  

---

## 🚀 **Características Principales**
✔ **Carga de Datos**: Permite subir un archivo CSV personalizado o usar un dataset predefinido.  
✔ **Modelos de Predicción**:  
  - **Facebook Prophet** → Modelo desarrollado por Meta, especializado en series temporales.  
  - **SAP IBP (SARIMA)** → Modelo ARIMA con estacionalidad, utilizado en planificación de la demanda.  
  - **Oracle SCM (XGBoost)** → Simulación del sistema de planificación de demanda de Oracle usando Machine Learning.  
✔ **Interfaz en Streamlit**: Aplicación interactiva con cambio de tema (Modo Claro/Oscuro).  
✔ **Visualización de Estacionalidad**: Análisis de patrones de consumo en México.  
✔ **Entrenamiento en Vivo**: Permite reentrenar los modelos directamente desde la aplicación.  

---

## **Estructura del Proyecto**
```
📦 dairy-demand-forecasting  
┣ 📂 app/  
┃ ┣ 📜 app.py              # Aplicación principal en Streamlit  
┣ 📂 data/  
┃ ┣ 📜 dairy_forecast_data.csv  # Dataset predeterminado  
┃ ┣ 📜 dairy_seasonality.csv  # Datos de estacionalidad  
┣ 📂 models/  
┃ ┣ 📜 trained_model.pkl  # Modelo Prophet entrenado  
┃ ┣ 📜 sap_ibp_model.pkl  # Modelo SARIMA entrenado  
┃ ┣ 📜 oracle_scm_model.pkl  # Modelo XGBoost entrenado  
┣ 📂 src/  
┃ ┣ 📜 train_model.py         # Entrenamiento de Prophet  
┃ ┣ 📜 sap_ibp_forecast.py    # Entrenamiento de SARIMA  
┃ ┣ 📜 oracle_scm_forecast.py # Entrenamiento de XGBoost  
┃ ┣ 📜 seasonality_analysis.py # Análisis de estacionalidad  
┃ ┣ 📜 predict.py             # Predicción de demanda  
┣ 📜 requirements.txt         # Librerías necesarias  
┣ 📜 README.md               # Documentación del proyecto  
```

---

## **Cómo Ejecutar el Proyecto**
### **1️⃣ Clonar el Repositorio**
```bash
git clone https://github.com/mexmarv/dairy-demand-forecasting.git
cd dairy-demand-forecasting
```

### **2️⃣ Crear un Entorno Virtual y Activarlo**
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### **3️⃣ Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **4️⃣ Entrenar Modelos**
Ejecuta estos comandos para entrenar cada modelo:
```bash
python src/train_model.py         # Prophet  
python src/sap_ibp_forecast.py    # SAP IBP (SARIMA)  
python src/oracle_scm_forecast.py # Oracle SCM (XGBoost)  
```

### **5️⃣ Ejecutar la Aplicación**
```bash
streamlit run app/app.py
```
**Accede a la app en tu navegador:** [`http://localhost:8501`](http://localhost:8501)  

---

## **Modelos Utilizados**
### ** Facebook Prophet**
- 📊 Modelo desarrollado por **Meta (Facebook)** para series temporales.  
- ✅ Detecta **tendencias, estacionalidad y efectos especiales**.  

### **SAP IBP (LightGBM) (Optimizado)**
- 📊 **Este modelo NO es SAP IBP oficial**, pero sigue una aproximación similar con Machine Learning.  
- 📌 **SAP IBP original** usa ARIMA, ML y datos externos (clima, ventas, precios).  
- ✅ **LightGBM se usa como una alternativa ligera** que aprende patrones temporales eficientemente.  
 

### ** Oracle SCM (XGBoost)**
- 📊 Simulación del modelo de **Demand Planning** de **Oracle SCM**.  
- ✅ Usa Machine Learning (**XGBoost**) para mejorar la predicción de la demanda.  

---

##  **Estacionalidad de la Demanda de Leche en México**
### 🔍 **Fuentes de Datos**
📌 **INEGI** – Estadísticas agropecuarias.  
📌 **SIAP** – Servicio de Información Agroalimentaria y Pesquera.  
📌 **FAO Dairy Market Review** – Reporte internacional sobre consumo de lácteos.  

### **📊 Patrones de Demanda en México**
- **Alta demanda**: **Marzo–Agosto**   
- 🔹 **Picos de consumo**: **Diciembre (Navidad) y Abril (Semana Santa)**   
- 🔻 **Baja demanda**: **Septiembre–Noviembre** ❄  

---

## ⚙**Requerimientos**
- Python 3.8+
- GitHub CLI (`gh`)
- Librerías de Python (`pip install -r requirements.txt`)

---

## **Contribuciones**
¡Si quieres mejorar este proyecto, eres bienvenido! 🚀  
 **Para contribuir:**  
1. **Fork** este repositorio.  
2. Crea una nueva rama (`git checkout -b nueva_funcionalidad`).  
3. Realiza cambios y haz un commit (`git commit -m "Agregado XYZ"`).  
4. Haz un push y crea un Pull Request (`git push origin nueva_funcionalidad`).  

---

##  **Licencia**
Este proyecto es de código abierto bajo la licencia **MIT**.  

---

## **Contacto**
**[mexmarv@gmail.com](mailto:mexmarv@gmail.com)**  
**Repositorio en GitHub:** [mexmarv/dairy-demand-forecasting](https://github.com/mexmarv/dairy-demand-forecasting)  

