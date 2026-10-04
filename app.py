import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="centered"
)

# 2. Función para cargar el modelo (con caché para que sea rápido)
@st.cache_resource
def cargar_modelo():
    # Verificamos si el archivo existe en la misma carpeta
    ruta_modelo = 'logistic_regression_model.pkl'
    if not os.path.exists(ruta_modelo):
        return None
    # Cargamos el modelo usando joblib
    modelo = joblib.load(ruta_modelo)
    return modelo

# 3. Intentar cargar el modelo
modelo = cargar_modelo()

if modelo is None:
    st.error("⚠️ Error crítico: No se encontró el archivo 'logistic_regression_model.pkl' en la carpeta del proyecto.")
    st.stop() # Detiene la ejecución si no hay modelo
else:
    st.success("✅ Modelo de Machine Learning cargado correctamente.")

# 4. Título y descripción de la App
st.title("🩺 Predictor de Diabetes")
st.markdown("""
Esta aplicación utiliza un modelo de **Regresión Logística** para predecir 
la probabilidad de que un paciente tenga diabetes basándose en 8 variables clínicas.
Por favor, ingresa los datos del paciente en el formulario inferior.
""")
st.markdown("---")

# 5. Formulario de entrada de datos
with st.form("formulario_prediccion"):
    st.subheader("📋 Datos Clínicos del Paciente")
    
    # Dividir en dos columnas para que se vea ordenado
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Embarazos (Pregnancies)", min_value=0, max_value=20, value=1, step=1, help="Número de embarazos")
        glucose = st.number_input("Glucosa (Glucose)", min_value=0, max_value=300, value=120, step=1, help="Concentración de glucosa en plasma a las 2 horas")
        blood_pressure = st.number_input("Presión Arterial (BloodPressure)", min_value=0, max_value=200, value=70, step=1, help="Presión arterial diastólica (mm Hg)")
        skin_thickness = st.number_input("Grosor de Piel (SkinThickness)", min_value=0, max_value=100, value=20, step=1, help="Grosor del pliegue cutáneo del tríceps (mm)")
        
    with col2:
        insulin = st.number_input("Insulina (Insulin)", min_value=0, max_value=900, value=80, step=1, help="Insulina sérica a las 2 horas (mu U/ml)")
        bmi = st.number_input("IMC (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f", help="Índice de Masa Corporal (peso en kg/(altura en m)^2)")
        diabetes_pedigree = st.number_input("Función Pedigrí (DiabetesPedigreeFunction)", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f", help="Función que estima la probabilidad genética de diabetes")
        age = st.number_input("Edad (Age)", min_value=1, max_value=120, value=30, step=1, help="Edad del paciente en años")
    
    st.markdown("---")
    # Botón de envío
    boton_predecir = st.form_submit_button("🔍 Realizar Predicción", use_container_width=True)

# 6. Lógica de Predicción (se ejecuta al presionar el botón)
if boton_predecir:
    # Crear un DataFrame con los datos ingresados
    # IMPORTANTE: El orden de las columnas debe ser el mismo que usaste para entrenar el modelo
    datos_paciente = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Realizar la predicción
    try:
        prediccion = modelo.predict(datos_paciente)
        probabilidad = modelo.predict_proba(datos_paciente)[0][1] # Probabilidad de clase 1 (Diabetes)
        
        # Mostrar resultados visualmente atractivos
        st.markdown("---")
        st.subheader("📊 Resultado del Análisis")
        
        if prediccion[0] == 1:
            st.error(f"### ⚠️ Resultado: POSITIVO para Diabetes")
            st.metric(label="Probabilidad estimada", value=f"{probabilidad*100:.2f}%")
            st.warning("Se recomienda consultar a un médico especialista para un diagnóstico confirmatorio.")
        else:
            st.success(f"### ✅ Resultado: NEGATIVO para Diabetes")
            st.metric(label="Probabilidad estimada", value=f"{probabilidad*100:.2f}%")
            st.info("Los indicadores se encuentran dentro de los rangos normales según el modelo.")
            
        # Mostrar los datos ingresados (opcional, para transparencia)
        with st.expander("Ver datos ingresados"):
            st.dataframe(datos_paciente)
            
    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")
        st.write("Asegúrate de que el modelo haya sido entrenado con exactamente estas 8 columnas en este orden.")

# 7. Pie de página
st.markdown("---")
st.caption("Desarrollado para el curso de Fundamentos de Machine Learning | Desplegado en Streamlit")
