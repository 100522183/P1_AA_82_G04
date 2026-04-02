import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.title("Predicción de Suscripción Bancaria")
st.write("Introduce los datos del cliente para evaluar la probabilidad de éxito.")

# Cargar el modelo (debe incluir el preprocesamiento/pipeline)
model = joblib.load('models/modelo_final.joblib')

# Formulario de entrada
with st.form("client_data"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        job = st.selectbox("Trabajo", ["admin.", "blue-collar", "technician", "services", "management", "etc"])
        balance = st.number_input("Balance anual", value=1000)
        
    with col2:
        housing = st.selectbox("¿Tiene hipoteca?", ["yes", "no"])
        duration = st.number_input("Duración último contacto (seg)", value=150)
        pdays = st.number_input("Días desde último contacto (-1 si nunca)", value=-1)

    submitted = st.form_submit_button("Predecir")

if submitted:
    # Crear DataFrame con los mismos nombres de columna que el original
    input_data = pd.DataFrame([{
        'age': age, 'job': job, 'balance': balance, 
        'housing': housing, 'duration': duration, 'pdays': pdays,
        # Añade aquí todas las variables que pida tu modelo
    }])
    
    # Procesar pdays_never si lo incluiste en el entrenamiento
    input_data['pdays_never'] = (input_data['pdays'] == -1).astype(int)
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)
    
    if prediction[0] == 'yes':
        st.success(f"El cliente PROBABLEMENTE se suscribirá (Probabilidad: {prob[0][1]:.2f})")
    else:
        st.error(f"El cliente NO se suscribirá (Probabilidad: {prob[0][0]:.2f})")