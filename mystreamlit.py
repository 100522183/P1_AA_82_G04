import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.title("Predicción de Suscripción Bancaria")
st.write("Introduce los datos del cliente para evaluar la probabilidad de éxito.")

def crear_nunca_contactado_eliminar_pdays(df):
    df_copy = df.copy()
    df_copy["nunca_contactado"] = (df_copy['pdays'] == -1).astype(int)
    df_copy = df_copy.drop('pdays', axis=1)
    return df_copy
# Cargar el modelo (debe incluir el preprocesamiento/pipeline)
model = joblib.load('models/modelo_final.joblib')

# Formulario de entrada
with st.form("client_data"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        job = st.selectbox("Trabajo", ["management", "blue-collar", "technician", "admin.", "services", "retired", "self-employed", "student", "unemployed", "entrepreneur", "housemaid", "unknown"])
        marital = st.selectbox("Estado civil", ["married", "single", "divorced"])
        education = st.selectbox("Educación", ["secondary", "tertiary", "primary", "unknown"])
        balance = st.number_input("Balance anual", value=1500)
        housing = st.selectbox("¿Tiene hipoteca?", ["yes", "no"])
        loan = st.selectbox("¿Tiene préstamos?", ["yes", "no"])

    with col2:
        default = st.selectbox("¿Crédito impagado?", ["no", "yes"])
        contact = st.selectbox("Método de contacto", ["cellular", "unknown", "telephone"])
        month = st.selectbox("Mes último contacto", ["may", "aug", "jul", "jun", "nov", "apr", "feb", "oct", "jan", "sep", "mar", "dec"])
        day = st.slider("Día del mes", 1, 31, 15)
        duration = st.number_input("Duración último contacto (seg)", value=150)
        campaign = st.number_input("Contactos en esta campaña", min_value=1, value=1)
        pdays = st.number_input("Días desde último contacto (-1 = nunca)", value=-1)
        previous = st.number_input("Contactos previos", min_value=0, value=0)
        poutcome = st.selectbox("Resultado campaña anterior", ["unknown", "failure", "success", "other"])

    submitted = st.form_submit_button("Analizar Cliente")

if submitted:
    # 1. Crear el diccionario con TODAS las columnas que diste (incluyendo las faltantes)
    input_dict = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }
    
    # 2. Convertir a DataFrame (el formato que espera la Pipeline)
    input_data = pd.DataFrame([input_dict])
    
    # 3. Predicción
    # La Pipeline se encarga de crear 'nunca_contactado' si incluiste el paso
    # y de aplicar el OneHotEncoding y el Scaler automáticamente.
    prediction = model.predict(input_data)
    probabilidades = model.predict_proba(input_data)[0] # [prob_no, prob_yes]
    
    st.divider()
    
    if prediction[0] == 'yes' or prediction[0] == 1:
        st.success(f"###Resultado: El cliente se suscribirá")
        st.write(f"Probabilidad de éxito: **{probabilidades[1]:.2%}**")
    else:
        st.error(f"### Resultado: El cliente no se suscribirá")
        st.write(f"Probabilidad de rechazo: **{probabilidades[0]:.2%}**")