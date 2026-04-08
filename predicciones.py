import pandas as pd
import numpy as np
import random
import joblib

SEMILLA = 100522258
np.random.seed(SEMILLA)
random.seed(SEMILLA)
def crear_nunca_contactado_eliminar_pdays(df):
    df_copy = df.copy()
    df_copy["nunca_contactado"] = (df_copy['pdays'] == -1).astype(int)
    # Si en tu entrenamiento NO borraste pdays, comenta la línea de abajo
    # df_copy = df_copy.drop('pdays', axis=1)
    return df_copy

# CARGAR EL MODELO
try:
    model = joblib.load('models/modelo_final.joblib')
    print("Modelo cargado correctamente.\n")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# SIMULAR 3 CLIENTES
# Creamos un diccionario con todas las columnas necesarias
data = {
    'age': [32, 55, 23],
    'job': ['management', 'retired', 'student'],
    'marital': ['married', 'divorced', 'single'],
    'education': ['tertiary', 'primary', 'secondary'],
    'default': ['no', 'no', 'no'],
    'balance': [2500, 150, 500],
    'housing': ['no', 'yes', 'no'],
    'loan': ['no', 'no', 'no'],
    'contact': ['cellular', 'telephone', 'cellular'],
    'day': [15, 5, 21],
    'month': ['aug', 'may', 'sep'],
    'duration': [450, 80, 1100], # Segundos
    'campaign': [1, 3, 1],
    'pdays': [-1, 120, -1], # El primer y tercer cliente son nuevos
    'previous': [0, 1, 0],
    'poutcome': ['unknown', 'failure', 'unknown']
}

clientes = pd.DataFrame(data)

# 4. REALIZAR PREDICCIONES
# La Pipeline se encarga de crear 'nunca_contactado', escalar y codificar
predicciones = model.predict(clientes)
probabilidades = model.predict_proba(clientes)

# 5. MOSTRAR RESULTADOS LIMPITOS
print("--- RESULTADOS DE LA PREDICCIÓN ---")
for i in range(len(clientes)):
    resultado = "SÍ" if predicciones[i] in ['yes', 1] else "NO"
    prob_exito = probabilidades[i][1] # Probabilidad de la clase 'yes'
    
    print(f"Cliente {i+1} \n({clientes.loc[[i]].to_string(index=False)}):")
    print(f"  > ¿Se suscribirá?: {resultado}")
    if resultado == "SÍ":
        print(f"  > Probabilidad de suscripción: {prob_exito:.2%}")
    else:
        prob_exito = 1-prob_exito
        print(f"  > Probabilidad de no suscripción: {prob_exito:.2%}")
    print("-" * 35)