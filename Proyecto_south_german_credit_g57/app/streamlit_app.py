import streamlit as st
import requests
import json
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Riesgo Crediticio",
    page_icon="",
    layout="wide"
)

# --- URL del Backend (FastAPI) ---
API_URL = "http://127.0.0.1:8000/predict" 

# =============================================================================
# DICCIONARIOS DE TRADUCCIÓN
# =============================================================================
# Mapas para traducir las opciones del usuario a los valores float que la API espera.

status_map = {
    "Sin cuenta corriente": 1.0,
    "Cuenta con saldo bajo (< 0)": 2.0,
    "Cuenta con saldo medio (0-200 DM)": 3.0,
    "Cuenta con saldo alto (> 200 DM)": 4.0
}

history_map = {
    "Crédito pagado debidamente hasta ahora": 0.0,
    "Créditos pagados debidamente en este banco": 1.0,
    "Retrasos en pagos en el pasado": 2.0,
    "Cuenta crítica / otros créditos existentes": 3.0,
    "Todos los créditos en este banco pagados": 4.0
}

purpose_map = {
    "Carro (nuevo)": 0.0,
    "Carro (usado)": 1.0,
    "Muebles/Equipo": 2.0,
    "Radio/TV": 3.0,
    "Electrodomésticos": 4.0,
    "Reparaciones": 5.0,
    "Educación": 6.0,
    "Vacaciones": 7.0,
    "Capacitación": 8.0,
    "Negocios": 9.0,
    "Otros": 10.0
}

savings_map = {
    "Desconocido / Sin ahorros": 1.0,
    "< 100 DM": 2.0,
    "100 - 500 DM": 3.0,
    "500 - 1000 DM": 4.0,
    "> 1000 DM": 5.0
}

employment_map = {
    "Desempleado": 1.0,
    "< 1 año": 2.0,
    "1 - 4 años": 3.0,
    "4 - 7 años": 4.0,
    "> 7 años": 5.0
}

status_sex_map = {
    "Hombre: Divorciado/Separado": 1.0,
    "Mujer: Divorciada/Separada/Casada": 2.0,
    "Hombre: Soltero": 3.0,
    "Hombre: Casado/Viudo": 4.0,
    "Mujer: Soltera": 5.0
}

debtors_map = {
    "Ninguno": 1.0,
    "Co-solicitante": 2.0,
    "Garante": 3.0
}

property_map = {
    "Bienes raíces (Casa/Terreno)": 1.0,
    "Seguro de vida / Ahorros": 2.0,
    "Carro u otro": 3.0,
    "Desconocido / Sin propiedad": 4.0
}

plans_map = {
    "Bancarios": 1.0,
    "Tiendas": 2.0,
    "Ninguno": 3.0
}

housing_map = {
    "Alquiler": 1.0,
    "Vivienda propia": 2.0,
    "Gratis": 3.0
}

job_map = {
    "No calificado - no residente": 1.0,
    "No calificado - residente": 2.0,
    "Calificado / Obrero": 3.0,
    "Gerente / Ejecutivo / Autónomo": 4.0
}

telephone_map = {"No": 1.0, "Sí (registrado)": 2.0}
foreign_map = {"Sí": 1.0, "No": 2.0}

# =============================================================================
# INTERFAZ DE USUARIO (STREAMLIT)
# =============================================================================

# --- Título y Encabezado ---
st.title("Simulador de Riesgo Crediticio (G57)")
st.write("""
Esta aplicación consume un microservicio de Machine Learning para predecir si un 
solicitante de crédito es de **Alto Riesgo** (Clase 1) o **Bajo Riesgo** (Clase 0).
Complete el formulario para obtener una predicción.
""")

# --- Sidebar con Instrucciones ---
image_path = os.path.join(os.path.dirname(__file__), "a57.png")
with st.sidebar:
    st.header("Instrucciones")
    st.write("""
    1.  Complete los 20 campos del formulario.
    2.  Use los menús desplegables y sliders para valores categóricos y numéricos.
    3.  Los valores por defecto son del caso de prueba exitoso.
    4.  Haga clic en 'Predecir Riesgo' para ver el resultado.
    """)    
    st.image(image_path, caption="Su Socio Financiero")

# --- Formulario Organizado en Columnas ---
st.subheader("Formulario del Solicitante")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Detalles del Crédito")
    # --- duration ---
    duration_val = st.slider(
        "Duración del Crédito (Meses)", 
        min_value=4, max_value=72, value=12,
        help="Duración total del préstamo en meses."
    )
    # --- amount ---
    amount_val = st.number_input(
        "Monto del Crédito (DM)", 
        min_value=0.0, value=1934.0, step=100.0, format="%.2f",
        help="Monto total del crédito solicitado."
    )
    # --- purpose ---
    purpose_label = st.selectbox(
        "Propósito del Crédito", 
        options=list(purpose_map.keys()), 
        index=3 # Corresponde a 'Radio/TV' (3.0)
    )
    # --- installment_rate ---
    installment_val = st.slider(
        "Tasa de Pago (% de Ingreso)", 
        min_value=1, max_value=4, value=2,
        help="Porcentaje del ingreso disponible destinado al pago."
    )
    # --- other_installment_plans ---
    plans_label = st.selectbox(
        "Otros Planes de Pago", 
        options=list(plans_map.keys()), 
        index=2 # Corresponde a 'Ninguno' (3.0)
    )

with col2:
    st.markdown("##### Perfil Personal")
    # --- age ---
    age_val = st.slider(
        "Edad", 
        min_value=18, max_value=75, value=26,
        help="Edad del solicitante."
    )
    # --- personal_status_sex ---
    status_sex_label = st.selectbox(
        "Estado Personal y Sexo", 
        options=list(status_sex_map.keys()), 
        index=2 # Corresponde a 'Hombre: Soltero' (3.0)
    )
    # --- job ---
    job_label = st.selectbox(
        "Nivel de Empleo/Cargo", 
        options=list(job_map.keys()), 
        index=2 # Corresponde a 'Calificado / Obrero' (3.0)
    )
    # --- people_liable ---
    people_val = st.slider(
        "Personas a Mantener", 
        min_value=1, max_value=4, value=2
    )
    # --- telephone ---
    tel_label = st.radio(
        "Tiene Teléfono Registrado", 
        options=list(telephone_map.keys()), 
        index=0 # Corresponde a 'No' (1.0)
    )
    # --- foreign_worker ---
    foreign_label = st.radio(
        "Es Trabajador Extranjero", 
        options=list(foreign_map.keys()), 
        index=1 # Corresponde a 'No' (2.0)
    )

with col3:
    st.markdown("##### Historial y Patrimonio")
    # --- status ---
    status_label = st.selectbox(
        "Estado de la Cuenta Corriente", 
        options=list(status_map.keys()), 
        index=3 # Corresponde a 'Alta (> 200 DM)' (4.0)
    )
    # --- credit_history ---
    history_label = st.selectbox(
        "Historial Crediticio", 
        options=list(history_map.keys()), 
        index=4 # Corresponde a 'Todos los créditos pagados' (4.0)
    )
    # --- savings ---
    savings_label = st.selectbox(
        "Cuenta de Ahorros", 
        options=list(savings_map.keys()), 
        index=0 # Corresponde a 'Sin ahorros' (1.0)
    )
    # --- employment_duration ---
    employment_label = st.selectbox(
        "Tiempo Empleado", 
        options=list(employment_map.keys()), 
        index=4 # Corresponde a '> 7 años' (5.0)
    )
    # --- other_debtors ---
    debtors_label = st.selectbox(
        "Otros Deudores / Garantes", 
        options=list(debtors_map.keys()), 
        index=0 # Corresponde a 'Ninguno' (1.0)
    )
    # --- present_residence ---
    residence_val = st.slider(
        "Tiempo en Residencia Actual (años)", 
        min_value=1, max_value=4, value=2
    )
    # --- property ---
    property_label = st.selectbox(
        "Propiedad Principal", 
        options=list(property_map.keys()), 
        index=3 # Corresponde a 'Sin propiedad' (4.0)
    )
    # --- housing ---
    housing_label = st.selectbox(
        "Tipo de Vivienda", 
        options=list(housing_map.keys()), 
        index=1 # Corresponde a 'Vivienda propia' (2.0)
    )
    # --- number_credits ---
    credits_val = st.slider(
        "Créditos en este banco", 
        min_value=1, max_value=4, value=2
    )

# --- Botón de Predicción ---
st.divider()

if st.button("Predecir Riesgo", type="primary", use_container_width=True):
    
    # 1. TRADUCIR y Recolectar datos del formulario
    payload = {
        # Columna 1
        "duration": float(duration_val),
        "amount": float(amount_val),
        "purpose": purpose_map[purpose_label],
        "installment_rate": float(installment_val),
        "other_installment_plans": plans_map[plans_label],
        
        # Columna 2
        "age": float(age_val),
        "personal_status_sex": status_sex_map[status_sex_label],
        "job": job_map[job_label],
        "people_liable": float(people_val),
        "telephone": telephone_map[tel_label],
        "foreign_worker": foreign_map[foreign_label],

        # Columna 3
        "status": status_map[status_label],
        "credit_history": history_map[history_label],
        "savings": savings_map[savings_label],
        "employment_duration": employment_map[employment_label],
        "other_debtors": debtors_map[debtors_label],
        "present_residence": float(residence_val),
        "property": property_map[property_label],
        "housing": housing_map[housing_label],
        "number_credits": float(credits_val)
    }

    # 2. Enviar la petición POST al backend (FastAPI)
    try:
        with st.spinner("Realizando predicción..."):
            response = requests.post(API_URL, json=payload)
        
        # 3. Procesar la respuesta
        if response.status_code == 200:
            result = response.json()
            
            # Leer las claves en español devueltas por la API
            label = result['etiqueta_prediccion']
            prob = result['probabilidad']

            if label == "Riesgo Alto":
                st.error(f"**Predicción: {label}**", icon="🚨")
                st.write(f"El solicitante tiene una probabilidad de **{prob:.2%}** de ser un riesgo alto (Clase 1).")
                st.progress(prob, text=f"{prob:.2%} de confianza")
            else:
                st.success(f"**Predicción: {label}**", icon="✅")
                st.write(f"El solicitante tiene una probabilidad de **{prob:.2%}** de ser un riesgo bajo (Clase 0).")
                st.progress(prob, text=f"{prob:.2%} de confianza")

            # Mostrar métrica
            st.metric(label=f"Probabilidad de ser {label}", value=f"{prob:.2%}")
            
            # Mostrar el JSON de respuesta (para depuración)
            with st.expander("Ver respuesta JSON de la API"):
                st.json(result)
            with st.expander("Ver JSON enviado a la API (Payload)"):
                st.json(payload)

        else:
            # Mostrar error si la API falla
            st.error(f"Error de la API (Código: {response.status_code}): {response.text}")
            with st.expander("Ver JSON enviado a la API (Payload)"):
                st.json(payload)

    except requests.exceptions.ConnectionError:
        st.error("Error de Conexión: No se pudo conectar a la API de FastAPI.")
        st.write("Asegúrate de que el contenedor de Docker esté corriendo en `http://127.0.0.1:8000`.")
    
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")