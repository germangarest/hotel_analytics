import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predicción precio medio por noche",
    page_icon="💰",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .metric-card {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    /* Spinner personalizado */
    .prediction-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #2ecc71;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Predicción precio medio por noche")
st.markdown("""
    Esta herramienta utiliza un modelo avanzado de Machine Learning (R² = 0.75) para predecir 
    el precio medio por noche (ADR) para una reserva basándose en múltiples características.
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('src/models/adr_gbr.joblib')
        return model
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el modelo de predicción de precios. {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Formulario principal
with st.form("price_prediction_form"):
    st.subheader("📝 Detalles de la reserva")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📅 Información temporal")
        arrival_date = st.date_input(
            "Fecha de llegada prevista",
            value=datetime.now(),
            min_value=datetime(2015, 1, 1),
            max_value=datetime(2025, 12, 31),
            help="Selecciona la fecha en la que el huésped tiene previsto llegar al hotel"
        )
        
        lead_time = st.number_input(
            "Anticipación de la reserva (días)",
            min_value=0,
            value=30,
            help="¿Con cuántos días de antelación se realiza la reserva?"
        )
        
        total_nights = st.number_input(
            "Duración de la estancia (noches)",
            min_value=1,
            value=3,
            help="Número total de noches que el huésped planea alojarse"
        )
        
        is_weekend = st.checkbox(
            "¿Llegada en fin de semana?",
            value=False,
            help="Indica si la llegada es en sábado o domingo"
        )
    
    with col2:
        st.markdown("### 👥 Información de huéspedes")
        adults = st.number_input(
            "Número de adultos",
            min_value=1,
            value=2,
            help="Adultos en la reserva"
        )
        
        children = st.number_input(
            "Número de niños",
            min_value=0,
            value=0,
            help="Niños entre 3 y 12 años"
        )
        
        babies = st.number_input(
            "Número de bebés",
            min_value=0,
            value=0,
            help="Niños menores de 3 años"
        )
        
        total_of_special_requests = st.number_input(
            "Solicitudes especiales",
            min_value=0,
            value=0,
            help="Número de peticiones especiales realizadas por el cliente"
        )
    
    with col3:
        st.markdown("### 🏨 Detalles del Alojamiento")
        meal = st.selectbox(
            "Régimen de comidas",
            options=['BB', 'FB', 'HB', 'SC'],
            index=0,
            help="""
            BB: Bed & Breakfast (Alojamiento y desayuno)
            FB: Full Board (Pensión completa)
            HB: Half Board (Media pensión)
            SC: Self Catering (Sin comidas)
            """
        )
        
        market_segment = st.selectbox(
            "Canal de reserva",
            options=['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Aviation'],
            index=0,
            help = """
- Direct: Reservas hechas directamente con el hotel (sitio web, teléfono o recepción).
- Corporate: Reservas de empresas para empleados o eventos, con tarifas negociadas.
- Online TA: Reservas a través de agencias de viaje en línea (Booking, Expedia).
- Offline TA/TO: Reservas mediante agencias de viaje o touroperadores tradicionales.
- Groups: Reservas para grupos grandes (tours, bodas, eventos).
- Aviation: Reservas ligadas a aerolíneas (tripulaciones o pasajeros en tránsito).
"""

        )
        
        deposit_type = st.selectbox(
            "Tipo de depósito",
            options=['No Deposit', 'Refundable', 'Non Refund'],
            index=0,
            help="""
            No Deposit: Depósito no reembolsable.
            Refundable: Depósito reembolsable.
            Non Refund: Sin depósito.
            """
        )
        
        reserved_room_type = st.selectbox(
            "Tipo de habitación",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            index=0,
            help="""
            Categoría de la habitación reservada:
            A: Estándar básica.
            B: Estándar con mejores vistas.
            C: Habitación superior.
            D: Habitación deluxe.
            E: Suite junior.
            F: Suite premium.
            G: Suite ejecutiva.
            H: Suite presidencial.
            """
        )

    predict_button = st.form_submit_button("💡 Calcular precio estimado")

if predict_button:
    # Mostrar spinner personalizado
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("""
        <div class="prediction-spinner"></div>
        <p style='text-align: center; color: #666;'>Calculando predicción...</p>
    """, unsafe_allow_html=True)
    
    # Calcular predicción
    # Calcular noches de fin de semana y entre semana
    weekend_ratio = 0.3 if not is_weekend else 0.4
    stays_in_weekend_nights = int(total_nights * weekend_ratio)
    stays_in_week_nights = total_nights - stays_in_weekend_nights
    
    # Calcular características derivadas
    total_guests = adults + children + babies
    avg_guests_per_night = total_guests * total_nights
    booking_flexibility = 1 if deposit_type == 'No Deposit' else 0
    
    # Preparar los datos para la predicción
    input_dict = {
        # Características temporales
        'lead_time': lead_time,
        'arrival_date_year': arrival_date.year,
        'arrival_date_month': arrival_date.strftime('%B'),
        'arrival_date_day_of_month': arrival_date.day,
        'stays_in_weekend_nights': stays_in_weekend_nights,
        'stays_in_week_nights': stays_in_week_nights,
        
        # Información de huéspedes
        'adults': adults,
        'children': children,
        'babies': babies,
        'total_guests': total_guests,
        
        # Información de la reserva
        'meal': meal,
        'market_segment': market_segment,
        'deposit_type': deposit_type,
        'reserved_room_type': reserved_room_type,
        'total_of_special_requests': total_of_special_requests,
        
        # Características derivadas
        'is_weekend_arrival': int(is_weekend),  # Convertir a int como en el modelo
        'total_nights': total_nights,
        'avg_guests_per_night': avg_guests_per_night,
        'booking_flexibility': booking_flexibility
    }
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Realizar la predicción
    predicted_price = model.predict(input_df)[0]
    
    # Ajustar valores atípicos como en el entrenamiento
    if predicted_price < 0:
        predicted_price = 0
    
    # Después de la predicción, eliminar el spinner
    spinner_placeholder.empty()
    
    # Mostrar resultados
    st.write("---")
    st.subheader("💰 Precio estimado")
    
    results_container = st.container()
    with results_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Mostrar precio estimado
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center;'>Precio Medio por Noche</h3>
                    <h2 style='text-align: center; font-size: 2.5em;'>
                        {predicted_price:.2f}€
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Mostrar precio total
            total_price = predicted_price * total_nights
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center;'>Precio Total Estimado</h3>
                    <h2 style='text-align: center; font-size: 2em;'>
                        {total_price:.2f}€
                    </h2>
                    <p style='text-align: center;'>Para {total_nights} noches</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Factores que influyen en el precio
            st.markdown("### 📊 Factores que influyen en el precio")
            
            price_factors = []
            
            if is_weekend:
                price_factors.append("📅 Llegada en fin de semana")
            
            if total_nights > 7:
                price_factors.append("📏 Estancia larga")
            
            if total_guests > 2:
                price_factors.append("👥 Grupo grande")
            
            if meal != 'BB':
                price_factors.append("🍽️ Régimen de comidas superior")
            
            if deposit_type == 'Non Refund':
                price_factors.append("💰 Tarifa no reembolsable")
            
            if total_of_special_requests > 0:
                price_factors.append("✨ Solicitudes especiales")
            
            for factor in price_factors:
                st.markdown(f"- {factor}")
            
            # Recomendaciones
            st.markdown("### 💡 Recomendaciones")
            
            recommendations = []
            
            if is_weekend:
                recommendations.append("- Considerar fechas entre semana para mejores tarifas")
            
            if total_nights >= 7:
                recommendations.append("- Preguntar por descuentos para estancias largas")
            
            if meal != 'BB':
                recommendations.append("- Comparar el costo-beneficio de diferentes regímenes de comidas")
            
            if deposit_type == 'No Deposit':
                recommendations.append("- Valorar tarifas no reembolsables para obtener mejor precio")
            
            st.info("\n".join(recommendations) if recommendations else "No hay recomendaciones específicas para esta reserva.")
            
            # Mostrar precisión del modelo
            st.markdown("### 🎯 Precisión del modelo")
            st.progress(0.75)
            st.caption("El modelo tiene un R² score de 0.75 en la predicción de precios")
