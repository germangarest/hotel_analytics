import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predicción de cancelaciones",
    page_icon="❌",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #e74c3c;
    }
    .metric-card {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    /* Spinner personalizado */
    .prediction-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #e74c3c;
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

st.title("❌ Predicción de cancelaciones hoteleras")
st.markdown("""
    Esta herramienta utiliza un modelo avanzado de Machine Learning (Precisión: 84.35%) para predecir 
    la probabilidad de que una reserva sea cancelada, basándose en múltiples factores.
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('src/models/cancelacion_model.joblib')
        return model
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el modelo de predicción de cancelaciones. {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Formulario principal
with st.form("cancellation_prediction_form"):
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
        
        high_season = st.checkbox(
            "¿Temporada alta?",
            value=arrival_date.strftime('%B') in ['July', 'August', 'December'],
            help="Julio, agosto y diciembre se consideran temporada alta"
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

        # Sección expandible para información del cliente
        with st.expander("👤 Información detallada del cliente (opcional)", expanded=False):
            is_repeated_guest = st.checkbox(
                "¿Cliente repetidor?",
                value=False,
                help="Indica si el cliente ha estado antes en el hotel"
            )
            
            previous_cancellations = st.number_input(
                "Cancelaciones previas",
                min_value=0,
                value=0,
                help="Número de reservas anteriores que el cliente ha cancelado"
            )
            
            previous_bookings = st.number_input(
                "Reservas previas completadas",
                min_value=0,
                value=0,
                help="Número de reservas anteriores completadas por el cliente"
            )
            
            booking_changes = st.number_input(
                "Cambios en la reserva",
                min_value=0,
                value=0,
                help="Número de modificaciones realizadas a la reserva"
            )
    
    with col3:
        st.markdown("### 🏨 Detalles del alojamiento")
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

        # Sección expandible para detalles adicionales
        with st.expander("🔍 Detalles adicionales (opcional)", expanded=False):
            customer_type = st.selectbox(
                "Tipo de cliente",
                options=['Transient', 'Contract', 'Group', 'Transient-Party'],
                index=0,
                help="""
                Transient: Huéspedes individuales, no parte de grupos ni contratos.
                Contract: Huéspedes bajo acuerdos corporativos o contratos a largo plazo.
                Group: Huéspedes que forman parte de un grupo organizado (tours, eventos).
                Transient-Party: Huéspedes individuales que forman parte de una pequeña fiesta o evento social.
                """
            )
            
            required_car_parking_spaces = st.number_input(
                "Plazas de parking requeridas",
                min_value=0,
                value=0,
                help="Número de plazas de aparcamiento solicitadas"
            )
            
            total_of_special_requests = st.number_input(
                "Solicitudes especiales",
                min_value=0,
                value=0,
                help="Número de peticiones especiales realizadas por el cliente"
            )
            
            days_in_waiting_list = st.number_input(
                "Días en lista de espera",
                min_value=0,
                value=0,
                help="Días que la reserva estuvo en lista de espera"
            )
        
        adr = st.number_input(
            "Tarifa diaria (€)",
            min_value=0.0,
            value=100.0,
            help="Precio medio por noche"
        )

    predict_button = st.form_submit_button("🔍 Analizar riesgo de cancelación")

if predict_button:
    # Mostrar spinner personalizado
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("""
        <div class="prediction-spinner"></div>
        <p style='text-align: center; color: #666;'>Analizando riesgo de cancelación...</p>
    """, unsafe_allow_html=True)
    
    # Calcular características derivadas
    total_guests = adults + children + babies
    stays_in_weekend_nights = int(total_nights * (0.4 if is_weekend else 0.3))
    stays_in_week_nights = total_nights - stays_in_weekend_nights
    avg_guests_per_night = total_guests * total_nights
    booking_flexibility = 1 if deposit_type == 'No Deposit' else 0
    price_per_night = adr
    total_cost = adr * total_nights
    repeated_guest_value = previous_cancellations if is_repeated_guest else 0
    cancellation_risk = previous_cancellations / (previous_bookings + 1)
    lead_time_category = 'medium'  # Simplificado para la interfaz
    
    # Preparar datos para la predicción
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
        'customer_type': customer_type,
        'adr': adr,
        
        # Características opcionales (ahora incluidas correctamente)
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests,
        'booking_changes': booking_changes,
        'days_in_waiting_list': days_in_waiting_list,
        
        # Información del cliente
        'previous_cancellations': previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings,
        'is_repeated_guest': int(is_repeated_guest),
        
        # Características derivadas
        'is_weekend_arrival': int(is_weekend),
        'total_nights': total_nights,
        'avg_guests_per_night': avg_guests_per_night,
        'booking_flexibility': booking_flexibility,
        'high_season': int(high_season),
        'lead_time_category': lead_time_category,
        'price_per_night': price_per_night,
        'total_cost': total_cost,
        'repeated_guest_value': repeated_guest_value,
        'cancellation_risk': cancellation_risk
    }
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Realizar predicción
    cancellation_prob = model.predict_proba(input_df)[0][1]
    
    # Después de la predicción, eliminar el spinner
    spinner_placeholder.empty()
    
    # Mostrar resultados
    st.write("---")
    st.subheader("🎯 Análisis de riesgo")
    
    results_container = st.container()
    with results_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Mostrar probabilidad de cancelación
            risk_color = (
                "risk-high" if cancellation_prob > 0.7
                else "risk-medium" if cancellation_prob > 0.3
                else "risk-low"
            )
            
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center;'>Probabilidad de cancelación</h3>
                    <h2 style='text-align: center; font-size: 2.5em;' class='{risk_color}'>
                        {cancellation_prob:.1%}
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Nivel de riesgo
            risk_level = (
                "ALTO" if cancellation_prob > 0.7
                else "MEDIO" if cancellation_prob > 0.3
                else "BAJO"
            )
            
            risk_description = {
                "ALTO": "Esta reserva tiene un alto riesgo de cancelación. Se recomienda tomar medidas preventivas.",
                "MEDIO": "Esta reserva tiene un riesgo moderado de cancelación. Se sugiere seguimiento.",
                "BAJO": "Esta reserva tiene un riesgo bajo de cancelación. Proceder normalmente."
            }
            
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center;'>Nivel de riesgo</h3>
                    <h2 style='text-align: center; font-size: 2em;' class='risk-{risk_level.lower()}'>
                        {risk_level}
                    </h2>
                    <p style='text-align: center;'>{risk_description[risk_level]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Factores de riesgo
            st.markdown("### 📊 Factores de riesgo")
            
            risk_factors = []
            
            if lead_time > 60:
                risk_factors.append("⚠️ Reserva realizada con mucha antelación")
            
            if high_season:
                risk_factors.append("⚠️ Reserva en temporada alta")
            
            if deposit_type == 'No Deposit':
                risk_factors.append("⚠️ Sin depósito")
            
            if previous_cancellations > 0:
                risk_factors.append(f"⚠️ Cliente con {previous_cancellations} cancelaciones previas")
            
            if total_cost > 500:
                risk_factors.append("⚠️ Reserva de alto valor")
            
            if total_nights > 7:
                risk_factors.append("⚠️ Estancia larga")
            
            if not risk_factors:
                risk_factors.append("✅ No se detectan factores de riesgo significativos")
            
            for factor in risk_factors:
                st.markdown(f"- {factor}")
            
            # Recomendaciones
            st.markdown("### 💡 Recomendaciones")
            
            recommendations = []
            
            if cancellation_prob > 0.7:
                recommendations.extend([
                    "- Solicitar un depósito de garantía",
                    "- Contactar al cliente para confirmar la reserva",
                    "- Considerar overbooking controlado",
                    "- Preparar plan de contingencia"
                ])
            elif cancellation_prob > 0.3:
                recommendations.extend([
                    "- Hacer seguimiento periódico de la reserva",
                    "- Enviar recordatorios amigables",
                    "- Ofrecer servicios adicionales para aumentar el compromiso"
                ])
            else:
                recommendations.extend([
                    "- Proceder con la gestión normal de la reserva",
                    "- Mantener la comunicación estándar con el cliente"
                ])
            
            if deposit_type == 'No Deposit':
                recommendations.append("- Sugerir tarifa con depósito a cambio de descuento")
            
            if lead_time > 60:
                recommendations.append("- Programar recordatorios periódicos")
            
            st.info("\n".join(recommendations))
            
            # Mostrar precisión del modelo
            st.markdown("### 🎯 Precisión del Modelo")
            st.progress(0.8435)
            st.caption("El modelo tiene una precisión del 84.35% en la predicción de cancelaciones")