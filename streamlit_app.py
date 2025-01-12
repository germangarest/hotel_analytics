import streamlit as st

st.set_page_config(
    page_title="Hotel analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# 🏨 Hotel analytics")
st.markdown("""
    Bienvenido a la plataforma de análisis hotelero. Esta herramienta te ayuda a:
    
    ### 🔮 Predicción de cancelaciones
    - Predice la probabilidad de que una reserva sea cancelada
    - Obtén recomendaciones basadas en el riesgo de cancelación
    - Optimiza tu gestión de reservas
    
    ### 💰 Predicción de precios
    - Estima el precio medio por noche para una reserva
    - Analiza los factores que influyen en el precio
    - Recibe recomendaciones para optimizar tarifas
    
    Selecciona una de las opciones en el menú lateral para comenzar.
""")

# Información adicional
st.sidebar.success("Selecciona un modelo de los de arriba.")

# Footer
st.markdown("---")
st.markdown("Desarrollado con ❤️ por Germán, Carlos y José Antonio")
