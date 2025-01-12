import streamlit as st
import joblib
from PIL import Image
import numpy as np
import warnings

# Ignorar las advertencias de versión de scikit-learn
warnings.filterwarnings('ignore', category=UserWarning)

# Configuración de la página
st.set_page_config(
    page_title="Predicción estrellas por imagen",
    page_icon="⭐",
    layout="wide"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #ffd700;
    }
    .metric-card {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    /* Spinner personalizado */
    .prediction-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #ffd700;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .star-rating {
        color: #ffd700;
        font-size: 2.5em;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⭐ Predicción de estrellas por imagen")
st.markdown("""
    Esta herramienta utiliza un modelo de Machine Learning, SIN la ultilización de REDES NEURONALES, para predecir 
    la clasificación por estrellas de un hotel basándose en una imagen de sus instalaciones.
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        return joblib.load('src/models/hoteles_foto.joblib')
    except Exception as e:
        st.error("Error al cargar el modelo. Asegúrate de que el archivo 'src/models/hoteles_foto.joblib' existe.")
        return None

model = load_model()
if model is None:
    st.stop()

def preprocess_image(image):
    try:
        # Redimensionar la imagen al tamaño que espera el modelo (30x90)
        image = image.resize((90, 30))
        # Convertir a array y normalizar
        img_array = np.array(image) / 255.0
        # Aplanar la imagen (convertir de 3D a 2D)
        img_array = img_array.reshape(1, -1)
        return img_array
    except Exception as e:
        st.error(f"Error al preprocesar la imagen: {str(e)}")
        return None

# Formulario principal
with st.form("image_prediction_form"):
    st.subheader("📸 Subir imagen del hotel")
    
    # Campo para subir imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen del hotel",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen clara de las instalaciones del hotel"
    )
    
    # Mostrar imagen subida
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Crear dos columnas y mostrar la imagen en una de ellas para reducir el ancho
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Imagen subida", width=400)
        except Exception as e:
            st.error(f"Error al abrir la imagen: {str(e)}")
    
    predict_button = st.form_submit_button("⭐ Predecir clasificación")

if predict_button and uploaded_file is not None:
    # Mostrar spinner personalizado
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("""
        <div class="prediction-spinner"></div>
        <p style='text-align: center; color: #666;'>Analizando imagen...</p>
    """, unsafe_allow_html=True)
    
    try:
        # Preprocesar imagen y hacer predicción
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            prediction = model.predict(processed_image)[0]
            predicted_stars = prediction  # Asumiendo que el modelo ya predice directamente el número de estrellas
            confidence = model.predict_proba(processed_image)[0].max() * 100
            
            # Eliminar spinner
            spinner_placeholder.empty()
            
            # Mostrar resultados
            st.write("---")
            st.subheader("🎯 Resultados del análisis")
            
            results_container = st.container()
            with results_container:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Mostrar predicción de estrellas
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='text-align: center;'>Clasificación predicha</h3>
                            <div class="star-rating">{"⭐" * int(predicted_stars)}</div>
                            <p style='text-align: center;'>{int(predicted_stars)} estrellas</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar confianza de la predicción
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='text-align: center;'>Confianza de la predicción</h3>
                            <h2 style='text-align: center; font-size: 2em;'>
                                {confidence:.1f}%
                            </h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Características detectadas
                    st.markdown("### 📊 Características comunes en hoteles con estas estrellas")
                    
                    # Mapeo de características por número de estrellas
                    features = {
                        5: ["🏊‍♂️ Piscina de lujo", "🍽️ Restaurantes gourmet", "🎭 Instalaciones de entretenimiento premium"],
                        4: ["🏊‍♂️ Piscina", "🍽️ Múltiples restaurantes", "💆‍♂️ Spa"],
                        3: ["🍽️ Restaurante", "🏋️‍♂️ Gimnasio", "🎯 Áreas recreativas"],
                        2: ["☕ Cafetería", "🛋️ Sala de estar común"],
                        1: ["🛏️ Servicios básicos"]
                    }
                    
                    for feature in features[int(predicted_stars)]:
                        st.markdown(f"- {feature}")

                    # Recomendaciones
                    st.markdown("### 📊 Recomendaciones")
                    st.markdown("⚠️ Recuerda que esta predicción es solo orientativa y puede variar dependiendo de la ubicación del hotel y de las características de la imagen.")
                    st.markdown("- Asegurate de que la imagen sea clara y de buena calidad.")
                    st.markdown("- Utiliza imágenes bien iluminadas y representativas del hotel.")

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        spinner_placeholder.empty()