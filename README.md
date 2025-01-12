# 🏨 Hotel Analytics

![Hotel](src/img/hotel.jpg) 

## 📋 Descripción
Hotel Analytics es una plataforma integral de análisis hotelero que utiliza modelos de Machine Learning para proporcionar predicciones y análisis en tres áreas clave:

1. **🔮 Predicción de Cancelaciones**
   - Predice la probabilidad de que una reserva sea cancelada
   - Análisis de factores de riesgo
   - Recomendaciones personalizadas para gestión de reservas
   - Precisión del modelo: 84.35%

2. **💰 Predicción de Precios**
   - Estima el precio medio por noche óptimo
   - Análisis de factores que influyen en el precio
   - Recomendaciones para optimización de tarifas
   - R² Score: 0.75

3. **⭐ Clasificación por Imagen**
   - Predice la categoría de estrellas de un hotel basándose en imágenes
   - Análisis de características comunes por categoría
   - Implementación sin redes neuronales
   - Interfaz intuitiva para carga de imágenes

## 🛠️ Tecnologías Utilizadas
- **Frontend**: Streamlit
- **Análisis de Datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Procesamiento de Imágenes**: Pillow
- **Gestión de Modelos**: Joblib

## 📁 Estructura del Proyecto
```
hotel-analytics/
├── Home.py                    # Página principal de la aplicación
├── pages/                     # Páginas de la aplicación
│   ├── 1_prediccion_cancelaciones.py
│   ├── 2_prediccion_precio.py
│   └── 3_prediccion_estrellas.py
├── src/
│   ├── models/               # Modelos entrenados
│   │   ├── cancelacion_model.joblib
│   │   ├── adr_gbr.joblib
│   │   └── hoteles_foto.joblib
│   └── utils/               # Utilidades y scripts de entrenamiento
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/germangarest/hotel-analytics.git
   cd hotel-analytics
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecución
1. Iniciar la aplicación:
   ```bash
   streamlit run Home.py
   ```
2. Abrir el navegador en `http://localhost:8501`

## 🎯 Características Principales

### Predicción de Cancelaciones
- Análisis de múltiples factores de riesgo
- Recomendaciones basadas en el nivel de riesgo
- Visualización clara de la probabilidad de cancelación

### Predicción de Precios
- Estimación precisa de precios por noche
- Análisis de factores que influyen en el precio
- Recomendaciones para optimización de ingresos

### Clasificación por Imagen
- Interfaz intuitiva para carga de imágenes
- Análisis de características por categoría
- Predicciones rápidas y precisas

## 👥 Autores
- **Germán García Estévez**
- **Carlos López Muñoz**
- **José Antonio García Antona**

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.
