import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

print("Cargando datos...")
# Cargar y preparar datos
df = pd.read_csv('src/data/hotel_bookings.csv')

# Añadir características derivadas
print("Creando características adicionales...")
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['is_weekend_arrival'] = df['arrival_date_day_of_month'].apply(lambda x: x % 7 in [0, 6])
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['avg_guests_per_night'] = df['total_guests'] * df['total_nights']
df['booking_flexibility'] = np.where(df['deposit_type'] == 'No Deposit', 1, 0)

# Seleccionar características relevantes
features = [
    'lead_time', 'arrival_date_year', 'arrival_date_month',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'meal', 'market_segment', 'deposit_type', 'reserved_room_type',
    'total_of_special_requests', 'total_guests', 'is_weekend_arrival',
    'total_nights', 'avg_guests_per_night', 'booking_flexibility'
]

X = df[features]
y = df['adr']

print("Limpiando datos...")
# Limpiar datos y manejar valores atípicos
y = y.replace([np.inf, -np.inf], np.nan)
y_mean = y.mean()
y_std = y.std()
y = np.clip(y, y_mean - 3*y_std, y_mean + 3*y_std)  # Eliminar valores atípicos
y = y.fillna(y_mean)

# Separar características numéricas y categóricas
numeric_features = [
    'lead_time', 'arrival_date_year', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
    'children', 'babies', 'total_of_special_requests', 'total_guests',
    'total_nights', 'avg_guests_per_night', 'booking_flexibility'
]

categorical_features = [
    'meal', 'market_segment', 'deposit_type',
    'reserved_room_type', 'arrival_date_month'
]

# Crear preprocesadores con manejo robusto de valores atípicos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Usar mediana en lugar de media
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])

# Crear preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear pipeline completo con hiperparámetros optimizados
print("Creando pipeline...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=500,  # Más árboles
        learning_rate=0.05,  # Tasa de aprendizaje más baja
        max_depth=6,  # Profundidad mayor
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,  # Usar 80% de las muestras en cada árbol
        max_features='sqrt',  # Usar sqrt(n_features) características en cada split
        random_state=42
    ))
])

print("Dividiendo datos en entrenamiento y prueba...")
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("Entrenando modelo...")
model.fit(X_train, y_train)

# Evaluar modelo con validación cruzada
print("Evaluando modelo con validación cruzada...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Puntuaciones de validación cruzada: {cv_scores}")
print(f"Media de validación cruzada R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluar en conjunto de prueba
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"R² en entrenamiento: {train_score:.4f}")
print(f"R² en prueba: {test_score:.4f}")

# Guardar modelo
print("Guardando modelo...")
joblib.dump(model, 'src/models/adr_gbr.joblib')
print("¡Modelo guardado exitosamente!")