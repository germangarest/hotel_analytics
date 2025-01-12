import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from datetime import datetime

print("Cargando datos...")
# Cargar datos
df = pd.read_csv('src/data/hotel_bookings.csv')

# Crear características adicionales
print("Creando características avanzadas...")
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['is_weekend_arrival'] = df['arrival_date_day_of_month'].apply(lambda x: x % 7 in [0, 6])
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['avg_guests_per_night'] = df['total_guests'] * df['total_nights']
df['booking_flexibility'] = np.where(df['deposit_type'] == 'No Deposit', 1, 0)
df['high_season'] = df['arrival_date_month'].isin(['July', 'August', 'December'])
df['lead_time_category'] = pd.qcut(df['lead_time'], q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
df['price_per_night'] = df['adr'] / (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
df['total_cost'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
df['repeated_guest_value'] = np.where(df['is_repeated_guest'] == 1, df['previous_cancellations'], 0)
df['cancellation_risk'] = df['previous_cancellations'] / (df['previous_bookings_not_canceled'] + 1)

# Seleccionar características
features = [
    'lead_time', 'arrival_date_year', 'arrival_date_month',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'meal', 'market_segment', 'deposit_type', 'customer_type',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'is_repeated_guest',
    'total_guests', 'is_weekend_arrival', 'total_nights',
    'avg_guests_per_night', 'booking_flexibility', 'high_season',
    'lead_time_category', 'price_per_night', 'total_cost',
    'repeated_guest_value', 'cancellation_risk'
]

# Preparar datos
X = df[features]
y = df['is_canceled']

# Separar características
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Crear preprocesadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
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

# Crear pipeline con Random Forest
print("Creando pipeline...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])

# Crear pipeline con Gradient Boosting
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    ))
])

# Dividir datos
print("Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar y evaluar Random Forest
print("\nEntrenando Random Forest...")
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nMétricas Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall: {recall_score(y_test, rf_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, rf_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_pred_proba):.4f}")

# Entrenar y evaluar Gradient Boosting
print("\nEntrenando Gradient Boosting...")
gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)
gb_pred_proba = gb_pipeline.predict_proba(X_test)[:, 1]

print("\nMétricas Gradient Boosting:")
print(f"Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
print(f"Precision: {precision_score(y_test, gb_pred):.4f}")
print(f"Recall: {recall_score(y_test, gb_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, gb_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, gb_pred_proba):.4f}")

# Seleccionar el mejor modelo
rf_f1 = f1_score(y_test, rf_pred)
gb_f1 = f1_score(y_test, gb_pred)

best_model = rf_pipeline if rf_f1 > gb_f1 else gb_pipeline
model_name = "Random Forest" if rf_f1 > gb_f1 else "Gradient Boosting"

print(f"\nGuardando el mejor modelo ({model_name})...")
joblib.dump(best_model, 'src/models/cancelacion_model.joblib')
print("¡Modelo guardado exitosamente!")
