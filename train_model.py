import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json

# Load Framingham dataset
print("Loading Framingham dataset...")
df = pd.read_csv('framingham.csv')

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values
# For numeric columns, fill with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Feature engineering - create derived features
df['pack_years'] = (df['cigsPerDay'] * df['age']) / 20  # Approximate pack-years
df['smoking_intensity'] = df['currentSmoker'] * df['cigsPerDay']

# Define mandatory and optional features
mandatory_features = [
    'age',
    'male',  # gender (1=male, 0=female)
    'currentSmoker',
    'cigsPerDay',
]

optional_features = [
    'BPMeds',
    'prevalentStroke',
    'prevalentHyp',
    'diabetes',
    'totChol',
    'sysBP',
    'diaBP',
    'BMI',
    'heartRate',
    'glucose',
    'pack_years',
    'smoking_intensity'
]

all_features = mandatory_features + optional_features
target = 'TenYearCHD'

# Prepare features and target
X = df[all_features].copy()
y = df[target].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")

# Build Neural Network Model
def create_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')  # Output probability
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model

# Create and train model
print("\nBuilding model...")
model = create_model(X_train_scaled.shape[1])
model.summary()

# Add callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

print("\nTraining model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    class_weight={0: 1.0, 1: 3.0}  # Handle class imbalance
)

# Evaluate model
print("\nEvaluating model on test set...")
test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate additional metrics
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
print("\nSaving model and preprocessing objects...")
model.save('cvd_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names and metadata
model_metadata = {
    'mandatory_features': mandatory_features,
    'optional_features': optional_features,
    'all_features': all_features,
    'feature_means': X_train.mean().to_dict(),
    'feature_stds': X_train.std().to_dict(),
    'test_accuracy': float(test_results[1]),
    'test_auc': float(test_results[2])
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\nModel training complete!")
print("Saved files:")
print("  - cvd_model.h5 (Neural network model)")
print("  - scaler.pkl (Feature scaler)")
print("  - model_metadata.json (Model metadata)")
print("\nYou can now use these files in your Flask application.")