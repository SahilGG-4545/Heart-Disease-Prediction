"""
Script to train and save the Random Forest model for heart disease prediction.
This model will be used by the Flask application.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
dt = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Rename features
print("Preprocessing data...")
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
              'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
              'exercise_induced_angina', 'st_depression', 'st_slope', 'target']

# Convert categorical features
dt['chest_pain_type'] = dt['chest_pain_type'].map({
    1: 'typical angina',
    2: 'atypical angina',
    3: 'non-anginal pain',
    4: 'asymptomatic'
})

dt['rest_ecg'] = dt['rest_ecg'].map({
    0: 'normal',
    1: 'ST-T wave abnormality',
    2: 'left ventricular hypertrophy'
})

dt['st_slope'] = dt['st_slope'].map({
    0: 'normal',
    1: 'upsloping',
    2: 'flat',
    3: 'downsloping'
})

dt["sex"] = dt.sex.apply(lambda x: 'male' if x == 1 else 'female')

# Remove outliers using z-score
dt_numeric = dt[['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved']]
z = np.abs(stats.zscore(dt_numeric))
dt = dt[(z < 3).all(axis=1)]

# Encode categorical variables
dt = pd.get_dummies(dt, drop_first=True)

# Drop resting_blood_pressure feature (based on feature selection in notebook)
X = dt.drop(['target', 'resting_blood_pressure'], axis=1)
y = dt['target']

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5
)

# Normalize numeric features
scaler = MinMaxScaler()
numeric_features = ['age', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train Random Forest model (best performer)
print("Training Random Forest model...")
rf_model = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

# Save model and scaler
print("\nSaving model and scaler...")
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names and numeric features names for preprocessing
feature_info = {
    'feature_names': X.columns.tolist(),
    'numeric_features': numeric_features,
    'categorical_mappings': {
        'chest_pain_type': ['atypical angina', 'asymptomatic', 'non-anginal pain', 'typical angina'],
        'rest_ecg': ['ST-T wave abnormality', 'left ventricular hypertrophy', 'normal'],
        'st_slope': ['downsloping', 'flat', 'upsloping'],
        'sex': ['male']
    }
}
joblib.dump(feature_info, 'feature_info.pkl')

print("\nModel saved as 'model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("Feature info saved as 'feature_info.pkl'")
