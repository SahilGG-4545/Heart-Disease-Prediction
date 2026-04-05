"""
Preprocessing utility for heart disease prediction.
Handles input validation and feature transformation.
"""

import pandas as pd
import numpy as np

class HeartDiseasePreprocessor:
    """Handles preprocessing and validation of patient data."""
    
    # Feature mappings for categorical variables
    CHEST_PAIN_MAPPING = {
        'typical_angina': 'typical angina',
        'atypical_angina': 'atypical angina',
        'non_anginal_pain': 'non-anginal pain',
        'asymptomatic': 'asymptomatic'
    }
    
    REST_ECG_MAPPING = {
        'normal': 'normal',
        'st_t_abnormality': 'ST-T wave abnormality',
        'left_vent_hypertrophy': 'left ventricular hypertrophy'
    }
    
    ST_SLOPE_MAPPING = {
        'normal': 'normal',
        'upsloping': 'upsloping',
        'flat': 'flat',
        'downsloping': 'downsloping'
    }
    
    SEX_MAPPING = {
        'male': 'male',
        'female': 'female'
    }
    
    def __init__(self, feature_names, numeric_features, scaler):
        """
        Initialize preprocessor with feature names and scaler.
        
        Args:
            feature_names: List of all feature names
            numeric_features: List of numeric feature names
            scaler: Fitted MinMaxScaler object
        """
        self.feature_names = feature_names
        self.numeric_features = numeric_features
        self.scaler = scaler
    
    def validate_inputs(self, data):
        """
        Validate input data for realistic medical values.
        
        Args:
            data: Dictionary with patient information
            
        Returns:
            Tuple: (is_valid, error_message)
        """
        errors = []
        
        # Age validation
        age = data.get('age')
        if age is None or not self._is_number(age):
            errors.append("Age must be a valid number")
        elif not (1 <= float(age) <= 120):
            errors.append("Age must be between 1 and 120 years")
        
        # Sex validation
        sex = data.get('sex')
        if sex not in self.SEX_MAPPING:
            errors.append("Sex must be 'male' or 'female'")
        
        # Chest pain type validation
        chest_pain = data.get('chest_pain_type')
        if chest_pain not in self.CHEST_PAIN_MAPPING:
            errors.append("Invalid chest pain type")
        
        # Cholesterol validation
        cholesterol = data.get('cholesterol')
        if cholesterol is None or not self._is_number(cholesterol):
            errors.append("Cholesterol must be a valid number")
        elif float(cholesterol) < 0 or float(cholesterol) > 600:
            errors.append("Cholesterol must be between 0 and 600 mg/dL")
        
        # Fasting blood sugar validation
        fbs = data.get('fasting_blood_sugar')
        if fbs not in ['0', '1', 0, 1]:
            errors.append("Fasting blood sugar must be 0 or 1")
        
        # Rest ECG validation
        rest_ecg = data.get('rest_ecg')
        if rest_ecg not in self.REST_ECG_MAPPING:
            errors.append("Invalid rest ECG type")
        
        # Max heart rate validation
        max_hr = data.get('max_heart_rate_achieved')
        if max_hr is None or not self._is_number(max_hr):
            errors.append("Max heart rate must be a valid number")
        elif not (60 <= float(max_hr) <= 220):
            errors.append("Max heart rate must be between 60 and 220 bpm")
        
        # Exercise induced angina validation
        eia = data.get('exercise_induced_angina')
        if eia not in ['0', '1', 0, 1]:
            errors.append("Exercise induced angina must be 0 or 1")
        
        # ST depression validation
        st_dep = data.get('st_depression')
        if st_dep is None or not self._is_number(st_dep):
            errors.append("ST depression must be a valid number")
        elif float(st_dep) < 0 or float(st_dep) > 10:
            errors.append("ST depression must be between 0 and 10")
        
        # ST slope validation
        st_slope = data.get('st_slope')
        if st_slope not in self.ST_SLOPE_MAPPING:
            errors.append("Invalid ST slope type")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, ""
    
    def preprocess(self, data):
        """
        Transform user input into model-ready format.
        
        Args:
            data: Dictionary with patient information
            
        Returns:
            numpy array ready for model prediction
        """
        # Validate first
        is_valid, error_msg = self.validate_inputs(data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Create dataframe with all features set to 0
        df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # Set numeric features
        df['age'] = float(data['age'])
        df['cholesterol'] = float(data['cholesterol'])
        df['max_heart_rate_achieved'] = float(data['max_heart_rate_achieved'])
        df['st_depression'] = float(data['st_depression'])
        df['fasting_blood_sugar'] = int(data['fasting_blood_sugar'])
        df['exercise_induced_angina'] = int(data['exercise_induced_angina'])
        
        # Set categorical one-hot encoded features
        if data['sex'] == 'male':
            df['sex_male'] = 1
        
        # Chest pain type
        chest_pain_encoded = self.CHEST_PAIN_MAPPING[data['chest_pain_type']]
        for col in df.columns:
            if col.startswith('chest_pain_type_'):
                if col == f"chest_pain_type_{chest_pain_encoded}":
                    df[col] = 1
        
        # Rest ECG
        rest_ecg_encoded = self.REST_ECG_MAPPING[data['rest_ecg']]
        for col in df.columns:
            if col.startswith('rest_ecg_'):
                if col == f"rest_ecg_{rest_ecg_encoded}":
                    df[col] = 1
        
        # ST Slope
        st_slope_encoded = self.ST_SLOPE_MAPPING[data['st_slope']]
        for col in df.columns:
            if col.startswith('st_slope_'):
                if col == f"st_slope_{st_slope_encoded}":
                    df[col] = 1
        
        # Normalize numeric features
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        
        return df.values[0]
    
    @staticmethod
    def _is_number(value):
        """Check if value can be converted to float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class ModelComparisonData:
    """Provides cross-validation comparison data for different models."""
    
    @staticmethod
    def get_comparison_data():
        """Return cross-validation accuracy scores from training."""
        return {
            'models': [
                'Stacked Classifier',
                'Random Forest',
                'XGBoost',
                'GBM'
            ],
            'accuracy': [
                0.8936,
                0.8809,
                0.8851,
                0.8383
            ]
        }


class DataVisualizationHelper:
    """Helper class for providing data for visualizations."""
    
    @staticmethod
    def get_age_distribution():
        """Age distribution statistics."""
        return {
            'bins': [20, 30, 40, 50, 60, 70, 80],
            'description': 'Average age of patients is around 55 years'
        }
    
    @staticmethod
    def get_risk_level(probability):
        """
        Classify risk level based on prediction probability.
        
        Args:
            probability: Model prediction probability (0-1)
            
        Returns:
            Tuple: (risk_level, color_code)
        """
        if probability < 0.33:
            return 'Low Risk', '#28a745'  # Green
        elif probability < 0.67:
            return 'Medium Risk', '#ffc107'  # Yellow
        else:
            return 'High Risk', '#dc3545'  # Red
