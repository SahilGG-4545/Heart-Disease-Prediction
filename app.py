"""
Heart Disease Prediction Flask Web Application
A production-ready web app for predicting heart disease risk using machine learning.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
from utils.preprocess import HeartDiseasePreprocessor, ModelComparisonData, DataVisualizationHelper

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load model, scaler, and feature info at startup
print("Loading model and resources...")
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_info = joblib.load('feature_info.pkl')

# Initialize preprocessor
preprocessor = HeartDiseasePreprocessor(
    feature_names=feature_info['feature_names'],
    numeric_features=feature_info['numeric_features'],
    scaler=scaler
)

print("[OK] Model loaded successfully")
print("[OK] Features:", feature_info['feature_names'])


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page with project introduction."""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page."""
    return render_template('predict.html')


@app.route('/result')
def result_page():
    """Result display page."""
    return render_template('result.html')


@app.route('/comparison')
def comparison():
    """Model comparison page."""
    comparison_data = ModelComparisonData.get_comparison_data()
    return render_template('comparison.html', data=comparison_data)


@app.route('/visualizations')
def visualizations():
    """Data visualization page."""
    return render_template('visualizations.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for heart disease prediction.
    
    Expected JSON:
    {
        "age": 54,
        "sex": "male",
        "chest_pain_type": "typical_angina",
        "cholesterol": 230,
        "fasting_blood_sugar": 0,
        "rest_ecg": "normal",
        "max_heart_rate_achieved": 150,
        "exercise_induced_angina": 0,
        "st_depression": 1.2,
        "st_slope": "upsloping"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate inputs
        is_valid, error_msg = preprocessor.validate_inputs(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Preprocess data
        features = preprocessor.preprocess(data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]
        
        # Classify risk level
        risk_level, color = DataVisualizationHelper.get_risk_level(probability)
        
        # Format response
        response = {
            'success': True,
            'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability': round(float(probability), 4),
            'risk_level': risk_level,
            'risk_color': color,
            'confidence': f"{round(max(probability, 1-probability) * 100, 2)}%",
            'input_data': data
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/validate', methods=['POST'])
def api_validate():
    """
    Validate user input without making prediction.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        is_valid, error_msg = preprocessor.validate_inputs(data)
        
        return jsonify({
            'success': is_valid,
            'error': error_msg if not is_valid else None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and features."""
    try:
        comparison_data = ModelComparisonData.get_comparison_data()
        return jsonify({
            'success': True,
            'features': feature_info['feature_names'],
            'numeric_features': feature_info['numeric_features'],
            'model_accuracy': 0.8936,
            'model_precision': 0.8657,
            'model_recall': 0.9431,
            'model_f1': 0.9027,
            'models_comparison': comparison_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/comparison-data', methods=['GET'])
def comparison_data_api():
    """Get model comparison data for visualization."""
    try:
        data = ModelComparisonData.get_comparison_data()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Page not found'
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    # Check if required files exist
    if not os.path.exists('model.pkl'):
        print("ERROR: model.pkl not found. Please run train_model.py first.")
        exit(1)
    
    print("\n" + "="*50)
    print("Heart Disease Prediction Web App")
    print("="*50)
    print("Starting Flask server...")
    print("→ Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
