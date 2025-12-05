from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Enable CORS for all routes (allows Netlify frontend to call this API)
CORS(app)

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Student Performance Predictor API",
        "endpoints": {
            "POST /api/predict": "Get prediction for student performance"
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    Expects JSON body with:
    - gender: str
    - ethnicity: str
    - parental_level_of_education: str
    - lunch: str
    - test_preparation_course: str
    - reading_score: float
    - writing_score: float
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        required_fields = [
            'gender', 'ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Create CustomData object
        custom_data = CustomData(
            gender=data['gender'],
            race_ethnicity=data['ethnicity'],
            parental_level_of_education=data['parental_level_of_education'],
            lunch=data['lunch'],
            test_preparation_course=data['test_preparation_course'],
            reading_score=float(data['reading_score']),
            writing_score=float(data['writing_score'])
        )
        
        # Convert to DataFrame
        pred_df = custom_data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return jsonify({
            "success": True,
            "prediction": float(results[0]),
            "input": {
                "gender": data['gender'],
                "ethnicity": data['ethnicity'],
                "parental_level_of_education": data['parental_level_of_education'],
                "lunch": data['lunch'],
                "test_preparation_course": data['test_preparation_course'],
                "reading_score": data['reading_score'],
                "writing_score": data['writing_score']
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
