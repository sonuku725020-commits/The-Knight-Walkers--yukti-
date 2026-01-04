from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load Model
def load_model():
    """Load model and artifacts"""
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "models" / "heart_disease_model.pkl"
        scaler_path = base_path / "models" / "scaler.pkl"
        feature_names_path = base_path / "models" / "feature_names.pkl"

        # Suppress sklearn version warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(feature_names_path)

        return model, scaler, feature_names, True
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, False

model, scaler, feature_names, model_loaded = load_model()

# Pydantic models for input/output
class PatientData(BaseModel):
    age: int
    sex: int  # 0=Female, 1=Male
    chest_pain_type: int  # 1-4
    bp: int
    cholesterol: int
    fbs_over_120: int  # 0=No, 1=Yes
    ekg_results: int  # 0-2
    max_hr: int
    exercise_angina: int  # 0=No, 1=Yes
    st_depression: float
    slope_of_st: int  # 1-3
    number_of_vessels_fluro: int  # 0-3
    thallium: int  # 3-6

class PredictionResponse(BaseModel):
    prediction: int  # 0=No disease, 1=Disease
    probability: float
    risk_level: str
    risk_emoji: str
    recommendations: list[str]

# Helper functions (copied from app.py)
def preprocess_input(input_data):
    """Preprocess user input for prediction"""
    df = pd.DataFrame([input_data])

    # Feature Engineering
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    if 'BP' in df.columns:
        df['BP_Category'] = pd.cut(df['BP'], bins=[0, 120, 140, 300], labels=[0, 1, 2]).astype(int)
    if 'Cholesterol' in df.columns:
        df['Chol_Risk'] = (df['Cholesterol'] > 200).astype(int)
    if 'Max HR' in df.columns:
        df['HR_Risk'] = (df['Max HR'] < 100).astype(int)

    # Reorder columns
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled_data = scaler.transform(df)
    return scaled_data

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "🟢"
    elif probability < 0.6:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"

def get_recommendations(probability, data):
    """Generate basic recommendations (fallback)"""
    recommendations = []

    if probability > 0.5:
        recommendations.append("🏥 Schedule an appointment with a cardiologist for comprehensive evaluation")

    if data.get('Cholesterol', 0) > 200:
        recommendations.append("🥗 Reduce cholesterol intake - follow a heart-healthy diet rich in omega-3 fatty acids")

    if data.get('BP', 0) > 140:
        recommendations.append("💊 Monitor blood pressure regularly and consider lifestyle modifications")

    if data.get('Max HR', 200) < 100:
        recommendations.append("🏃 Increase physical activity gradually with doctor's approval")

    if data.get('FBS over 120', 0) == 1:
        recommendations.append("🍬 Control blood sugar levels through diet and regular monitoring")

    if data.get('Age', 0) > 50:
        recommendations.append("📅 Regular annual health checkups and cardiovascular screenings recommended")

    # Additional recommendations
    additional_recs = [
        "💧 Stay hydrated and drink at least 8 glasses of water daily",
        "🛌 Ensure 7-8 hours of quality sleep each night",
        "🚭 Avoid smoking and limit alcohol consumption",
        "🧘 Practice stress management techniques like meditation or yoga",
        "🥦 Include more fruits, vegetables, and whole grains in your diet"
    ]

    for rec in additional_recs:
        if len(recommendations) >= 5:
            break
        if rec not in recommendations:
            recommendations.append(rec)

    return recommendations[:5]

def get_gemini_recommendations(probability, data):
    """Generate AI recommendations using Google Gemini"""
    global client

    try:
        if not client:
            return get_recommendations(probability, data)

        prompt = f"""
        You are a medical health advisor AI. Based on the following heart disease risk assessment data,
        provide EXACTLY 5 personalized, actionable health recommendations.

        Focus on evidence-based advice for heart health improvement.

        **Risk Assessment Results:**
        - Risk Probability: {probability*100:.1f}%
        - Risk Level: {'High' if probability > 0.6 else 'Medium' if probability > 0.3 else 'Low'}

        **Patient Data:**
        - Age: {data.get('Age', 'N/A')} years
        - Sex: {'Male' if data.get('Sex', 0) == 1 else 'Female'}
        - Blood Pressure: {data.get('BP', 'N/A')} mmHg
        - Cholesterol: {data.get('Cholesterol', 'N/A')} mg/dL
        - Maximum Heart Rate: {data.get('Max HR', 'N/A')} bpm
        - Fasting Blood Sugar > 120: {'Yes' if data.get('FBS over 120', 0) == 1 else 'No'}

        **Instructions:**
        1. Provide EXACTLY 5 recommendations
        2. Start each recommendation with a relevant emoji
        3. Make each recommendation specific and actionable
        4. Consider the patient's specific risk factors
        5. Include both immediate actions and lifestyle changes

        Format: Start each recommendation on a new line with an emoji.
        """

        response = client.generate_content(prompt)

        recommendations_text = response.text.strip()

        # Parse recommendations
        lines = recommendations_text.split('\n')
        recommendations = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('**'):
                continue
            if line.startswith(('-', '*', '•')):
                line = line[1:].strip()
            if len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                line = line[2:].strip()

            if line and len(line) > 10:
                if not any(ord(c) > 127 for c in line[:2]):
                    emojis = ['🏥', '🥗', '💊', '🏃', '🍬', '📅', '✅', '🩺', '💓', '🥦', '🚶', '🏋️', '🛌', '🚭', '💧']
                    line = f"{emojis[len(recommendations) % len(emojis)]} {line}"
                recommendations.append(line)

        # Ensure at least 5
        if len(recommendations) < 5:
            fallback_recs = get_recommendations(probability, data)
            for rec in fallback_recs:
                if rec not in recommendations and len(recommendations) < 5:
                    recommendations.append(rec)

        return recommendations[:5]

    except Exception as e:
        print(f"AI recommendations unavailable: {str(e)}")
        return get_recommendations(probability, data)

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(data: PatientData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_data = {
        'Age': data.age,
        'Sex': data.sex,
        'Chest pain type': data.chest_pain_type,
        'BP': data.bp,
        'Cholesterol': data.cholesterol,
        'FBS over 120': data.fbs_over_120,
        'EKG results': data.ekg_results,
        'Max HR': data.max_hr,
        'Exercise angina': data.exercise_angina,
        'ST depression': data.st_depression,
        'Slope of ST': data.slope_of_st,
        'Number of vessels fluro': data.number_of_vessels_fluro,
        'Thallium': data.thallium
    }

    try:
        # Preprocess and predict
        processed_data = preprocess_input(input_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

        risk_level, emoji = get_risk_level(probability)

        # Get recommendations
        recommendations = get_gemini_recommendations(probability, input_data)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            risk_emoji=emoji,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)