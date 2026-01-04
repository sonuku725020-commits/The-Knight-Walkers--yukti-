# ============================================
# HEART DISEASE PREDICTOR - UPDATED VERSION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# New Google GenAI package
import google.generativeai as genai

# Configure Gemini API with new package
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        st.warning(f"⚠️ Failed to initialize Gemini: {e}")
else:
    st.warning("⚠️ GEMINI_API_KEY not found in environment variables")

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# Load Model
# ============================================

@st.cache_resource
def load_model():
    """Load model and artifacts"""
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "models" / "heart_disease_model.pkl"
        scaler_path = base_path / "models" / "scaler.pkl"
        feature_names_path = base_path / "models" / "feature_names.pkl"
        
        # Suppress sklearn version warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(feature_names_path)
        
        return model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

model, scaler, feature_names, model_loaded = load_model()

# ============================================
# Helper Functions
# ============================================

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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled_data = scaler.transform(df)
    return scaled_data

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "🟢", "#28a745"
    elif probability < 0.6:
        return "Medium Risk", "🟡", "#ffc107"
    else:
        return "High Risk", "🔴", "#dc3545"

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

    # Additional recommendations to ensure at least 5
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
    """Generate AI recommendations using Google Gemini - at least 5 recommendations"""
    global client
    
    try:
        if not client:
            st.warning("⚠️ Gemini client not initialized. Using basic recommendations.")
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
        - Blood Pressure: {data.get('BP', 'N/A')} mmHg {'(High)' if data.get('BP', 0) > 140 else '(Normal)' if data.get('BP', 0) > 90 else '(Low)'}
        - Cholesterol: {data.get('Cholesterol', 'N/A')} mg/dL {'(High)' if data.get('Cholesterol', 0) > 200 else '(Normal)'}
        - Maximum Heart Rate: {data.get('Max HR', 'N/A')} bpm
        - Fasting Blood Sugar > 120: {'Yes (Elevated)' if data.get('FBS over 120', 0) == 1 else 'No (Normal)'}
        - Chest Pain Type: {data.get('Chest pain type', 'N/A')}
        - Exercise Induced Angina: {'Yes' if data.get('Exercise angina', 0) == 1 else 'No'}
        - ST Depression: {data.get('ST depression', 'N/A')}

        **Instructions:**
        1. Provide EXACTLY 5 recommendations
        2. Start each recommendation with a relevant emoji
        3. Make each recommendation specific and actionable
        4. Consider the patient's specific risk factors
        5. Include both immediate actions and lifestyle changes
        
        Format: Start each recommendation on a new line with an emoji.
        """

        # Use new API format
        response = client.generate_content(prompt)
        
        recommendations_text = response.text.strip()

        # Parse recommendations
        lines = recommendations_text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('**'):
                continue
            # Remove bullet points or numbers
            if line.startswith(('-', '*', '•')):
                line = line[1:].strip()
            # Remove numbered prefixes like "1.", "2.", etc.
            if len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                line = line[2:].strip()
            
            if line and len(line) > 10:
                # Add emoji if missing
                if not any(ord(c) > 127 for c in line[:2]):
                    emojis = ['🏥', '🥗', '💊', '🏃', '🍬', '📅', '✅', '🩺', '💓', '🥦', '🚶', '🏋️', '🛌', '🚭', '💧']
                    line = f"{emojis[len(recommendations) % len(emojis)]} {line}"
                recommendations.append(line)
        
        # Ensure at least 5 recommendations
        if len(recommendations) < 5:
            fallback_recs = get_recommendations(probability, data)
            for rec in fallback_recs:
                if rec not in recommendations and len(recommendations) < 5:
                    recommendations.append(rec)
        
        return recommendations[:5]

    except Exception as e:
        st.warning(f"⚠️ AI recommendations unavailable: {str(e)}")
        return get_recommendations(probability, data)

def create_gauge_chart(probability):
    """Create gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)"},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff4b4b"},
            'steps': [
                {'range': [0, 30], 'color': '#28a745'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#dc3545'}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.title("❤️ Heart Disease Predictor")
    st.markdown("---")
    
    # Model Status
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Found")
    
    # Gemini Status
    if client:
        st.success("✅ Gemini AI Connected")
    else:
        st.warning("⚠️ Gemini Not Connected")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 Dashboard", "📖 About"]
    )
    
    st.markdown("---")
    st.info("**Model:** XGBoost\n\n**Accuracy:** ~92%\n\n**AI:** Google Gemini 2.0")

# ============================================
# Home Page
# ============================================

if page == "🏠 Home":
    st.title("❤️ Heart Disease Predictor")
    st.subheader("AI-Powered Heart Disease Risk Prediction")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "92%+")
    with col2:
        st.metric("Features", "13+")
    with col3:
        st.metric("Response", "Instant")
    with col4:
        st.metric("AI Recs", "5+")
    
    st.markdown("---")
    
    st.markdown("""
    ### How It Works
    1. **Enter Data** - Input patient health metrics
    2. **AI Analysis** - ML model processes data
    3. **Get Results** - Receive risk score and **5+ AI recommendations**
    
    ### Features
    - 🤖 **Google Gemini 2.0 AI** for personalized recommendations
    - 📊 **XGBoost ML Model** for accurate predictions
    - 📈 **Interactive visualizations** for easy understanding
    
    ### Get Started
    👈 Select **Prediction** from the sidebar to begin!
    """)

# ============================================
# Prediction Page
# ============================================

elif page == "🔮 Prediction":
    st.title("🔮 Heart Disease Risk Prediction")
    st.markdown("Enter patient details below:")
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please check if model files exist.")
        st.stop()
    
    st.markdown("---")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Patient Info")
        
        age = st.slider("Age", 1, 120, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_encoded = 1 if sex == "Male" else 0
        
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        )
        chest_pain_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain) + 1
        
        bp = st.slider("Blood Pressure (mmHg)", 50, 250, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        max_hr = st.slider("Maximum Heart Rate", 50, 250, 150)
    
    with col2:
        st.subheader("🏥 Medical Tests")
        
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        fbs_encoded = 1 if fbs == "Yes" else 0
        
        ekg = st.selectbox(
            "EKG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        ekg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ekg)
        
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        st_depression = st.slider("ST Depression", 0.0, 10.0, 1.0, 0.1)
        
        slope = st.selectbox("Slope of ST", ["Upsloping", "Flat", "Downsloping"])
        slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1
        
        vessels = st.slider("Number of Major Vessels", 0, 3, 0)
        
        thallium = st.selectbox("Thallium Test", ["Normal", "Fixed Defect", "Reversible Defect"])
        thallium_encoded = ["Normal", "Fixed Defect", "Reversible Defect"].index(thallium) + 3
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔮 Predict Risk", use_container_width=True, type="primary"):

        with st.spinner("🔄 Analyzing with AI..."):
            try:
                # Call the backend API
                api_data = {
                    "age": age,
                    "sex": sex_encoded,
                    "chest_pain_type": chest_pain_encoded,
                    "bp": bp,
                    "cholesterol": cholesterol,
                    "fbs_over_120": fbs_encoded,
                    "ekg_results": ekg_encoded,
                    "max_hr": max_hr,
                    "exercise_angina": exercise_angina_encoded,
                    "st_depression": st_depression,
                    "slope_of_st": slope_encoded,
                    "number_of_vessels_fluro": vessels,
                    "thallium": thallium_encoded
                }

                response = requests.post("http://localhost:8000/predict", json=api_data)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    probability = result['probability']
                    risk_level = result['risk_level']
                    emoji = result['risk_emoji']
                    recommendations = result['recommendations']

                    # Results
                    st.markdown("---")
                    st.subheader("📊 Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

                        st.markdown(f"""
                        ### {emoji} {result_text}

                        **Risk Level:** {risk_level}

                        **Probability:** {probability*100:.1f}%

                        **Confidence:** {max(probability, 1-probability)*100:.1f}%
                        """)

                    with col2:
                        fig = create_gauge_chart(probability)
                        st.plotly_chart(fig, use_container_width=True)

                    # AI Recommendations Section
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Recommendations (Powered by Google Gemini)")
                    st.caption(f"Generated {len(recommendations)} personalized recommendations based on your health data")

                    # Display recommendations in a nice format
                    for i, rec in enumerate(recommendations, 1):
                        st.info(f"**{i}.** {rec}")

                    # Summary
                    st.markdown("---")
                    st.subheader("📋 Patient Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Age", f"{age} years")
                    col2.metric("BP", f"{bp} mmHg", "High" if bp > 140 else "Normal")
                    col3.metric("Cholesterol", f"{cholesterol} mg/dL", "High" if cholesterol > 200 else "Normal")
                    col4.metric("Max HR", f"{max_hr} bpm")

                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================
# Dashboard Page
# ============================================

elif page == "📊 Dashboard":
    st.title("📊 Model Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.92, 0.89, 0.94, 0.91, 0.95]
        })
        
        fig = px.bar(metrics_df, x='Score', y='Metric', orientation='h', 
                     color='Score', color_continuous_scale='RdYlGn')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution")
        
        risk_df = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [45, 30, 25]
        })
        
        fig = px.pie(risk_df, values='Count', names='Risk Level',
                     color_discrete_sequence=['#28a745', '#ffc107', '#dc3545'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    if model_loaded and hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(importance_df, x='Feature', y='Importance', color='Importance')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# About Page
# ============================================

elif page == "📖 About":
    st.title("📖 About")
    
    st.markdown("""
    ## Heart Disease Predictor
    
    An AI-powered tool for predicting heart disease risk with **personalized recommendations**.
    
    ### Technology Stack
    - **ML Model:** XGBoost Classifier
    - **AI Recommendations:** Google Gemini 2.0 Flash
    - **Frontend:** Streamlit
    - **Visualizations:** Plotly
    
    ### Features Used for Prediction
    - Age, Sex, Chest Pain Type
    - Blood Pressure, Cholesterol
    - Fasting Blood Sugar, EKG Results
    - Max Heart Rate, Exercise Angina
    - ST Depression, Slope of ST
    - Number of Vessels, Thallium Test
    
    ### AI Recommendations
    The system uses **Google Gemini 2.0 AI** to generate at least **5 personalized recommendations** 
    based on your specific health data and risk factors.
    
    ### ⚠️ Disclaimer
    This tool is for **educational purposes only**. 
    Always consult a healthcare professional for medical advice.
    """)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Powered by Google Gemini 2.0 AI")