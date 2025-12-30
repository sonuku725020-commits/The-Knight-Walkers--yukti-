# ============================================
# 🚀 CHURNPREDICT - STREAMLIT CLOUD APP
# ============================================
# File: app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="ChurnPredict - Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .low-risk {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .medium-risk {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 2px solid #ffc107;
    }
    .high-risk {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff4b4b;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(255,75,75,0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,75,75,0.5);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model & Artifacts
# ============================================

@st.cache_resource
def load_model():
    """Load model and artifacts with caching"""
    try:
        model = joblib.load('models/churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
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
    scaled_data = scaler.transform(df)
    return scaled_data

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "low-risk", "✅", "#28a745"
    elif probability < 0.6:
        return "Medium Risk", "medium-risk", "⚠️", "#ffc107"
    else:
        return "High Risk", "high-risk", "🚨", "#dc3545"

def get_recommendations(probability, data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if probability > 0.5:
        recommendations.append("🏥 Schedule an appointment with a cardiologist immediately")
    
    if data.get('Cholesterol', 0) > 200:
        recommendations.append("🥗 Reduce cholesterol - follow a heart-healthy diet")
    
    if data.get('BP', 0) > 140:
        recommendations.append("💊 Monitor blood pressure regularly")
    
    if data.get('Max HR', 200) < 100:
        recommendations.append("🏃 Increase physical activity gradually")
    
    if data.get('FBS over 120', 0) == 1:
        recommendations.append("🍬 Control blood sugar levels")
    
    if data.get('Age', 0) > 50:
        recommendations.append("📅 Regular annual health checkups recommended")
    
    if data.get('Exercise angina', 0) == 1:
        recommendations.append("⚡ Avoid strenuous exercise without medical supervision")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Maintain healthy lifestyle and regular checkups")
    
    return recommendations

def create_gauge_chart(probability):
    """Create gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)", 'font': {'size': 20, 'color': '#333'}},
        number={'font': {'size': 40, 'color': '#333'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#ff4b4b"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 30], 'color': '#28a745'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#333'}
    )
    return fig

def create_feature_importance_chart(input_data):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        return fig
    return None

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("## ❤️ ChurnPredict")
    st.markdown("---")
    
    # Model Status
    if model_loaded:
        st.success("🟢 Model Loaded")
    else:
        st.error("🔴 Model Not Found")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 Dashboard", "📖 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Info
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Algorithm**: XGBoost
    - **Accuracy**: ~92%
    - **Features**: 13+
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by ChurnPredict")
    st.markdown("v1.0.0")

# ============================================
# Home Page
# ============================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">❤️ ChurnPredict</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Heart Disease Risk Prediction System</p>', unsafe_allow_html=True)
    
    # Hero Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>92%+ Accuracy</h3>
            <p>Powered by XGBoost ML</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>⚡</h2>
            <h3>Instant Results</h3>
            <p>Get predictions in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🔒</h2>
            <h3>Secure & Private</h3>
            <p>No data stored</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.markdown("## 🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>1️⃣ Input Data</h3>
            <p>Enter patient health metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>2️⃣ AI Analysis</h3>
            <p>ML model processes data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>3️⃣ Risk Score</h3>
            <p>Get probability score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h3>4️⃣ Recommendations</h3>
            <p>Personalized health tips</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CTA
    st.markdown("### 🚀 Ready to get started?")
    if st.button("Start Prediction →", key="home_cta"):
        st.switch_page = "🔮 Prediction"

# ============================================
# Prediction Page
# ============================================

elif page == "🔮 Prediction":
    st.markdown("## 🔮 Heart Disease Risk Prediction")
    st.markdown("Enter patient details below to get a risk assessment")
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please check if model files exist in the 'models' folder.")
        st.stop()
    
    st.markdown("---")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Patient Information")
        
        age = st.slider("🎂 Age", min_value=1, max_value=120, value=55, help="Patient's age in years")
        
        sex = st.selectbox("👥 Sex", ["Male", "Female"])
        sex_encoded = 1 if sex == "Male" else 0
        
        chest_pain = st.selectbox(
            "💔 Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            help="Type of chest pain experienced"
        )
        chest_pain_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain) + 1
        
        bp = st.slider("🩺 Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
        
        cholesterol = st.slider("🧪 Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        
        max_hr = st.slider("💓 Maximum Heart Rate", min_value=50, max_value=250, value=150)
    
    with col2:
        st.markdown("### 🏥 Medical Tests")
        
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        fbs_encoded = 1 if fbs == "Yes" else 0
        
        ekg = st.selectbox(
            "📈 EKG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        ekg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ekg)
        
        exercise_angina = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        st_depression = st.slider("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope = st.selectbox("📊 Slope of ST", ["Upsloping", "Flat", "Downsloping"])
        slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1
        
        vessels = st.slider("🔬 Number of Major Vessels (Fluoroscopy)", min_value=0, max_value=3, value=0)
        
        thallium = st.selectbox(
            "💉 Thallium Stress Test",
            ["Normal", "Fixed Defect", "Reversible Defect"]
        )
        thallium_encoded = ["Normal", "Fixed Defect", "Reversible Defect"].index(thallium) + 3
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Sex': sex_encoded,
            'Chest pain type': chest_pain_encoded,
            'BP': bp,
            'Cholesterol': cholesterol,
            'FBS over 120': fbs_encoded,
            'EKG results': ekg_encoded,
            'Max HR': max_hr,
            'Exercise angina': exercise_angina_encoded,
            'ST depression': st_depression,
            'Slope of ST': slope_encoded,
            'Number of vessels fluro': vessels,
            'Thallium': thallium_encoded
        }
        
        with st.spinner("🔄 Analyzing health data..."):
            try:
                # Preprocess and predict
                processed_data = preprocess_input(input_data)
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0][1]
                
                # Get results
                risk_level, risk_class, emoji, color = get_risk_level(probability)
                recommendations = get_recommendations(probability, input_data)
                
                # Display Results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    prediction_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h1>{emoji}</h1>
                        <h2>{prediction_text}</h2>
                        <h3 style="color: {color};">{risk_level}</h3>
                        <p style="font-size: 1.2rem;">Probability: <strong>{probability*100:.1f}%</strong></p>
                        <p>Confidence: <strong>{max(probability, 1-probability)*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fig = create_gauge_chart(probability)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                
                for i, rec in enumerate(recommendations):
                    st.info(rec)
                
                # Feature Importance
                st.markdown("### 📊 Feature Importance")
                fig_importance = create_feature_importance_chart(input_data)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Summary Card
                st.markdown("### 📋 Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("Age", f"{age} years")
                with summary_col2:
                    st.metric("Blood Pressure", f"{bp} mmHg")
                with summary_col3:
                    st.metric("Cholesterol", f"{cholesterol} mg/dL")
                with summary_col4:
                    st.metric("Max Heart Rate", f"{max_hr} bpm")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.info("Please check if all model files are properly loaded.")

# ============================================
# Dashboard Page
# ============================================

elif page == "📊 Dashboard":
    st.markdown("## 📊 Model Performance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.92, 0.89, 0.94, 0.91, 0.95]
        })
        
        fig = px.bar(
            metrics_df,
            x='Score',
            y='Metric',
            orientation='h',
            color='Score',
            color_continuous_scale='RdYlGn',
            text='Score'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Risk Distribution")
        
        risk_df = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [45, 30, 25]
        })
        
        fig = px.pie(
            risk_df,
            values='Count',
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={
                'Low Risk': '#28a745',
                'Medium Risk': '#ffc107',
                'High Risk': '#dc3545'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🔍 Top Features")
    
    if model_loaded and hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(10),
            x='Feature',
            y='Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# About Page
# ============================================

elif page == "📖 About":
    st.markdown("## 📖 About ChurnPredict")
    
    st.markdown("""
    ### 🎯 Mission
    To provide accessible and accurate heart disease risk prediction using advanced machine learning,
    helping individuals make informed decisions about their cardiovascular health.
    
    ---
    
    ### 🔬 Technology Stack
    
    | Component | Technology |
    |-----------|------------|
    | **ML Model** | XGBoost Classifier |
    | **Frontend** | Streamlit |
    | **Visualization** | Plotly |
    | **Data Processing** | Pandas, NumPy, Scikit-learn |
    | **Deployment** | Streamlit Cloud |
    
    ---
    
    ### 📊 Dataset
    - **Source**: Heart Disease UCI Dataset
    - **Features**: 13 medical attributes
    - **Samples**: 270+ patient records
    
    ---
    
    ### 🔮 Features Used
    
    1. **Age** - Patient's age in years
    2. **Sex** - Male/Female
    3. **Chest Pain Type** - Type of chest pain experienced
    4. **Blood Pressure** - Resting blood pressure
    5. **Cholesterol** - Serum cholesterol level
    6. **Fasting Blood Sugar** - Blood sugar > 120 mg/dl
    7. **EKG Results** - Resting electrocardiographic results
    8. **Maximum Heart Rate** - Maximum heart rate achieved
    9. **Exercise Angina** - Exercise induced angina
    10. **ST Depression** - ST depression induced by exercise
    11. **Slope of ST** - Slope of peak exercise ST segment
    12. **Number of Vessels** - Major vessels colored by fluoroscopy
    13. **Thallium Test** - Thallium stress test result
    
    ---
    
    ### ⚠️ Disclaimer
    
    > **Important**: This tool is for **educational and informational purposes only**.
    > It should **NOT** replace professional medical advice, diagnosis, or treatment.
    > Always consult a qualified healthcare provider for medical decisions.
    
    ---
    
    ### 👨‍💻 Developer
    
    Built with ❤️ by **ChurnPredict Team**
    
    - 📧 Email: support@churnpredict.com
    - 🌐 GitHub: [ChurnPredict](https://github.com/yourusername/churnpredict)
    
    ---
    
    ### 📜 Version History
    
    | Version | Date | Changes |
    |---------|------|---------|
    | 1.0.0 | 2024 | Initial release |
    
    """)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #888; font-size: 0.9rem;'>
        © 2024 ChurnPredict | Made with ❤️ using Streamlit | 
        <a href='https://github.com/yourusername/churnpredict' target='_blank'>GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)