# Heart Disease Prediction

An AI-powered web application for predicting heart disease risk with personalized recommendations using machine learning and Google Gemini AI.

## Features

- 🤖 **XGBoost ML Model** for accurate heart disease prediction
- 🧠 **Google Gemini 2.0 AI** for personalized health recommendations
- 📊 **Interactive Dashboard** with model metrics and visualizations
- 🚀 **FastAPI Backend** for scalable API endpoints
- 🎨 **Streamlit Frontend** for user-friendly interface
- 📈 **Real-time Risk Assessment** with probability scores

## Architecture

- **Backend**: FastAPI server handling ML predictions and AI recommendations
- **Frontend**: Streamlit web application for user interaction
- **AI**: Google Gemini for generating personalized health advice
- **ML**: XGBoost classifier trained on heart disease dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sonuku725020-commits/Heart_Disease_Prediction1.git
cd Heart_Disease_Prediction1
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Running the Backend API

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### Running the Frontend

```bash
streamlit run app.py
```

The web app will be available at `http://localhost:8501`

### API Endpoints

- `GET /` - Health check
- `POST /predict` - Heart disease prediction with recommendations

## API Usage Example

```python
import requests

data = {
    "age": 55,
    "sex": 1,
    "chest_pain_type": 1,
    "bp": 120,
    "cholesterol": 200,
    "fbs_over_120": 0,
    "ekg_results": 0,
    "max_hr": 150,
    "exercise_angina": 0,
    "st_depression": 1.0,
    "slope_of_st": 1,
    "number_of_vessels_fluro": 0,
    "thallium": 3
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
```

## Model Features

The prediction model uses 13 health indicators:
- Age, Sex, Chest Pain Type
- Blood Pressure, Cholesterol
- Fasting Blood Sugar, EKG Results
- Maximum Heart Rate, Exercise Angina
- ST Depression, Slope of ST
- Number of Vessels, Thallium Test

## AI Recommendations

The system provides 5 personalized recommendations based on:
- Risk probability and level
- Individual health metrics
- Evidence-based medical advice
- Powered by Google Gemini 2.0 AI

## Disclaimer

⚠️ **This tool is for educational purposes only. Always consult a healthcare professional for medical advice.**

## Technologies Used

- **Machine Learning**: XGBoost, scikit-learn
- **AI**: Google Gemini 2.0
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

## License

This project is open source and available under the MIT License.