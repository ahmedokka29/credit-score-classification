# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from utils.predictions import predict_credit_score
from utils.preprocessing import preprocess_input_data
import plotly.graph_objects as go
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model artifacts


@st.cache_resource
def load_model_artifacts():
    # Load your saved model files here
    model = joblib.load('models/best_rf_model_*.joblib')  # Use actual filename
    scaler = joblib.load('models/feature_scaler_*.joblib')

    with open('models/model_metadata_*.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, scaler, metadata

# Main application


def main():
    st.title("🏦 Credit Score Prediction System")
    st.markdown("### AI-Powered Credit Assessment Tool")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["Single Prediction", "Batch Prediction", "Model Information"])

    # Load model
    try:
        model, scaler, metadata = load_model_artifacts()
        st.sidebar.success("✅ Model loaded successfully")
        st.sidebar.info(f"Model F1 Score: {metadata['f1_score']:.4f}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    if page == "Single Prediction":
        single_prediction_page(model, scaler, metadata)
    elif page == "Batch Prediction":
        batch_prediction_page(model, scaler, metadata)
    else:
        model_information_page(metadata)


def single_prediction_page(model, scaler, metadata):
    st.header("Single Customer Prediction")

    # Create input form based on your features
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Add your input fields based on your features
            age = st.number_input("Age", min_value=14, max_value=70, value=30)
            annual_income = st.number_input(
                "Annual Income", min_value=0, value=50000)
            # Add more fields based on your feature set

        with col2:
            # More input fields
            # payment_history = st.selectbox("Payment History", options=[...])
            # Add more fields
            pass

        submitted = st.form_submit_button("Predict Credit Score")

        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'annual_income': [annual_income],
                # Add all your features here
            })

            # Make prediction
            predictions = predict_credit_score(
                input_data, model, scaler,
                metadata['feature_names'], metadata['numerical_columns'], metadata
            )

            # Display results
            result = predictions[0]

            # Create gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                title={'text': "Prediction Confidence"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 90}}))

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric("Predicted Credit Score", result['predicted_label'])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': list(result['probabilities'].keys()),
                    'Probability': [v*100 for v in result['probabilities'].values()]
                })

                fig2 = px.bar(prob_df, x='Class', y='Probability',
                              title="Prediction Probabilities")
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
