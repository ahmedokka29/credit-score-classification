from xml.parsers.expat import model
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import warnings
import joblib
from utils.predictions import load_model, load_scaler, predict_with_thresholds_catboost
from utils.preprocessing import CreditScorePreprocessor, CreditDataPreprocessor
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_input_form():
    """Create the input form for user data."""
    st.title("💳 Credit Score Prediction")
    st.markdown("---")

    with st.form("credit_score_form"):
        st.subheader("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=14, max_value=100, value=30)
            occupation = st.selectbox("Occupation", [
                "", "architect", "developer", "doctor", "engineer", "entrepreneur",
                "journalist", "lawyer", "manager", "mechanic", "media_manager",
                "musician", "scientist", "teacher", "writer"
            ])
            annual_income = st.number_input(
                "Annual Income ($)", min_value=0.0, value=50000.0)
            monthly_inhand_salary = st.number_input(
                "Monthly In-hand Salary ($)", min_value=0.0, value=4000.0)

        with col2:
            # month = st.selectbox("Month", [
            #     "January", "February", "March", "May", "June", "July", "August", "September", "October", "November", "December"
            # ])
            num_bank_accounts = st.number_input(
                "Number of Bank Accounts", min_value=0, value=3)
            num_credit_card = st.number_input(
                "Number of Credit Cards", min_value=0, value=4)
            interest_rate = st.number_input(
                "Interest Rate (%)", min_value=0.0, max_value=50.0, value=15.0)

        st.subheader("Loan Information")

        col3, col4 = st.columns(2)

        with col3:
            num_of_loan = st.number_input(
                "Number of Loans", min_value=0, value=2)
            type_of_loan = st.multiselect("Types of Loan", [
                "mortgage_loan", "home_equity_loan", "debt_consolidation_loan",
                "credit-builder_loan", "auto_loan", "payday_loan",
                "not_specified", "personal_loan", "student_loan"
            ])
            delay_from_due_date = st.number_input(
                "Delay from Due Date (days)", min_value=0, value=15)
            num_of_delayed_payment = st.number_input(
                "Number of Delayed Payments", min_value=0.0, value=5.0)

        with col4:
            changed_credit_limit = st.number_input(
                "Changed Credit Limit ($)", value=0.0)
            num_credit_inquiries = st.number_input(
                "Number of Credit Inquiries", min_value=0, value=3)
            outstanding_debt = st.number_input(
                "Outstanding Debt ($)", min_value=0.0, value=20000.0)
            credit_utilization_ratio = st.number_input(
                "Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)

        st.subheader("Financial Behavior")

        col5, col6 = st.columns(2)

        with col5:
            st.write("**Credit History Age**")
            col_years, col_months = st.columns(2)
            with col_years:
                credit_history_years = st.number_input(
                    "Years", min_value=0, max_value=80, value=20)
            with col_months:
                credit_history_months = st.number_input(
                    "Months", min_value=0, max_value=11, value=0)

            # CHANGE: Added an empty string as the first option to force a selection.
            payment_of_min_amount = st.selectbox(
                "Payment of Minimum Amount", ["", "Yes", "No"])
            # CHANGE: Added an empty string as the first option to force a selection.
            credit_mix = st.selectbox(
                "Credit Mix", ["", "Good", "Standard", "Bad"])

        with col6:
            # CHANGE: Added an empty string as the first option to force a selection.
            payment_behaviour = st.selectbox("Payment Behaviour", [
                "", "high_spent_large_value_payments", "high_spent_medium_value_payments", "high_spent_small_value_payments",
                "low_spent_large_value_payments", "low_spent_medium_value_payments",
                "low_spent_small_value_payments"
            ])
            total_emi_per_month = st.number_input(
                "Total EMI per Month ($)", min_value=0.0, value=1000.0)
            amount_invested_monthly = st.number_input(
                "Amount Invested Monthly ($)", min_value=0.0, value=500.0)

        monthly_balance = st.number_input("Monthly Balance ($)", value=2000.0)

        submitted = st.form_submit_button(
            "Predict Credit Score", type="primary")

        if submitted:
            # --- START OF VALIDATION LOGIC ---
            # CHANGE: Added conditional checks for key categorical fields.
            if (not occupation or not credit_mix or not payment_of_min_amount or
                    not payment_behaviour):
                st.error("Please fill in all the required fields.")
                return None  # Stop execution if validation fails

            # CHANGE: Added validation for multi-select field.
            if not type_of_loan:
                st.error("Please select at least one type of loan.")
                return None

            # --- END OF VALIDATION LOGIC ---
            # First, construct the history age string from the two number inputs
            credit_history_age_str = f"{credit_history_years} Years and {credit_history_months} Months"
            # Prepare input data
            input_data = {
                'age': age,
                'occupation': occupation,
                'annual_income': annual_income,
                'monthly_inhand_salary': monthly_inhand_salary,
                # 'month': month,
                'num_bank_accounts': num_bank_accounts,
                'num_credit_card': num_credit_card,
                'interest_rate': interest_rate,
                'num_of_loan': num_of_loan,
                'type_of_loan': ', '.join(type_of_loan) if type_of_loan else '',
                'delay_from_due_date': delay_from_due_date,
                'num_of_delayed_payment': num_of_delayed_payment,
                'changed_credit_limit': changed_credit_limit,
                'num_credit_inquiries': num_credit_inquiries,
                'outstanding_debt': outstanding_debt,
                'credit_utilization_ratio': credit_utilization_ratio,
                'credit_history_age': credit_history_age_str,
                'payment_of_min_amount': payment_of_min_amount,
                'credit_mix': credit_mix,
                'payment_behaviour': payment_behaviour,
                'total_emi_per_month': total_emi_per_month,
                'amount_invested_monthly': amount_invested_monthly,
                'monthly_balance': monthly_balance
            }

            return input_data

    return None


def display_prediction(prediction, probability=None):
    """Display prediction results."""
    credit_score_mapping = {0: "Poor", 1: "Good", 2: "Standard"}

    # Color mapping for different scores
    color_mapping = {
        "Poor": "🔴",
        "Good": "🟢",
        "Standard": "🟡"
    }

    predicted_score = credit_score_mapping.get(prediction, "Unknown")

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px;
                    background-color: #2f3237; margin: 20px 0;'>
            <h2>{color_mapping.get(predicted_score, '❓')} Credit Score: {predicted_score}</h2>
        </div>
        """, unsafe_allow_html=True)

        if probability is not None:
            st.write("**Confidence Scores:**")
            for i, (score, prob) in enumerate(zip(["Poor", "Good", "Standard"], probability[0])):
                st.write(f"- {score}: {prob:.2%}")

    # Provide recommendations based on prediction
    st.subheader("💡 Recommendations")

    if predicted_score == "Poor":
        st.error("""
        **Recommendations to improve your credit score:**
        - Pay all bills on time, every time
        - Reduce credit utilization below 30%
        - Don't close old credit accounts
        - Consider a secured credit card if you have limited credit history
        - Monitor your credit report regularly
        """)
    elif predicted_score == "Standard":
        st.warning("""
        **Ways to achieve a 'Good' credit score:**
        - Maintain low credit utilization (under 10%)
        - Keep old accounts open to maintain credit history length
        - Diversify your credit mix responsibly
        - Avoid applying for multiple credit accounts in short periods
        """)
    else:
        st.success("""
        **Great job! Maintain your good credit score by:**
        - Continuing to pay bills on time
        - Keeping credit utilization low
        - Monitoring your credit report for errors
        - Being strategic about new credit applications
        """)


def batch_prediction_interface(model, thresholds, scaler):
    st.subheader("📂 Batch Credit Score Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head(10))

        # Run your bulk preprocessor
        preprocessor = CreditDataPreprocessor(
            dataframe=df)   # df = uploaded CSV
        processed_df = preprocessor.run_full_pipeline()

        # Drop target column if present
        if "credit_score" in processed_df.columns:
            X = processed_df.drop(columns=["credit_score"])
        else:
            X = processed_df
        scaler_features = scaler.feature_names_in_.tolist()
        # Scale features
        X[scaler_features] = scaler.transform(X[scaler_features])

        # Predict with thresholds
        predictions = predict_with_thresholds_catboost(model, X, thresholds)

        # Map predictions back to labels
        credit_score_mapping = {0: "Poor", 1: "Good", 2: "Standard"}
        processed_df["predicted_credit_score"] = [
            credit_score_mapping[p] for p in predictions]

        st.success("✅ Predictions generated successfully!")
        st.dataframe(
            processed_df[["predicted_credit_score"]].head(10))

        # Download option
        csv_out = processed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv_out,
            file_name="credit_predictions.csv",
            mime="text/csv"
        )


def main():
    """Main application function."""

    # Load model and scaler
    model, thresholds = load_model()
    scaler = load_scaler()
    # Initialize preprocessor with the scaler
    preprocessor = CreditScorePreprocessor(scaler=scaler)

    # Create sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.write("""
        This app predicts credit scores based on various financial and personal factors.

        **Credit Score Categories:**
        - 🔴 **Poor**: Needs significant improvement
        - 🟡 **Standard**: Average creditworthiness
        - 🟢 **Good**: Excellent creditworthiness

        **How to use:**
        1. Fill in your financial information
        2. Click 'Predict Credit Score'
        3. Review your results and recommendations
        """)

        st.markdown("---")
        st.caption(
            "⚠️ This tool is for educational purposes only and should not be considered as financial advice.")

    # Create main input form
    input_data = create_input_form()
    st.markdown("---")
    batch_prediction_interface(model, thresholds, scaler)
    # Process prediction if form is submitted
    if input_data:
        try:
            # Transform input data
            features = preprocessor.transform_input(input_data)

            st.success("✅ Data processed successfully!")

            # Display processed features (optional, for debugging)
            with st.expander("🔍 View Processed Features"):
                st.write("Feature vector shape:", features.shape)
                st.write("Sample of processed features:")
                feature_df = pd.DataFrame(features.T, columns=['Value'])
                st.dataframe(feature_df.head(10))

            # Make prediction
            if model is not None and scaler is not None:
                if thresholds:
                    prediction = predict_with_thresholds_catboost(
                        model, features, thresholds)[0]
                else:
                    prediction = model.predict(features)[0]

                # prediction = model.predict(features)[0]
                # Get prediction probabilities if available
                try:
                    probability = model.predict_proba(features)
                except:
                    probability = None

                # Display results
                display_prediction(prediction, probability)
            else:
                st.warning("""
                🚧 **Model not loaded yet!**
                
                To complete the setup:
                1. Train your credit scoring model
                2. Save it as a pickle file
                3. Update the `load_model()` function to load your trained model
                4. Replace the placeholder with actual model loading code
                
                For now, here's a preview of your processed features ready for prediction.
                """)

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.write("Please check your input values and try again.")


if __name__ == "__main__":
    main()
