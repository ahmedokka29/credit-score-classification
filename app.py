import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import warnings
import joblib


warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


class CreditScorePreprocessor:
    """
    Preprocessing pipeline for credit score prediction.
    Transforms raw input features to model-ready format.
    """

    def __init__(self, scaler):  # <-- Add scaler as an argument
        """Initialize the preprocessor."""
        self.scaler = scaler  # <-- Store the scaler instance
        self.loan_types = [
            'mortgage_loan', 'home_equity_loan', 'debt_consolidation_loan',
            'credit-builder_loan', 'auto_loan', 'payday_loan',
            'not_specified', 'personal_loan', 'student_loan'
        ]

        self.occupations = [
            'architect', 'developer', 'doctor', 'engineer', 'entrepreneur',
            'journalist', 'lawyer', 'manager', 'mechanic', 'media_manager',
            'musician', 'scientist', 'teacher', 'writer'
        ]

        self.months = [
            'august', 'february', 'january', 'july', 'june',
            'march', 'may'
        ]

    def clean_and_standardize_text(self, text: str) -> str:
        """Clean and standardize text input."""
        if pd.isna(text) or text == '':
            return np.nan
        return str(text).lower().replace(' ', '_').strip()

    def parse_credit_history_age(self, credit_history: str) -> float:
        """Parse credit history age from text format to months."""
        if pd.isna(credit_history) or str(credit_history).lower() == 'na':
            return np.nan

        try:
            # Handle format like "22 Years and 1 Months"
            parts = str(credit_history).replace('_', ' ').split(' and ')
            years = int(parts[0].split(' ')[0]) * 12
            months = int(parts[1].split(' ')[0])
            return years + months
        except:
            return np.nan

    def process_loan_types(self, type_of_loan: str) -> Dict[str, int]:
        """Process loan types and create binary features."""
        loan_features = {f'has_{loan}': 0 for loan in self.loan_types}

        if pd.notna(type_of_loan):
            # Split loan types by various delimiters
            loan_list = str(type_of_loan).replace(
                '_and_', ', ').replace(' and ', ', ').split(', ')
            loan_list = [loan.strip().lower().replace(' ', '_')
                         for loan in loan_list]

            for loan in loan_list:
                feature_name = f'has_{loan}'
                if feature_name in loan_features:
                    loan_features[feature_name] = 1

        return loan_features

    def create_occupation_features(self, occupation: str) -> Dict[str, int]:
        """Create occupation dummy variables."""
        occupation_features = {
            f'occupation_{occ}': 0 for occ in self.occupations}

        if pd.notna(occupation):
            clean_occupation = self.clean_and_standardize_text(occupation)
            feature_name = f'occupation_{clean_occupation}'
            if feature_name in occupation_features:
                occupation_features[feature_name] = 1

        return occupation_features

    # def create_month_features(self, month: str) -> Dict[str, int]:
    #     """Create month dummy variables."""
    #     month_features = {f'month_{m}': 0 for m in self.months}

    #     if pd.notna(month):
    #         clean_month = self.clean_and_standardize_text(month)
    #         feature_name = f'month_{clean_month}'
    #         if feature_name in month_features:
    #             month_features[feature_name] = 1

    #     return month_features

    def create_credit_mix_features(self, credit_mix: str) -> Dict[str, int]:
        """Create credit mix dummy variables."""
        credit_mix_features = {'credit_mix_good': 0, 'credit_mix_standard': 0}

        if pd.notna(credit_mix):
            clean_credit_mix = self.clean_and_standardize_text(credit_mix)
            if clean_credit_mix == 'good':
                credit_mix_features['credit_mix_good'] = 1
            elif clean_credit_mix == 'standard':
                credit_mix_features['credit_mix_standard'] = 1

        return credit_mix_features

    def create_payment_features(self, payment_of_min_amount: str, payment_behaviour: str) -> Dict[str, int]:
        """Create payment-related features."""
        payment_features = {
            'payment_of_min_amount_yes': 0,
            'high_spent_medium_value_payments': 0,
            'high_spent_small_value_payments': 0,
            'low_spent_large_value_payments': 0,
            'low_spent_medium_value_payments': 0,
            'low_spent_small_value_payments': 0
        }

        # Payment of minimum amount
        if pd.notna(payment_of_min_amount):
            if self.clean_and_standardize_text(payment_of_min_amount) == 'yes':
                payment_features['payment_of_min_amount_yes'] = 1

        # Payment behaviour
        if pd.notna(payment_behaviour):
            clean_behaviour = self.clean_and_standardize_text(
                payment_behaviour)
            feature_name = clean_behaviour
            if feature_name in payment_features:
                payment_features[feature_name] = 1

        return payment_features

    def handle_outliers(self, value: float, column: str) -> float:
        """Handle outliers based on business logic."""
        if pd.isna(value):
            return value

        # Define reasonable ranges based on domain knowledge
        outlier_ranges = {
            'age': (18, 80),
            'annual_income': (0, 500000),
            'monthly_inhand_salary': (0, 50000),
            'num_bank_accounts': (0, 20),
            'num_credit_card': (0, 20),
            'interest_rate': (0, 50),
            'num_of_loan': (0, 20),
            'delay_from_due_date': (0, 60),
            'num_of_delayed_payment': (0, 50),
            'num_credit_inquiries': (0, 20),
            'outstanding_debt': (0, 200000),
            'credit_utilization_ratio': (0, 100),
            'total_emi_per_month': (0, 10000),
            'amount_invested_monthly': (0, 10000),
            'monthly_balance': (-5000, 20000)
        }

        if column in outlier_ranges:
            min_val, max_val = outlier_ranges[column]
            if value < min_val or value > max_val:
                # Return median value for outliers
                median_values = {
                    'age': 30,
                    'annual_income': 50000,
                    'monthly_inhand_salary': 4000,
                    'num_bank_accounts': 3,
                    'num_credit_card': 4,
                    'interest_rate': 15,
                    'num_of_loan': 2,
                    'delay_from_due_date': 15,
                    'num_of_delayed_payment': 5,
                    'num_credit_inquiries': 3,
                    'outstanding_debt': 20000,
                    'credit_utilization_ratio': 30,
                    'total_emi_per_month': 1000,
                    'amount_invested_monthly': 500,
                    'monthly_balance': 2000
                }
                return median_values.get(column, value)

        return value

    def transform_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Transform raw input data to model features."""

        # Initialize feature vector with expected feature names
        features = {}

        # Basic numerical features
        numerical_features = [
            'age', 'annual_income', 'monthly_inhand_salary', 'num_bank_accounts',
            'num_credit_card', 'interest_rate', 'num_of_loan', 'delay_from_due_date',
            'num_of_delayed_payment', 'changed_credit_limit', 'num_credit_inquiries',
            'outstanding_debt', 'credit_utilization_ratio', 'total_emi_per_month',
            'amount_invested_monthly', 'monthly_balance'
        ]

        for feature in numerical_features:
            value = input_data.get(feature, np.nan)
            if pd.notna(value):
                features[feature] = self.handle_outliers(float(value), feature)
            else:
                features[feature] = np.nan

        # Process credit history age
        credit_history_raw = input_data.get('credit_history_age', '')
        features['credit_history_age'] = self.parse_credit_history_age(
            credit_history_raw)

        # Process categorical features
        loan_features = self.process_loan_types(
            input_data.get('type_of_loan', ''))
        features.update(loan_features)

        # month_features = self.create_month_features(
        #     input_data.get('month', ''))
        # features.update(month_features)

        occupation_features = self.create_occupation_features(
            input_data.get('occupation', ''))
        features.update(occupation_features)

        credit_mix_features = self.create_credit_mix_features(
            input_data.get('credit_mix', ''))
        features.update(credit_mix_features)

        payment_features = self.create_payment_features(
            input_data.get('payment_of_min_amount', ''),
            input_data.get('payment_behaviour', '')
        )
        features.update(payment_features)

        # Fill missing numerical values with medians
        feature_medians = {
            'age': 30, 'annual_income': 50000, 'monthly_inhand_salary': 4000,
            'num_bank_accounts': 3, 'num_credit_card': 4, 'interest_rate': 15,
            'num_of_loan': 2, 'delay_from_due_date': 15, 'num_of_delayed_payment': 5,
            'changed_credit_limit': 0, 'num_credit_inquiries': 3, 'outstanding_debt': 20000,
            'credit_utilization_ratio': 30, 'credit_history_age': 240,
            'total_emi_per_month': 1000, 'amount_invested_monthly': 500,
            'monthly_balance': 2000
        }

        for feature, median_val in feature_medians.items():
            if pd.isna(features.get(feature)):
                features[feature] = median_val

        # Expected feature order (based on your final feature set)
        expected_features = [
            'age', 'annual_income', 'monthly_inhand_salary', 'num_bank_accounts',
            'num_credit_card', 'interest_rate', 'num_of_loan', 'delay_from_due_date',
            'num_of_delayed_payment', 'changed_credit_limit', 'num_credit_inquiries',
            'outstanding_debt', 'credit_utilization_ratio', 'credit_history_age',
            'total_emi_per_month', 'amount_invested_monthly', 'monthly_balance'
        ]

        # Add all loan type features
        expected_features.extend([f'has_{loan}' for loan in self.loan_types])

        # Add month features
        # expected_features.extend([f'month_{month}' for month in self.months])

        # Add occupation features
        expected_features.extend(
            [f'occupation_{occ}' for occ in self.occupations])

        # Add other categorical features
        expected_features.extend(['credit_mix_good', 'credit_mix_standard'])
        expected_features.extend(['payment_of_min_amount_yes'])
        expected_features.extend([
            'high_spent_medium_value_payments', 'high_spent_small_value_payments',
            'low_spent_large_value_payments',
            'low_spent_medium_value_payments',  'low_spent_small_value_payments'
        ])

        # Create feature array in expected order
        feature_array = []
        for feature_name in expected_features:
            feature_array.append(features.get(feature_name, 0))

        final_features = np.array(feature_array).reshape(1, -1)

        # 🚀 SCALING STEP: Apply the scaler to the numerical features
        if self.scaler:
            # Get the indices of the numerical features that need scaling
            numerical_feature_names = [
                'age', 'annual_income', 'monthly_inhand_salary', 'num_bank_accounts',
                'num_credit_card', 'interest_rate', 'num_of_loan', 'delay_from_due_date',
                'num_of_delayed_payment', 'changed_credit_limit', 'num_credit_inquiries',
                'outstanding_debt', 'credit_utilization_ratio', 'credit_history_age',
                'total_emi_per_month', 'amount_invested_monthly', 'monthly_balance'
            ]
            numerical_indices = [expected_features.index(
                feat) for feat in numerical_feature_names]

            # Apply scaling only to the numerical columns
            final_features[:, numerical_indices] = self.scaler.transform(
                final_features[:, numerical_indices])

        return final_features


def load_scaler():
    """Load the trained StandardScaler."""
    scaler = joblib.load("models/feature_scaler.joblib")
    return scaler


def load_model():
    """
    Load the trained model.
    """
    model = joblib.load("models/best_rf_model.joblib")
    return model


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
            month = st.selectbox("Month", [
                "January", "February", "March", "May", "June", "July", "August", "September", "October", "November", "December"
            ])
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

            payment_of_min_amount = st.selectbox(
                "Payment of Minimum Amount", ["Yes", "No"])
            credit_mix = st.selectbox(
                "Credit Mix", ["Good", "Standard", "Bad"])

        with col6:
            payment_behaviour = st.selectbox("Payment Behaviour", [
                "high_spent_large_value_payments", "high_spent_medium_value_payments", "high_spent_small_value_payments",
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
                    background-color: #f0f2f6; margin: 20px 0;'>
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


def main():
    """Main application function."""

    # Load model and scaler
    model = load_model()
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
                prediction = model.predict(features)[0]

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
