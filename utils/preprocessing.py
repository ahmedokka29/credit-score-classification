"""
Credit Score Data Preprocessing Pipeline
======================================
This script handles comprehensive data cleaning, imputation, and feature engineering
for credit scoring dataset with temporal customer data.
"""
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, List, Union


warnings.filterwarnings('ignore')


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


"""
Credit Score Data Preprocessing Pipeline - Modified for Test Data
================================================================
This script handles comprehensive data cleaning, imputation, and feature engineering
for credit scoring dataset with temporal customer data.
Modified to handle test data with hard-coded dummy variables.
"""

warnings.filterwarnings('ignore')


class CreditDataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for credit scoring data.
    Modified to handle test data with hard-coded dummy variables.

    This class handles:
    - Data cleaning and corruption removal
    - Missing value imputation with customer-aware logic
    - Feature engineering and encoding with hard-coded dummy variables
    - Outlier detection and treatment
    """

    def __init__(self, file_path: str = None, dataframe: pd.DataFrame = None):
        """Initialize the preprocessor with either a file path or a DataFrame."""
        self.file_path = file_path
        self.df = dataframe
        self.is_training = False  # Flag to distinguish between training and test data

        # Hard-coded categorical values based on training data
        self.occupations = [
            'architect', 'developer', 'doctor', 'engineer',
            'entrepreneur', 'journalist', 'lawyer',
            'manager', 'mechanic', 'media_manager',
            'musician', 'scientist', 'teacher', 'writer'
        ]

        self.loan_types = [
            'credit-builder_loan', 'not_specified', 'home_equity_loan',
            'debt_consolidation_loan', 'mortgage_loan', 'auto_loan',
            'student_loan', 'personal_loan', 'payday_loan'
        ]

        # Hard-coded credit mix categories
        self.credit_mix_categories = ['bad', 'good', 'standard']

        # Hard-coded payment of min amount categories
        self.payment_min_categories = ['no', 'yes']

        # Hard-coded payment behaviour categories
        self.payment_behaviour_categories = [
            'high_spent_large_value_payments',
            'high_spent_medium_value_payments',
            'high_spent_small_value_payments',
            'low_spent_large_value_payments',
            'low_spent_medium_value_payments',
            'low_spent_small_value_payments'
        ]

        self.setup_display_options()

    def setup_display_options(self):
        """Configure pandas display options for better data inspection."""
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.float_format', lambda x: '%.5f' % x)

    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data preparation."""
        print("Loading data...")

        if self.df is not None:
            print(f"Data already provided: {self.df.shape}")
        elif self.file_path is not None:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded from file: {self.df.shape}")
        else:
            raise ValueError("Either file_path or dataframe must be provided.")

        self.df.columns = self.df.columns.str.lower()
        return self.df

    def remove_irrelevant_columns(self) -> pd.DataFrame:
        """Remove columns that don't contribute to credit scoring."""
        columns_to_drop = ['id', 'name', 'ssn', 'month']
        existing_columns = [
            col for col in columns_to_drop if col in self.df.columns]
        if existing_columns:
            self.df = self.df.drop(existing_columns, axis=1)
            print(f"Dropped irrelevant columns: {existing_columns}")
        return self.df

    def clean_corrupted_data(self) -> pd.DataFrame:
        """Remove corrupted and invalid data patterns found in the dataset."""
        print("Cleaning corrupted data...")

        # Handle specific extreme corrupted values
        extreme_value_mapping = {
            '__-333333333333333333333333333__': np.nan,
            '__10000__': np.nan
        }
        self.df.replace(extreme_value_mapping, inplace=True)

        # Define invalid patterns found in the data
        invalid_patterns = ['', 'nan', '!@9#%8', '#F%$D@*&8', 'NM', 'nm']

        # Strip underscores and replace invalid patterns
        self.df = self.df.applymap(
            lambda x: x if x is np.nan or not isinstance(x, str)
            else str(x).strip('_')
        ).replace(invalid_patterns, np.nan)

        print("Corrupted data patterns cleaned")
        return self.df

    def convert_data_types(self) -> pd.DataFrame:
        """Convert columns to appropriate data types after cleaning."""
        print("Converting data types...")

        # Numeric columns that should be converted
        numeric_conversions = {
            'age': int,
            'annual_income': float,
            'num_of_loan': int,
            'num_of_delayed_payment': float,
            'changed_credit_limit': float,
            'outstanding_debt': float,
            'amount_invested_monthly': float,
            'monthly_balance': float
        }

        for col, dtype in numeric_conversions.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                except ValueError as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")

        return self.df

    def standardize_string_columns(self) -> pd.DataFrame:
        """Standardize string columns to lowercase with underscores."""
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)

        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

        print(f"Standardized {len(string_columns)} string columns")
        return self.df

    def handle_customer_stable_features(self) -> pd.DataFrame:
        """Handle features that should be stable within each customer's timeline."""
        print("Handling customer-stable features...")

        stable_features = ['occupation']

        for feature in stable_features:
            if feature in self.df.columns and 'customer_id' in self.df.columns:
                # Forward fill then backward fill within customer groups
                self.df[feature] = self.df.groupby(
                    'customer_id')[feature].fillna(method='ffill')
                self.df[feature] = self.df.groupby(
                    'customer_id')[feature].fillna(method='bfill')

        return self.df

    def clean_age_column(self) -> pd.DataFrame:
        """Clean age column with business logic validation."""
        print("Cleaning age column...")

        if 'age' in self.df.columns:
            # Set unrealistic ages to NaN (based on data analysis: 14-60 range)
            self.df.loc[(self.df['age'] < 14) | (
                self.df['age'] > 60), 'age'] = np.nan

            # Fill missing ages within customer groups if customer_id exists
            if 'customer_id' in self.df.columns:
                self.df['age'] = self.df.groupby('customer_id')['age'].fillna(
                    method='ffill').fillna(method='bfill')
            else:
                # For test data without customer_id, use median
                self.df['age'].fillna(self.df['age'].median(), inplace=True)

        return self.df

    def parse_credit_history_age(self, x) -> Union[float, int]:
        """Parse credit history age from text format to months."""
        if pd.isna(x) or str(x).lower() == 'na':
            return np.nan

        try:
            parts = str(x).replace('_', ' ').split(' and ')
            years = int(parts[0].split(' ')[0]) * 12
            months = int(parts[1].split(' ')[0])
            return years + months
        except:
            return np.nan

    def fill_credit_history_sequential(self, group) -> pd.Series:
        """Fill missing credit history values considering monthly progression."""
        group = group.copy()

        # Forward fill with increment
        for i in range(1, len(group)):
            if pd.isna(group.iloc[i]) and not pd.isna(group.iloc[i-1]):
                group.iloc[i] = group.iloc[i-1] + 1

        # Backward fill with decrement
        for i in range(len(group)-2, -1, -1):
            if pd.isna(group.iloc[i]) and not pd.isna(group.iloc[i+1]):
                group.iloc[i] = group.iloc[i+1] - 1

        return group

    def handle_credit_history_age(self) -> pd.DataFrame:
        """Transform and impute credit history age column."""
        print("Processing credit history age...")

        if 'credit_history_age' in self.df.columns:
            # Replace 'na' strings with proper NaN
            self.df['credit_history_age'] = self.df['credit_history_age'].replace({
                                                                                  'na': np.nan})

            # Parse text format to numeric (months)
            self.df['credit_history_age'] = self.df['credit_history_age'].apply(
                self.parse_credit_history_age)

            # Apply sequential filling within customer groups if customer_id exists
            if 'customer_id' in self.df.columns:
                self.df['credit_history_age'] = self.df.groupby('customer_id')['credit_history_age'].apply(
                    self.fill_credit_history_sequential
                ).reset_index(level=0, drop=True)
            else:
                # For test data without customer_id, use median
                self.df['credit_history_age'].fillna(
                    240, inplace=True)  # Default median

        return self.df

    def impute_customer_grouped_features(self) -> pd.DataFrame:
        """Impute missing values using customer-grouped statistics."""
        print("Imputing customer-grouped features...")

        # Features that should be filled within customer groups
        customer_grouped_features = {
            'monthly_inhand_salary': 'ffill_bfill',
            'credit_mix': 'ffill_bfill',
            'payment_of_min_amount': 'mode',
            'payment_behaviour': 'mode_safe',
            'num_of_delayed_payment': 'median',
            'changed_credit_limit': 'median'
        }

        for feature, method in customer_grouped_features.items():
            if feature not in self.df.columns:
                continue

            if 'customer_id' in self.df.columns:
                if method == 'ffill_bfill':
                    self.df[feature] = self.df.groupby(
                        'customer_id')[feature].fillna(method='ffill')
                    self.df[feature] = self.df.groupby(
                        'customer_id')[feature].fillna(method='bfill')

                elif method == 'mode':
                    self.df[feature] = self.df.groupby('customer_id')[feature].transform(
                        lambda x: x.mode()[0] if not x.mode().empty else np.nan
                    )

                elif method == 'mode_safe':
                    self.df[feature] = self.df.groupby('customer_id')[feature].transform(
                        lambda x: x.fillna(
                            x.mode()[0] if not x.mode().empty else 'unknown')
                    )

                elif method == 'median':
                    self.df[feature] = self.df.groupby('customer_id')[feature].transform(
                        lambda x: x.median() if not x.isnull().all() else np.nan
                    )
            else:
                # For test data without customer_id, use global imputation
                if method in ['ffill_bfill', 'mode', 'mode_safe']:
                    if feature == 'payment_behaviour':
                        self.df[feature].fillna('unknown', inplace=True)
                    elif feature in ['credit_mix', 'payment_of_min_amount']:
                        mode_val = self.df[feature].mode(
                        )[0] if not self.df[feature].mode().empty else 'unknown'
                        self.df[feature].fillna(mode_val, inplace=True)
                    else:
                        self.df[feature].fillna(method='ffill', inplace=True)
                        self.df[feature].fillna(method='bfill', inplace=True)
                elif method == 'median':
                    median_val = self.df[feature].median()
                    self.df[feature].fillna(median_val, inplace=True)

        return self.df

    def handle_remaining_missing_values(self) -> pd.DataFrame:
        """Handle remaining missing values with appropriate strategies."""
        print("Handling remaining missing values...")

        # Mean imputation for balance-related features
        mean_imputation_cols = ['monthly_balance', 'amount_invested_monthly']
        for col in mean_imputation_cols:
            if col in self.df.columns:
                if 'customer_id' in self.df.columns:
                    self.df[col] = self.df.groupby('customer_id')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Median imputation for count-based features with zero handling
        median_imputation_cols = ['num_of_loan', 'num_credit_inquiries',
                                  'num_bank_accounts', 'total_emi_per_month']

        for col in median_imputation_cols:
            if col in self.df.columns:
                if 'customer_id' in self.df.columns:
                    # Customer-level median first
                    self.df[col] = self.df.groupby('customer_id')[col].transform(
                        lambda x: x.median() if not x.isnull().all() else np.nan
                    )

                # Replace invalid zeros with NaN, then global median imputation
                self.df[col] = self.df[col].replace(0, np.nan)
                self.df[col].fillna(self.df[col].median(), inplace=True)

                # Convert to integer for count-based features
                if col in ['num_of_loan', 'num_credit_inquiries', 'num_bank_accounts']:
                    self.df[col] = self.df[col].astype(int)

        return self.df

    def engineer_loan_features(self) -> pd.DataFrame:
        """Create binary features for different loan types - Hard-coded for test data."""
        print("Engineering loan type features...")

        # Create binary features for each loan type
        for loan_type in self.loan_types:
            feature_name = f'has_{loan_type}'

            if 'type_of_loan' in self.df.columns:
                # Check if customer has this loan type
                self.df[feature_name] = self.df['type_of_loan'].apply(
                    lambda x: 1 if pd.notna(x) and loan_type in str(x) else 0
                )
            else:
                # If type_of_loan column doesn't exist, set all to 0
                self.df[feature_name] = 0

        return self.df

    def create_occupation_features(self) -> pd.DataFrame:
        """Create hard-coded occupation dummy variables."""
        print("Creating occupation features...")

        # Create binary features for each occupation
        for occupation in self.occupations:
            feature_name = f'occupation_{occupation}'

            if 'occupation' in self.df.columns:
                self.df[feature_name] = self.df['occupation'].apply(
                    lambda x: 1 if pd.notna(x) and str(
                        x).lower().replace(' ', '_') == occupation else 0
                )
            else:
                # If occupation column doesn't exist, set all to 0
                self.df[feature_name] = 0

        return self.df

    def create_hardcoded_dummy_variables(self) -> pd.DataFrame:
        """Create hard-coded dummy variables for categorical features."""
        print("Creating hard-coded dummy variables...")

        # Credit Mix features (drop first: 'bad' is the reference)
        if 'credit_mix' in self.df.columns:
            self.df['credit_mix_good'] = self.df['credit_mix'].apply(
                lambda x: 1 if pd.notna(x) and str(x).lower() == 'good' else 0
            )
            self.df['credit_mix_standard'] = self.df['credit_mix'].apply(
                lambda x: 1 if pd.notna(x) and str(
                    x).lower() == 'standard' else 0
            )
        else:
            self.df['credit_mix_good'] = 0
            self.df['credit_mix_standard'] = 0

        # Payment of minimum amount features (drop first: 'no' is the reference)
        if 'payment_of_min_amount' in self.df.columns:
            self.df['payment_of_min_amount_yes'] = self.df['payment_of_min_amount'].apply(
                lambda x: 1 if pd.notna(x) and str(x).lower() == 'yes' else 0
            )
        else:
            self.df['payment_of_min_amount_yes'] = 0

        # Payment behaviour features (drop first: 'high_spent_large_value_payments' is the reference)
        payment_behaviour_features = [
            'high_spent_medium_value_payments',
            'high_spent_small_value_payments',
            'low_spent_large_value_payments',
            'low_spent_medium_value_payments',
            'low_spent_small_value_payments'
        ]

        if 'payment_behaviour' in self.df.columns:
            for feature in payment_behaviour_features:
                self.df[feature] = self.df['payment_behaviour'].apply(
                    lambda x: 1 if pd.notna(x) and str(
                        x).lower().replace(' ', '_') == feature else 0
                )
        else:
            for feature in payment_behaviour_features:
                self.df[feature] = 0

        return self.df

    def drop_original_categorical_columns(self) -> pd.DataFrame:
        """Drop original categorical columns after creating dummy variables."""
        columns_to_drop = [
            'customer_id', 'type_of_loan', 'occupation', 'credit_mix',
            'payment_of_min_amount', 'payment_behaviour'
        ]

        existing_columns_to_drop = [
            col for col in columns_to_drop if col in self.df.columns]
        if existing_columns_to_drop:
            self.df = self.df.drop(existing_columns_to_drop, axis=1)
            print(
                f"Dropped original categorical columns: {existing_columns_to_drop}")

        return self.df

    def detect_and_treat_outliers(self, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Detect and treat outliers in specified columns."""
        print(f"Treating outliers in {len(columns)} columns...")

        for col in columns:
            if col not in self.df.columns:
                continue

            # Calculate IQR bounds
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Count outliers
            outliers_count = len(self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ])

            if outliers_count > 0:
                print(f"{col}: {outliers_count} outliers detected")

                # Treat outliers based on column characteristics
                if col == 'total_emi_per_month':
                    # Use median of non-outliers for EMI
                    non_outlier_median = self.df[
                        (self.df[col] >= lower_bound) & (
                            self.df[col] <= upper_bound)
                    ][col].median()

                    self.df.loc[self.df[col] > upper_bound,
                                col] = non_outlier_median
                else:
                    # Use median for other features
                    if 'customer_id' in self.df.columns:
                        outlier_mask = (self.df[col] > upper_bound) | (
                            self.df[col] < lower_bound)
                        self.df.loc[outlier_mask, col] = self.df.groupby('customer_id')[col].transform(
                            lambda x: x.mode()[0] if not x.mode(
                            ).empty else x.median()
                        )[outlier_mask]
                    else:
                        # For test data without customer_id
                        outlier_mask = (self.df[col] > upper_bound) | (
                            self.df[col] < lower_bound)
                        self.df.loc[outlier_mask, col] = self.df[col].median()

        return self.df

    def generate_data_summary(self) -> pd.DataFrame:
        """Generate final data summary and statistics."""
        print("\n" + "="*50)
        print("DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Missing values per column:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values remaining!")

        # print(f"\nData types:")
        # print(self.df.dtypes.value_counts())

        # print(f"\nColumn names:")
        # print(list(self.df.columns))

        return self.df.describe()

    def run_full_pipeline(self) -> pd.DataFrame:
        """Execute the complete preprocessing pipeline."""
        print("Starting Credit Score Data Preprocessing Pipeline...")
        print("="*60)

        # Load and initial cleaning
        self.load_data()
        self.remove_irrelevant_columns()
        self.clean_corrupted_data()
        self.convert_data_types()
        self.standardize_string_columns()

        # Handle missing values with domain knowledge
        self.handle_customer_stable_features()
        self.clean_age_column()
        self.handle_credit_history_age()
        self.impute_customer_grouped_features()
        self.handle_remaining_missing_values()

        # Feature engineering with hard-coded dummy variables
        self.engineer_loan_features()
        self.create_occupation_features()
        self.create_hardcoded_dummy_variables()

        # Outlier treatment for key numerical columns
        numerical_columns = [
            'num_credit_card', 'interest_rate', 'num_credit_inquiries',
            'annual_income', 'total_emi_per_month'
        ]
        # Only treat outliers for existing columns
        existing_numerical_columns = [
            col for col in numerical_columns if col in self.df.columns]
        if existing_numerical_columns:
            self.detect_and_treat_outliers(existing_numerical_columns)

        # Final cleanup
        self.drop_original_categorical_columns()

        # Generate summary
        summary = self.generate_data_summary()
        print("\nPreprocessing pipeline completed successfully!")
        return self.df
