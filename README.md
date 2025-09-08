# Credit Score Classification Project

## Problem Description

A collected dataset from a basic bank contains extensive credit-related information. The management wants to build an intelligent system to segregate customers into credit score brackets, thus reducing manual efforts.

## Goal

Develop a robust and accurate predictive model for credit scores, categorizing customers into 'Good', 'Standard', or 'Poor'.

## Data Overview

- **Data Source:** Synthetic data that mimics a financial banking system.
- **Files:**
  - `train.csv` (28 columns): Contains 100,000 unique entries from 12,500 customers, with 8 monthly records (January to August).
  - `test.csv` (27 columns): Contains 50,000 unique entries from 12,500 customers, with 4 monthly records (September to December).
- **Original Columns:** Include details such as `id`, `customer_id`, `month`, `name`, `age`, `ssn`, `occupation`, `annual_income`, `monthly_inhand_salary`, `num_bank_accounts`, `num_credit_card`, `interest_rate`, `num_of_loan`, `type_of_loan`, `delay_from_due_date`, `num_of_delayed_payment`, `changed_credit_limit`, `num_credit_inquiries`, `credit_mix`, `outstanding_debt`, `credit_utilization_ratio`, `credit_history_age`, `payment_of_min_amount`, `total_emi_per_month`, `amount_invested_monthly`, `payment_behaviour`, and `monthly_balance`.
- **Target Column:** `credit_score`

## Data Cleaning

Each column features its unique set of challenges and noise:

- **General Cleaning:**

  - Removal of extraneous underscores and replacement of invalid patterns (e.g., `__10000__`, `!@9#%8`, etc.) with `np.nan`.
  - Conversion of mis-assigned data types from object to numeric types (int or float).

- **Age Column:**

  - Outliers managed by capping age values between 14 and 60. Values outside this range are set to `np.nan` and then filled using forward/backward fill based on `customer_id`.

- **Occupation & Payment Behaviour:**

  - Missing values in `occupation` are imputed by grouping by `customer_id` and applying forward/backward fill.
  - For `payment_behaviour`, missing values are filled with the most frequent value within each customer group.

- **Credit History Age:**

  - Values like "22_years_and_4_months" are processed by extracting and converting years and months to total months. Missing values are then filled by incrementing the previous or decrementing the next valid value.

## Data Encoding

- **Categorical Variables:**

  - Columns such as `occupation`, `credit_mix`, `payment_of_min_amount`, and `payment_behaviour` are encoded using one-hot encoding (similar to get dummies).

- **Target Variable:**

  - `credit_score` is mapped to numerical values: Poor = 0, Good = 1, Standard = 2.

- **Loan Type:**

  - The column `type_of_loan` may include multiple entries per record. These are split into individual components and manually encoded into dummy columns, for example:
    - `has_not_specified`
    - `has_credit-builder_loan`
    - `has_student_loan`
    - `has_mortgage_loan`
    - `has_debt_consolidation_loan`
    - `has_auto_loan`
    - `has_payday_loan`
    - `has_personal_loan`
    - `has_home_equity_loan`

## Data Splitting and Modeling

- **Data Splitting:**

  - Splitting is performed carefully based on `customer_id` to prevent data leakage between the training and testing datasets.

- **Modeling Approaches:**

  - The project leverages advanced techniques such as StratifiedKFold and RandomizedSearchCV, focusing on the F1 score as the primary evaluation metric.
  - Models employed include:
    - Random Forest
    - XGBoost
    - Balanced Random Forest
    - Balanced Bagging Random Forest
    - Balanced Bagging XGBoost
    - CatBoost

- **Handling Imbalanced Data:**

  - Instead of relying solely on oversampling techniques such as SMOTE, the project uses models that are inherently capable of handling imbalanced datasets.

## Model Tuning & Evaluation

The following details provide an overview of the model tuning and evaluation for various classifiers used in the project:

### CatBoost Model

- **Performance Summary:**
  - Standard F1-Score: 0.6902
  - Tuned F1-Score: 0.7000
  - Improvement: 0.0098

- **Best Parameters:** { 'colsample_bylevel': 0.8688542189623514, 'learning_rate': 0.16232392306574353, 'max_depth': 5, 'max_leaves': 32, 'min_data_in_leaf': 6, 'n_estimators': 379, 'reg_lambda': 2.351503172230192, 'subsample': 0.9933692563579372 }

- **Optimal Thresholds:**
  - Class 0: 0.5286 (F1: 0.7685)
  - Class 1: 0.6223 (F1: 0.7410)
  - Class 2: 0.2306 (F1: 0.7948)

### Balanced Bagging XGBoost

- Weighted F1 Score: 0.6947
- Macro F1 Score: 0.6859

### Balanced Bagging Random Forest

- Weighted F1 Score: 0.6959
- Macro F1 Score: 0.6870

### Balanced Random Forest

- Weighted F1 Score: 0.6964
- Macro F1 Score: 0.6873

### XGBoost

- Weighted F1 Score: 0.6962
- Macro F1 Score: 0.6706

### Random Forest

- Weighted F1 Score: 0.7050
- Macro F1 Score: 0.6909

## Project Structure

- **Notebooks:**

  - Contains exploratory and analysis notebooks (e.g., `notebook.ipynb`, `EDA.ipynb`).

- **Utilities:**

  - `utils/preprocessing.py` and `utils/predictions.py` provide scripts for data preparation and generating predictions.

- **Application Entry Point:**

  - `app.py` serves as the main application script.

- **Dependencies:**

  - Listed in `requirements.txt` for easy package installation.

- **Containerization:**

  - The `Dockerfile` provides instructions for building a containerized version of the application.

## Installation and Usage

### Prerequisites

Make sure you have Python installed along with the necessary packages. You can install the required packages using:

```sh
pip install -r requirements.txt
```

### Running the Application

To run the application, execute:

```sh
python app.py
```

### Docker Deployment

Build the Docker image using:

```sh
docker build -t credit-score-classification .
```

Run the Docker container with:

```sh
docker run -p 5000:5000 credit-score-classification
```

## Conclusion

This project presents a comprehensive approach to credit score classification, combining meticulous data cleaning, thoughtful encoding, and robust modeling techniques. The system is designed to efficiently manage and predict customer credit scores while addressing the challenges of imbalanced datasets. Explore the included notebooks for a detailed walkthrough of the analysis and modeling process.
