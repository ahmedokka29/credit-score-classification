FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY ["models/catboost_credit_model.pkl", "models/min_max_scaler.joblib", "./models/"]
COPY 'utils/predictions.py' ./utils/
COPY 'utils/preprocessing.py' ./utils/
COPY "app.py" .
# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]