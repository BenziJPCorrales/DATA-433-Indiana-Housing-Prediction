# Abalone Streamlit app - run after training the model in the notebook
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App and model artifact (must exist: run notebook first or copy abalone_model.joblib + feature_columns.joblib)
COPY app.py .
COPY abalone_model.joblib .
COPY feature_columns.joblib .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
