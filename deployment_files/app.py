"""
Streamlit app for Abalone Rings prediction.
Run with: uv run streamlit run app.py
Requires abalone_model.joblib and feature_columns.joblib (run the notebook first).
"""

import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "abalone_model.joblib"
FEATURES_PATH = "feature_columns.joblib"


def load_model_and_features():
    """Load the trained pipeline and feature column list."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return pipeline, feature_columns


def build_input_row(length, diameter, height, whole_weight, shucked_weight,
                   viscera_weight, shell_weight, sex, feature_columns):
    """Build a single row DataFrame in the order expected by the pipeline."""
    sex_m = 1 if sex == "M" else 0
    sex_i = 1 if sex == "I" else 0
    numeric_values = [length, diameter, height, whole_weight, shucked_weight,
                     viscera_weight, shell_weight]
    row = {}
    idx = 0
    for col in feature_columns:
        if col == "Sex_M":
            row[col] = sex_m
        elif col == "Sex_I":
            row[col] = sex_i
        else:
            row[col] = numeric_values[idx]
            idx += 1
    return pd.DataFrame([row], columns=feature_columns)


def main():
    st.title("Abalone Rings Predictor")
    st.markdown("Predict the number of rings (age proxy) from physical measurements. "
                "Uses the pre-trained model.")

    pipeline, feature_columns = load_model_and_features()
    if pipeline is None or feature_columns is None:
        st.error("Model not found. Run the Jupyter notebook to train and save the model first "
                 f"(create {MODEL_PATH} and {FEATURES_PATH} in this folder).")
        return

    with st.form("inputs"):
        st.subheader("Input features")
        length = st.number_input("Length (mm)", min_value=0.0, value=0.5, step=0.01)
        diameter = st.number_input("Diameter (mm)", min_value=0.0, value=0.4, step=0.01)
        height = st.number_input("Height (mm)", min_value=0.0, value=0.15, step=0.01)
        whole_weight = st.number_input("Whole weight (g)", min_value=0.0, value=1.0, step=0.01)
        shucked_weight = st.number_input("Shucked weight (g)", min_value=0.0, value=0.4, step=0.01)
        viscera_weight = st.number_input("Viscera weight (g)", min_value=0.0, value=0.2, step=0.01)
        shell_weight = st.number_input("Shell weight (g)", min_value=0.0, value=0.3, step=0.01)
        sex = st.selectbox("Sex", options=["M", "F", "I"], index=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row_df = build_input_row(
            length, diameter, height, whole_weight, shucked_weight,
            viscera_weight, shell_weight, sex, feature_columns
        )
        if row_df is not None:
            pred = pipeline.predict(row_df)
            pred_value = float(pred.squeeze())
            st.success(f"**Predicted number of rings:** {pred_value:.2f}")
            st.caption("Rings + 1.5 approximates age in years.")


if __name__ == "__main__":
    main()
