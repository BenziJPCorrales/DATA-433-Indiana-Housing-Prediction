"""
Gradio app for Abalone Rings prediction.
Run with: uv run python gradio_app.py
Requires abalone_model.joblib and feature_columns.joblib (run the notebook first).
"""

import gradio as gr
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


def predict(length, diameter, height, whole_weight, shucked_weight,
            viscera_weight, shell_weight, sex):
    """Gradio prediction function."""
    pipeline, feature_columns = load_model_and_features()
    if pipeline is None or feature_columns is None:
        return (
            "Error: Model not found. Run the Jupyter notebook to train and save "
            f"the model first (create {MODEL_PATH} and {FEATURES_PATH} in this folder)."
        )
    row_df = build_input_row(
        length, diameter, height, whole_weight, shucked_weight,
        viscera_weight, shell_weight, sex, feature_columns
    )
    pred = pipeline.predict(row_df)
    pred_value = float(pred.squeeze())
    return f"Predicted number of rings: {pred_value:.2f} (Rings + 1.5 ≈ age in years)"


def main():
    pipeline, feature_columns = load_model_and_features()
    if pipeline is None or feature_columns is None:
        # Still launch the app so the user sees the error in the output
        def _predict_err(*args):
            return (
                "Error: Model not found. Run the Jupyter notebook to train and save "
                f"the model first (create {MODEL_PATH} and {FEATURES_PATH} in this folder)."
            )
        iface = gr.Interface(
            fn=_predict_err,
            inputs=[
                gr.Number(label="Length (mm)", value=0.5),
                gr.Number(label="Diameter (mm)", value=0.4),
                gr.Number(label="Height (mm)", value=0.15),
                gr.Number(label="Whole weight (g)", value=1.0),
                gr.Number(label="Shucked weight (g)", value=0.4),
                gr.Number(label="Viscera weight (g)", value=0.2),
                gr.Number(label="Shell weight (g)", value=0.3),
                gr.Dropdown(choices=["M", "F", "I"], value="M", label="Sex"),
            ],
            outputs=gr.Textbox(label="Prediction"),
            title="Abalone Rings Predictor",
            description="Predict the number of rings from physical measurements.",
        )
    else:
        iface = gr.Interface(
            fn=predict,
            inputs=[
                gr.Number(label="Length (mm)", value=0.5),
                gr.Number(label="Diameter (mm)", value=0.4),
                gr.Number(label="Height (mm)", value=0.15),
                gr.Number(label="Whole weight (g)", value=1.0),
                gr.Number(label="Shucked weight (g)", value=0.4),
                gr.Number(label="Viscera weight (g)", value=0.2),
                gr.Number(label="Shell weight (g)", value=0.3),
                gr.Dropdown(choices=["M", "F", "I"], value="M", label="Sex"),
            ],
            outputs=gr.Textbox(label="Prediction"),
            title="Abalone Rings Predictor",
            description="Predict the number of rings from physical measurements.",
        )
    iface.launch()


if __name__ == "__main__":
    main()
