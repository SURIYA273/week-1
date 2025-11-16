# streamlit_app.py
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="EV Range Predictor", layout="centered", page_icon="🚗")

MODEL_PATH = "models/ev_range_predictor.joblib"
META_PATH = "models/metadata.json"
CLEANED_CSV = "electric_vehicles_cleaned_data_2025.csv"  # optional

st.title("🚗 EV Range Predictor")
st.write("Predict the driving range (km) of an electric vehicle using its specs.")

# --- Load model & metadata ---
@st.cache_resource
def load_model_and_meta():
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model not found. Train model and save to models/ev_range_predictor.joblib"

    if not os.path.exists(META_PATH):
        return None, None, "Metadata not found. Save models/metadata.json"

    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    return model, meta, None


model, meta, err = load_model_and_meta()
if err:
    st.error(err)
    st.stop()

FEATURES = meta["features"]
TARGET = meta.get("target", "range_km")

# --- Sidebar ---
st.sidebar.header("Input mode")
mode = st.sidebar.radio("Choose input mode", ("Single input (form)", "Batch input (CSV upload)"))


# Show sample dataset
if os.path.exists(CLEANED_CSV):
    if st.checkbox("Show sample of the cleaned dataset"):
        sample_df = pd.read_csv(CLEANED_CSV)
        st.dataframe(sample_df.head(8))


# --- Single Input Form ---
def single_input_form():
    st.subheader("Enter vehicle features")

    defaults = {
        'top_speed_kmh': 180,
        'battery_capacity_kWh': 75,
        'number_of_cells': 96,
        'torque_nm': 450,
        'efficiency_wh_per_km': 180,
        'acceleration_0_100_s': 6.5,
        'fast_charging_power_kw_dc': 150,
        'length_mm': 4680,
        'width_mm': 1840,
        'height_mm': 1440
    }

    inputs = {}

    with st.form("single_form"):
        for f in FEATURES:
            default = defaults.get(f, 0.0)
            inputs[f] = st.number_input(label=f, value=float(default))
        submitted = st.form_submit_button("Predict range")

    return inputs if submitted else None


# --- Batch Input ---
def batch_input():
    st.subheader("Upload CSV for batch predictions")
    st.write("CSV must contain these columns:")
    st.write(FEATURES)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        # Validate required columns
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return None

        df_proc = df[FEATURES].apply(pd.to_numeric, errors="coerce")

        if df_proc.isna().any().any():
            st.warning("Some values couldn't be converted to numeric and will be imputed.")

        if st.button("Predict for uploaded CSV"):
            return df_proc

    return None


# --- Run selected mode ---
user_input = None
if mode == "Single input (form)":
    user_input = single_input_form()
elif mode == "Batch input (CSV upload)":
    user_input = batch_input()


# --- Prediction ---
if user_input is not None:

    # --- Single input prediction ---
    if isinstance(user_input, dict):
        X = np.array([float(user_input[f]) for f in FEATURES]).reshape(1, -1)
        pred = model.predict(X)[0]

        st.success(f"✅ Predicted driving range: **{pred:.2f} km**")

        # Feature Importances
        try:
            rf_model = model.named_steps.get("rf", None)
            if hasattr(rf_model, "feature_importances_"):
                fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
                st.subheader("Feature Importances")
                st.bar_chart(fi)
        except Exception:
            pass

    # --- Batch prediction ---
    else:
        df_in = user_input
        preds = model.predict(df_in.fillna(df_in.median()))

        out = df_in.copy()
        out[TARGET] = preds

        st.success("✅ Batch predictions complete")
        st.dataframe(out.head(20))

        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, "ev_predictions.csv", mime="text/csv")


# --- Evaluation on cleaned CSV ---
st.markdown("---")
st.subheader("Quick model performance check")

if os.path.exists(CLEANED_CSV):
    df_all = pd.read_csv(CLEANED_CSV)

    if set(FEATURES + [TARGET]).issubset(df_all.columns):
        df_eval = df_all[FEATURES + [TARGET]].dropna()

        if len(df_eval) >= 10:
            sample = df_eval.sample(frac=0.2, random_state=42)

            X_eval = sample[FEATURES].apply(pd.to_numeric, errors="coerce")
            y_eval = sample[TARGET].apply(pd.to_numeric, errors="coerce")

            y_pred = model.predict(X_eval.fillna(X_eval.median()))
            rmse = (mean_squared_error(y_eval, y_pred)) ** 0.5

            st.write(f"Sample RMSE: **{rmse:.2f} km** on {len(sample)} rows")

        else:
            st.info("Not enough data to compute evaluation.")
    else:
        st.info("Cleaned CSV exists, but required columns are missing.")
else:
    st.info("No cleaned dataset found. Skipping evaluation.")

st.markdown("---")
st.write("Use realistic values for best predictions.")
