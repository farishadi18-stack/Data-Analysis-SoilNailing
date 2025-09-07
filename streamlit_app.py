# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import smogn

# ----------------------------
# Title
# ----------------------------
st.title("Preliminary Slope Stability Assessment Tool")
st.write("Estimate Factor of Safety (FoS) using soil shear strength and nail parameters.")

# ----------------------------
# User Input
# ----------------------------
c = st.number_input("Cohesion (kPa)", min_value=0.0, step=0.5)
phi = st.number_input("Friction Angle (°)", min_value=0.0, max_value=60.0, step=0.5)
nail_length = st.number_input("Nail Length (m)", min_value=3.0, step=1.0)
nail_diameter = st.number_input("Drillhole Diameter (mm)", min_value=50.0, step=5.0)
nail_inclination = st.number_input("Inclination (°)", min_value=0.0, max_value=30.0, step=1.0)
slope_angle = st.number_input("Slope Angle (°)", min_value=15.0, max_value=90.0, step=1.0)

# ----------------------------
# Load Dataset
# ----------------------------
try:
    df = pd.read_csv("new treated slope.csv")
    st.success("✅ Dataset loaded successfully!")
    st.write("Preview:", df.head())
    st.write("Columns detected:", df.columns.tolist())
except Exception as e:
    st.error(f"❌ Dataset not found or could not be read. Error: {e}")
    df = None

# ----------------------------
# Train Model (if dataset available)
# ----------------------------
model = None
if df is not None:
    try:
        X = df[["Cohesion", "Friction_Angle", "Nail_Length",
                "Drillhole_Diameter", "Nail_Inclination", "Slope_Angle"]]
        y = df["Factor_of_Safety"]

        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply SMOGN
        train_df = pd.concat([X_train, y_train], axis=1)
        train_bal = smogn.smoter(data=train_df, y="Factor_of_Safety")

        X_train_bal = train_bal[X.columns]
        y_train_bal = train_bal["Factor_of_Safety"]

        # Train final model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_bal, y_train_bal)

        st.info("✅ Model trained successfully!")

    except Exception as e:
        st.error(f"⚠️ Training failed. Please check your dataset columns. Error: {e}")
        model = None

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("🔮 Predict FoS"):
    if model is not None:
        input_data = np.array([[c, phi, nail_length, nail_diameter, nail_inclination, slope_angle]])
        fos_pred = model.predict(input_data)[0]
        st.success(f"Predicted Factor of Safety (FoS): {fos_pred:.3f}")
    else:
        st.warning("⚠️ Model is not available. Please check dataset and retrain.")
