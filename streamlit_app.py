# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
    df = pd.read_excel("new treated slope.csv")
except:
    st.error("❌ Dataset not found. Please upload 'new treated slope.csv' to your repo.")
    st.stop()

# ----------------------------
# Features and Target
# ----------------------------
X = df[["Cohesion", "Friction_Angle", "Nail_Length",
        "Drillhole_Diameter", "Nail_Inclination", "Slope_Angle"]]
y = df["Factor_of_Safety"]

# ----------------------------
# Split Data 80/20
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Apply SMOGN to Training Data
# ----------------------------
train_df = pd.concat([X_train, y_train], axis=1)
train_bal = smogn.smoter(
    data=train_df,
    y="Factor_of_Safety"
)

X_train_bal = train_bal[X.columns]
y_train_bal = train_bal["Factor_of_Safety"]

# ----------------------------
# K-Fold Cross Validation
# ----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=kf, scoring="r2")

st.write(f"📊 10-Fold CV Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ----------------------------
# Train Final Model (on balanced data)
# ----------------------------
model.fit(X_train_bal, y_train_bal)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict FoS"):
    input_data = np.array([[c, phi, nail_length, nail_diameter, nail_inclination, slope_angle]])
    fos_pred = model.predict(input_data)[0]
    st.success(f"Predicted Factor of Safety (FoS): {fos_pred:.3f}")
