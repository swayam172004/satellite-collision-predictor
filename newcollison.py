# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 20:35:14 2025

@author: swaya
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
from skyfield.api import EarthSatellite, load, utc
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
MODEL_FILE = "collision_predictor_model.pkl"
NUM_SATELLITES = 5
FORCED_COLLISIONS = [(0, 1, 20), (2, 3, 40)]
DURATION_MINUTES = 60
COLLISION_THRESHOLD_KM = 9000
NOISE_LEVEL_KM = 0.2

# -----------------------------
# Train model if not exists
# -----------------------------
if not os.path.exists(MODEL_FILE):
    with open('gp.txt') as file:
        lines = file.readlines()

    ts = load.timescale()
    start = datetime.now().replace(tzinfo=utc)
    times = ts.utc(start.year, start.month, start.day, start.hour,
                   start.minute + np.arange(0, DURATION_MINUTES, 1))

    # Load satellites
    satellites = []
    for i in range(0, NUM_SATELLITES * 3, 3):
        if i + 2 >= len(lines):
            break
        name = lines[i].strip()
        tle1 = lines[i+1].strip()
        tle2 = lines[i+2].strip()
        sat = EarthSatellite(tle1, tle2, name, ts)
        satellites.append(sat)

    # Positions with noise
    positions = []
    for sat in satellites:
        geo = sat.at(times)
        pos = geo.position.km.T
        noise = np.random.normal(0.0, NOISE_LEVEL_KM, pos.shape)
        positions.append(pos + noise)

    # Force collisions
    for (i, j, t_idx) in FORCED_COLLISIONS:
        if i >= len(positions) or j >= len(positions) or t_idx >= len(times):
            continue
        offset = np.random.uniform(-0.4, 0.4, 3)
        positions[j][t_idx] = positions[i][t_idx] + offset

    # Dataset
    data = []
    actual_satellites = len(positions)
    time_steps = len(times)

    for t in range(time_steps):
        for i in range(actual_satellites):
            for j in range(i+1, actual_satellites):
                p1 = positions[i][t]
                p2 = positions[j][t]
                distance = np.linalg.norm(p1 - p2)
                label = 1 if distance <= COLLISION_THRESHOLD_KM else 0
                features = np.concatenate((p1, p2, [distance]))
                data.append(np.append(features, label))

    df = pd.DataFrame(data, columns=['x1','y1','z1','x2','y2','z2','distance','label'])
    df.to_csv('generated_collision_dataset.csv', index=False)

    # Train
    X = df[['x1','y1','z1','x2','y2','z2','distance']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model trained and saved as '{MODEL_FILE}'")

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_FILE)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Satellite Collision Predictor", layout="centered")
st.title("🛰 Satellite Collision Predictor")

st.subheader("Enter Satellite Positions (km)")
col1, col2 = st.columns(2)

with col1:
    x1 = st.number_input("Satellite 1 - X:", value=0.0, format="%.3f")
    y1 = st.number_input("Satellite 1 - Y:", value=0.0, format="%.3f")
    z1 = st.number_input("Satellite 1 - Z:", value=0.0, format="%.3f")

with col2:
    x2 = st.number_input("Satellite 2 - X:", value=0.0, format="%.3f")
    y2 = st.number_input("Satellite 2 - Y:", value=0.0, format="%.3f")
    z2 = st.number_input("Satellite 2 - Z:", value=0.0, format="%.3f")

if st.button("🔍 Predict Collision"):
        # Positions & distance
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        distance = np.linalg.norm(p1 - p2)

        # Prediction
        features = np.concatenate((p1,p2,[distance])).reshape(1,-1)
        prediction = model.predict(features)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            if probs.shape[0] == 2:
                proba = probs[1] * 100
            else:
                proba = probs[0] * 100  # Only one class in training

        # Time estimate
        avg_speed_km_s = 7.8  # km/s LEO
        time_seconds = distance / avg_speed_km_s
        time_minutes = time_seconds / 60

        # Message
        if prediction==1:
            alert_msg = f"⚠ Collision Risk Detected!\nDistance: {distance:.3f} km\nProbability: {proba:.2f}%\nTime to collision: {time_minutes:.1f} min"
            conclusion = "Recommendation: Take preventive action. Adjust orbit or monitor closely."
        else:
            alert_msg = f"✅ No Collision Risk.\nDistance: {distance:.3f} km\nProbability: {proba:.2f}%\nTime: {time_minutes:.1f} min"
            conclusion = "Orbit is safe. No immediate action required."

        st.markdown(f"### Prediction Message:\n{alert_msg}")

        # 3D Visualization
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*p1, color='blue', s=50, label='Satellite 1')
        ax.scatter(*p2, color='red', s=50, label='Satellite 2')
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='green', linestyle='--', label='Distance Vector')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
        ax.set_title('Satellite Positions')
        ax.legend()
        st.pyplot(fig)

        # Risk Matrix
        risk_data = {
            "Alert": ["Collision Risk" if prediction==1 else "Safe"],
            "Distance (km)": [distance],
            "Probability (%)": [proba],
            "Time (min)": [time_minutes]
        }
        risk_df = pd.DataFrame(risk_data)
        st.subheader("Risk Matrix")
        st.dataframe(risk_df)

        # Conclusion
        st.subheader("Conclusion")
        st.markdown(conclusion)

   