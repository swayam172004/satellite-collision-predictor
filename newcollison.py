import numpy as np
from skyfield.api import EarthSatellite, load, wgs84, utc
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle 
# Parameters
NUM_SATELLITES = 5
FORCED_COLLISIONS = [(0, 1, 20), (2, 3, 40)]  # (sat1_idx, sat2_idx, time_idx)
DURATION_MINUTES = 60
COLLISION_THRESHOLD_KM = 9000

# Load TLE data
with open('gp.txt') as file:
    lines = file.readlines()

ts = load.timescale()
start = datetime.now().replace(tzinfo=utc)

# Generate times
times = ts.utc(start.year, start.month, start.day, start.hour,
               start.minute + np.arange(0, DURATION_MINUTES, 1))

# Load N satellites
satellites = []
names = []

for i in range(0, NUM_SATELLITES * 3, 3):
    if i + 2 >= len(lines):
        break  
    name = lines[i].strip()
    tle1 = lines[i+1].strip()
    tle2 = lines[i+2].strip()
    sat = EarthSatellite(tle1, tle2, name, ts)
    satellites.append(sat)
    names.append(name)

# Get positions
# Get positions with noise added
positions = []

# Noise level in km (adjust as needed)
NOISE_LEVEL_KM = 0.2  # small noise up to ±0.2 km

for sat in satellites:
    geo = sat.at(times)
    pos = geo.position.km.T  # (T, 3)

    # Add Gaussian noise to simulate drift/errors
    noise = np.random.normal(loc=0.0, scale=NOISE_LEVEL_KM, size=pos.shape)
    noisy_pos = pos + noise
    positions.append(noisy_pos)

# Force collisions with tiny noise (to simulate very close approach)
for (i, j, t_idx) in FORCED_COLLISIONS:
    if i >= len(positions) or j >= len(positions):
        print(f"⚠️ Invalid satellite index: i={i}, j={j}, skipping...")
        continue
    if t_idx >= len(times):
        print(f"⚠️ Invalid time index: t_idx={t_idx}, skipping...")
        continue

    offset = np.random.uniform(-0.4, 0.4, size=3)
    positions[j][t_idx] = positions[i][t_idx] + offset

# Generate dataset
data = []
actual_satellites = len(positions)
time_steps = len(times)

print(f"Satellites available: {actual_satellites}, Time steps: {time_steps}")
for t in range(time_steps):
    for i in range(actual_satellites*10000):
        for j in range(i+1, actual_satellites):
            p1 = positions[i][t]
            p2 = positions[j][t]
            distance = np.linalg.norm(p1 - p2)
            label = 1 if distance <= COLLISION_THRESHOLD_KM else 0
            features = np.concatenate((p1, p2, [distance]))
            data.append(np.append(features, label))

df = pd.DataFrame(data, columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'distance', 'label'])
df.to_csv('generated_collision_dataset.csv', index=False)

print("✅ Generated dataset with forced collisions saved as 'generated_collision_dataset.csv'")
