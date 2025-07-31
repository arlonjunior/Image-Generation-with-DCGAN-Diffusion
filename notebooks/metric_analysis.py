# metric_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────
# 📁 STEP 1: Dynamically Locate CSV Path from Project Root
# ─────────────────────────────────────────────────────────────────────

# Get absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Build full path to the metrics CSV inside the logs folder
csv_path = os.path.join(PROJECT_ROOT, 'logs', 'metrics.csv')

# 📌 Validate that the file exists before proceeding
if not os.path.exists(csv_path):
    print(f"⚠️ File not found: {csv_path}")
    exit()

# ─────────────────────────────────────────────────────────────────────
# 📊 STEP 2: Load and Prepare the CSV DataFrame
# ─────────────────────────────────────────────────────────────────────

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Sort entries by timestamp for chronological plotting
df = df.sort_values(by='timestamp')

# ─────────────────────────────────────────────────────────────────────
# 📉 STEP 3: Plot FID Score Over Time
# ─────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(df['timestamp'], df['fid_score'], 'r-o')  # Red circles for FID
plt.xticks(rotation=45)
plt.ylabel('FID Score')
plt.title('Model Evaluation: FID over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# 📈 STEP 4: Plot Inception Score with Error Bars
# ─────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.errorbar(
    df['timestamp'],
    df['inception_score'],
    yerr=df['inception_std'],
    fmt='g-s'  # Green squares with error bars
)
plt.xticks(rotation=45)
plt.ylabel('Inception Score')
plt.title('Model Evaluation: Inception Score over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# 📊 STEP 5: Combined Plot for FID & Inception Score
# ─────────────────────────────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Second y-axis for overlay

# Plot FID Score on first axis
ax1.plot(df['timestamp'], df['fid_score'], 'r--', label='FID')
ax1.set_ylabel('FID Score', color='r')

# Plot Inception Score on second axis
ax2.plot(df['timestamp'], df['inception_score'], 'g-', label='Inception')
ax2.set_ylabel('Inception Score', color='g')

# Final styling touches
ax1.set_xlabel('Timestamp')
plt.xticks(rotation=45)
plt.title('FID & Inception Score Trends')
ax1.grid(True)
plt.tight_layout()
plt.show()
