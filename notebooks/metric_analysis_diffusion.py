import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Dynamically Locate Diffusion Metrics CSV Path
# ─────────────────────────────────────────────────────────────────────

# Get absolute path of current script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level to project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Build full path to metrics_diffusion.csv inside logs folder
csv_path = os.path.join(PROJECT_ROOT, 'logs', 'metrics_diffusion.csv')

# Validate existence before proceeding
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit()

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Load and Prepare the CSV DataFrame
# ─────────────────────────────────────────────────────────────────────

# Load CSV into a DataFrame
df = pd.read_csv(csv_path)

# Sort by timestamp for accurate time-based visualization
df = df.sort_values(by='timestamp')

# ─────────────────────────────────────────────────────────────────────
# STEP 3: FID Score Over Time
# ─────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(df['timestamp'], df['fid_score'], 'b-o')  # Blue markers for FID
plt.xticks(rotation=45)
plt.ylabel('FID Score')
plt.title('Diffusion Model: FID over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Inception Score with Std Dev
# ─────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.errorbar(
    df['timestamp'],
    df['inception_score'],
    yerr=df['inception_std'],
    fmt='m-s'  # Magenta squares with error bars
)
plt.xticks(rotation=45)
plt.ylabel('Inception Score')
plt.title('Diffusion Model: Inception Score over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Combined Plot — FID and Inception Score
# ─────────────────────────────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# FID on left Y-axis
ax1.plot(df['timestamp'], df['fid_score'], 'b--', label='FID Score')
ax1.set_ylabel('FID Score', color='b')

# Inception Score on right Y-axis
ax2.plot(df['timestamp'], df['inception_score'], 'm-', label='Inception Score')
ax2.set_ylabel('Inception Score', color='m')

ax1.set_xlabel('Timestamp')
plt.xticks(rotation=45)
plt.title('Diffusion: FID & Inception Score Trends')
ax1.grid(True)
plt.tight_layout()
plt.show()
