"""
FEDXAI Phase 3.2: XAI Integration (SHAP Analysis)
Uses the same data pipeline as phase4_realistic.py for consistency.
Generates:
  1. shap_summary_plot.png  – Global Feature Importance (all 8 sensors)
  2. shap_force_plot_failure.html – Local explanation for a single failure case
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ── Configuration (MUST match phase4_realistic.py) ──────────────────────────
SEQ_LEN = 20
DATA_PATH = "training_subset.csv"
MODEL_PATH = "fedxai_edge_realistic.keras"

# ── 1. Custom Loss (needed to load the saved .keras model) ──────────────────
@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    w = y_true * 1.5 + (1.0 - y_true) * 1.0
    return tf.keras.backend.mean(w * bce)

# ── 2. Data Loading (identical logic to phase4_realistic.py) ────────────────
print("--- Phase 3.2: XAI Integration (SHAP) ---")
print(f"Loading data from {DATA_PATH}...")

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ['Silo_ID', 'Driver_Style', 'Fuel_Type', 'Ground Truth']]
NUM_FEATURES = len(feature_cols)
print(f"Features ({NUM_FEATURES}): {feature_cols}")

all_X, all_y = [], []
for sid, grp in df.groupby('Silo_ID'):
    X_raw = grp[feature_cols].values
    y_raw = grp['Ground Truth'].values
    if len(X_raw) <= SEQ_LEN:
        continue
    y_raw = np.round(y_raw).astype(int)
    for i in range(len(X_raw) - SEQ_LEN):
        all_X.append(X_raw[i:i + SEQ_LEN])
        all_y.append(y_raw[i + SEQ_LEN])

all_X = np.array(all_X, dtype=np.float32)
all_y = np.array(all_y, dtype=np.int32)

# Same 80/20 split with same seed as training
idx = np.arange(len(all_X))
np.random.seed(42)
np.random.shuffle(idx)
split = int(0.8 * len(idx))
X_test = all_X[idx[split:]]
y_test = all_y[idx[split:]]
print(f"Test set: {len(X_test)} samples  |  Features: {NUM_FEATURES}")

# ── 3. Load Model ──────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'weighted_loss': weighted_loss})
print("Model loaded successfully.")

# ── 4. Prepare SHAP Background & Explanation Samples ───────────────────────
# Background: 200 random test samples for baseline comparison
np.random.seed(123)
bg_idx = np.random.choice(len(X_test), min(200, len(X_test)), replace=False)
background = X_test[bg_idx]

# Explanation samples: 20 failures + 20 healthy
fail_idx = np.where(y_test == 1)[0]
healthy_idx = np.where(y_test == 0)[0]
print(f"Failures in test: {len(fail_idx)}  |  Healthy: {len(healthy_idx)}")

n_fail = min(20, len(fail_idx))
n_healthy = min(20, len(healthy_idx))
explain_idx = np.concatenate([fail_idx[:n_fail], healthy_idx[:n_healthy]])
X_explain = X_test[explain_idx]
print(f"Explaining {len(X_explain)} samples ({n_fail} failures + {n_healthy} healthy)")

# ── 5. Compute SHAP Values ─────────────────────────────────────────────────
print("Initializing SHAP GradientExplainer...")
explainer = shap.GradientExplainer(model, background)

print("Computing SHAP values (this may take a minute)...")
shap_values = explainer.shap_values(X_explain)

# GradientExplainer returns list[ndarray] for binary; unwrap
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print(f"SHAP values shape: {shap_values.shape}")  # (samples, SEQ_LEN, features)

# ── 6. Aggregate over time dimension ──────────────────────────────────────
# Sum |SHAP| across all 20 time-steps → (samples, features)
shap_agg = np.sum(np.abs(shap_values), axis=1)

# Mean feature values over time for colour coding
X_explain_mean = X_explain.mean(axis=1)

# ── 7. Summary Plot ───────────────────────────────────────────────────────
print("\nGenerating Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_agg,
    features=X_explain_mean,
    feature_names=feature_cols,
    show=False,
    max_display=NUM_FEATURES,   # Show ALL features
    sort=True
)
plt.title("Global Feature Importance (Aggregated over Time)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved shap_summary_plot.png")

# ── 8. Force Plot for a Failure Case ─────────────────────────────────────
print("\nGenerating Force Plot for Failure case...")
idx_failure = 0  # First sample is a failure (from fail_idx[:n_fail])

sample_shap = shap_agg[idx_failure]         # (features,)
sample_features = X_explain_mean[idx_failure]  # (features,)

# Base value = average model prediction on background
try:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0]) if len(base_value) > 0 else float(base_value)
except AttributeError:
    base_probs = model.predict(background, verbose=0)
    base_value = float(base_probs.mean())

print(f"Base value (avg prediction): {base_value:.4f}")

force_html = shap.force_plot(
    base_value,
    sample_shap,
    sample_features,
    feature_names=feature_cols,
    matplotlib=False,
    show=False
)
shap.save_html("shap_force_plot_failure.html", force_html)
print("✅ Saved shap_force_plot_failure.html")

# ── 9. Print Feature Ranking ─────────────────────────────────────────────
print("\n=== Feature Importance Ranking ===")
mean_abs_shap = shap_agg.mean(axis=0)
ranking = np.argsort(mean_abs_shap)[::-1]
for i, r in enumerate(ranking):
    print(f"   {i+1}. {feature_cols[r]:25s} → mean|SHAP| = {mean_abs_shap[r]:.6f}")

print("\n🎯 Analysis Complete!")
