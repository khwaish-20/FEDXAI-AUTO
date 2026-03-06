"""
FEDXAI Phase 3.2: XAI Integration -- SHAP + LIME
Uses the same data pipeline as phase4_realistic.py for consistency.

Generates:
  SHAP Outputs:
    1. shap_summary_plot.png         - Global feature importance (beeswarm)
    2. shap_force_plot_failure.html   - Local explanation for a single failure

  LIME Outputs:
    3. lime_explanation_failure.png   - Local LIME explanation for a failure case
    4. lime_explanation_healthy.png   - Local LIME explanation for a healthy case
    5. lime_feature_importance.png    - Aggregated LIME feature importance (bar chart)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# CONFIGURATION  (MUST match phase4_realistic.py)
# ===========================================================================
SEQ_LEN = 20
DATA_PATH = "training_subset.csv"
MODEL_PATH = "fedxai_edge_realistic.keras"

# ===========================================================================
# 1. CUSTOM LOSS  (needed to load the saved .keras model)
# ===========================================================================
@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    w = y_true * 1.5 + (1.0 - y_true) * 1.0
    return tf.keras.backend.mean(w * bce)

# ===========================================================================
# 2. DATA LOADING  (identical to phase4_realistic.py)
# ===========================================================================
print("=" * 60)
print("  FEDXAI Phase 3.2: XAI Integration (SHAP + LIME)")
print("=" * 60)
print(f"\nLoading data from {DATA_PATH}...")

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
X_train = all_X[idx[:split]]
y_train = all_y[idx[:split]]
X_test  = all_X[idx[split:]]
y_test  = all_y[idx[split:]]
print(f"Train: {len(X_train)} | Test: {len(X_test)} samples  |  Features: {NUM_FEATURES}")

# ===========================================================================
# 3. LOAD MODEL
# ===========================================================================
print(f"\nLoading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'weighted_loss': weighted_loss})
print("Model loaded successfully.\n")

# ===========================================================================
# 4. SELECT EXPLANATION SAMPLES
# ===========================================================================
fail_idx    = np.where(y_test == 1)[0]
healthy_idx = np.where(y_test == 0)[0]
print(f"Failures in test: {len(fail_idx)}  |  Healthy: {len(healthy_idx)}")

n_fail    = min(20, len(fail_idx))
n_healthy = min(20, len(healthy_idx))
explain_idx = np.concatenate([fail_idx[:n_fail], healthy_idx[:n_healthy]])
X_explain   = X_test[explain_idx]
y_explain   = y_test[explain_idx]
print(f"Explaining {len(X_explain)} samples ({n_fail} failures + {n_healthy} healthy)\n")

# ===========================================================================
#                         PART A:  S H A P
# ===========================================================================
print("-" * 60)
print("  PART A: SHAP (GradientExplainer)")
print("-" * 60)

# Background: 200 random test samples
np.random.seed(123)
bg_idx     = np.random.choice(len(X_test), min(200, len(X_test)), replace=False)
background = X_test[bg_idx]

print("Initializing SHAP GradientExplainer...")
explainer = shap.GradientExplainer(model, background)

print("Computing SHAP values (this may take a minute)...")
shap_values = explainer.shap_values(X_explain)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
# Squeeze trailing dimension if present: (40, 20, 8, 1) -> (40, 20, 8)
if shap_values.ndim == 4:
    shap_values = shap_values.squeeze(axis=-1)
print(f"SHAP values shape: {shap_values.shape}")

# Aggregate over time dimension: sum |SHAP| across 20 time-steps -> (samples, features)
shap_agg       = np.sum(np.abs(shap_values), axis=1)
X_explain_mean = X_explain.mean(axis=1)

# -- A1. SHAP Summary Plot --
print("\nGenerating SHAP Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_agg,
    features=X_explain_mean,
    feature_names=feature_cols,
    show=False,
    max_display=NUM_FEATURES,
    sort=True
)
plt.title("SHAP: Global Feature Importance (Aggregated over Time)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved shap_summary_plot.png")

# -- A2. SHAP Force Plot (Failure) --
print("Generating SHAP Force Plot for Failure case...")
sample_shap     = shap_agg[0]
sample_features = X_explain_mean[0]

try:
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0]) if hasattr(base_value, '__len__') and len(base_value) > 0 else float(base_value)
    else:
        base_value = float(base_value)
except AttributeError:
    base_probs = model.predict(background, verbose=0)
    base_value = float(base_probs.mean())

print(f"Base value (avg prediction): {base_value:.4f}")

# Ensure 1D arrays for force_plot
sample_shap_1d = np.array(sample_shap).flatten()
sample_feat_1d = np.array(sample_features).flatten()

force_html = shap.force_plot(
    base_value,
    sample_shap_1d,
    sample_feat_1d,
    feature_names=feature_cols,
    matplotlib=False,
    show=False
)
shap.save_html("shap_force_plot_failure.html", force_html)
print("[OK] Saved shap_force_plot_failure.html")

# -- A3. SHAP Feature Ranking --
print("\n=== SHAP Feature Importance Ranking ===")
mean_abs_shap = shap_agg.mean(axis=0)
shap_ranking  = np.argsort(mean_abs_shap)[::-1]
for i, r in enumerate(shap_ranking):
    print(f"   {i+1}. {feature_cols[r]:30s} -> mean|SHAP| = {mean_abs_shap[r]:.6f}")


# ===========================================================================
#                         PART B:  L I M E
# ===========================================================================
print("\n" + "-" * 60)
print("  PART B: LIME (Local Interpretable Model-Agnostic Explanations)")
print("-" * 60)

# -- B0. Flatten time-series for LIME --
# LIME works on tabular data. We average each sensor across the 20 time-steps
# to produce a single feature vector per sample.
X_train_flat   = X_train.mean(axis=1)     # (N_train, 8)
X_test_flat    = X_test.mean(axis=1)      # (N_test, 8)
X_explain_flat = X_explain.mean(axis=1)   # (40, 8)

# -- B1. LIME-compatible prediction wrapper --
# LIME passes 2D tabular data (N, 8). We need to expand it back to 3D (N, 20, 8)
# by repeating the flat values across all time-steps for model prediction.
def lime_predict_proba(X_flat_2d):
    """Wrapper: Takes (N, features) -> returns (N, 2) class probabilities."""
    X_3d = np.tile(X_flat_2d[:, np.newaxis, :], (1, SEQ_LEN, 1)).astype(np.float32)
    preds = model.predict(X_3d, verbose=0).flatten()
    return np.column_stack([1 - preds, preds])  # [P(healthy), P(failure)]

# -- B2. Initialize LIME Explainer --
print("Initializing LIME TabularExplainer...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_flat,
    feature_names=feature_cols,
    class_names=["Healthy", "Failure"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# -- B3. LIME Explanation for a Failure Case --
print("Generating LIME explanation for FAILURE case...")
fail_sample_flat = X_explain_flat[0]  # First sample is a failure
lime_exp_fail = lime_explainer.explain_instance(
    fail_sample_flat,
    lime_predict_proba,
    num_features=NUM_FEATURES,
    top_labels=1
)

lime_label = lime_exp_fail.available_labels()[0]

fig = lime_exp_fail.as_pyplot_figure(label=lime_label)
fig.set_size_inches(10, 5)
plt.title("LIME: Local Explanation for FAILURE Case", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("lime_explanation_failure.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved lime_explanation_failure.png")

# -- B4. LIME Explanation for a Healthy Case --
print("Generating LIME explanation for HEALTHY case...")
healthy_sample_flat = X_explain_flat[n_fail]  # First healthy sample
lime_exp_healthy = lime_explainer.explain_instance(
    healthy_sample_flat,
    lime_predict_proba,
    num_features=NUM_FEATURES,
    top_labels=1
)

lime_label_h = lime_exp_healthy.available_labels()[0]

fig = lime_exp_healthy.as_pyplot_figure(label=lime_label_h)
fig.set_size_inches(10, 5)
plt.title("LIME: Local Explanation for HEALTHY Case", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("lime_explanation_healthy.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved lime_explanation_healthy.png")

# -- B5. Aggregated LIME Feature Importance --
print("Computing aggregated LIME importance across all explanation samples...")
lime_importances = np.zeros(NUM_FEATURES)

for i in range(len(X_explain_flat)):
    exp = lime_explainer.explain_instance(
        X_explain_flat[i],
        lime_predict_proba,
        num_features=NUM_FEATURES,
        top_labels=1
    )
    label = exp.available_labels()[0]
    feature_weights = dict(exp.as_list(label=label))
    for j, feat_name in enumerate(feature_cols):
        for condition, weight in feature_weights.items():
            if feat_name in condition:
                lime_importances[j] += abs(weight)
                break

# Average across samples
lime_importances /= len(X_explain_flat)

# Plot aggregated LIME importance
lime_ranking = np.argsort(lime_importances)[::-1]
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, NUM_FEATURES))
bars = plt.barh(
    [feature_cols[r] for r in lime_ranking][::-1],
    [lime_importances[r] for r in lime_ranking][::-1],
    color=colors
)
plt.xlabel("Mean |LIME Weight|", fontsize=11)
plt.title("LIME: Global Feature Importance (Aggregated)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("lime_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved lime_feature_importance.png")

# -- B6. LIME Feature Ranking --
print("\n=== LIME Feature Importance Ranking ===")
for i, r in enumerate(lime_ranking):
    print(f"   {i+1}. {feature_cols[r]:30s} -> mean|LIME| = {lime_importances[r]:.6f}")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("  All outputs generated successfully!")
print("=" * 60)
print("  SHAP:")
print("    - shap_summary_plot.png")
print("    - shap_force_plot_failure.html")
print("  LIME:")
print("    - lime_explanation_failure.png")
print("    - lime_explanation_healthy.png")
print("    - lime_feature_importance.png")
print("=" * 60)
print("\n[DONE] XAI Analysis Complete (SHAP + LIME)!")
