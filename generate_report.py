"""Quick report from saved model — no retraining needed"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.metrics import classification_report, confusion_matrix

SEQ_LEN = 20

@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    w = y_true * 1.5 + (1.0 - y_true) * 1.0
    return tf.keras.backend.mean(w * bce)

# Load model
print("Loading model...")
model = tf.keras.models.load_model("fedxai_production_best.keras")

# Load data
df = pd.read_csv("training_subset.csv")
feature_cols = [c for c in df.columns if c not in ['Silo_ID','Driver_Style','Fuel_Type','Ground Truth']]

all_X, all_y = [], []
for sid, grp in df.groupby('Silo_ID'):
    X_raw = grp[feature_cols].values
    y_raw = grp['Ground Truth'].values
    if len(X_raw) <= SEQ_LEN:
        continue
    for i in range(len(X_raw) - SEQ_LEN):
        all_X.append(X_raw[i:i+SEQ_LEN])
        all_y.append(int(y_raw[i+SEQ_LEN]))

all_X = np.array(all_X)
all_y = np.array(all_y, dtype=int)

idx = np.arange(len(all_X))
np.random.seed(42)
np.random.shuffle(idx)
split = int(0.8 * len(idx))
X_test = all_X[idx[split:]]
y_test = all_y[idx[split:]]

# Evaluate
loss, acc, recall, prec = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("FEDXAI-AUTO: FINAL MODEL RESULTS")
print("=" * 60)
print(f"Test Samples: {len(y_test)} ({int(y_test.sum())} failures, {len(y_test)-int(y_test.sum())} healthy)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy","Failure"], digits=4))
print(f"Confusion Matrix:")
print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
final_acc = (cm[0][0]+cm[1][1]) / cm.sum()
miss = cm[1][0]/(cm[1][0]+cm[1][1]) if (cm[1][0]+cm[1][1])>0 else 0
print(f"\n  FINAL ACCURACY:  {final_acc*100:.2f}%")
print(f"  RECALL:          {recall*100:.2f}%")
print(f"  PRECISION:       {prec*100:.2f}%")
print(f"  MISS RATE (FN):  {miss*100:.2f}%")
print("=" * 60)
