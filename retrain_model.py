"""
FEDXAI-AUTO: Robust Retraining Script
- Uses standard binary_crossentropy (no custom weighted loss)
- Proper Train (70%) / Val (15%) / Test (15%) splits
- Early stopping based on validation accuracy
- Aims for >98% accuracy
- Re-exports to TFLite (Float32, Float16, INT8) and C Header
"""
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("FEDXAI-AUTO: ROBUST MODEL RETRAINING")
print("=" * 60)

BASE_DIR = r"c:\Users\kvars\OneDrive\Documents\FEDXAI-AUTO"
DATA_PATH = os.path.join(BASE_DIR, "training_subset.csv")
EDGE_DIR = os.path.join(BASE_DIR, "phase4_edge")
os.makedirs(EDGE_DIR, exist_ok=True)

# 1. Load Data
print("\n[1/6] Loading & Preprocessing Data...")
df = pd.read_csv(DATA_PATH)
feature_cols = ['Engine RPM','Fuel Pressure (Bar)','Coolant Temp (C)',
                'Intake Air Temp (C)','Catalyst Temp (C)',
                'O2 Sensor Voltage (V)','Fuel Trim Short-Term (%)','Fuel Trim Long-Term (%)']
SEQ_LEN = 20

X_raw = df[feature_cols].values
y_raw = np.round(df['Ground Truth'].values).astype(int)

print(f"  Dataset: {len(df)} rows, Classes: Healthy={np.sum(y_raw==0)}, Failure={np.sum(y_raw==1)}")

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Create Sequences
print("\n[2/6] Building Sequences...")
X_seq, y_seq = [], []
# Group by Silo_ID if possible to avoid crossing boundaries
if 'Silo_ID' in df.columns:
    for _, grp in df.groupby('Silo_ID'):
        idx = grp.index
        grp_X = X_scaled[idx]
        grp_y = y_raw[idx]
        if len(grp_X) <= SEQ_LEN: continue
        for i in range(len(grp_X) - SEQ_LEN):
            X_seq.append(grp_X[i:i+SEQ_LEN])
            y_seq.append(grp_y[i+SEQ_LEN])
else:
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        y_seq.append(y_raw[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"  Total Sequences: {len(X_seq)}")

# 3. Train / Val / Test Split
print("\n[3/6] Splitting Data (70/15/15)...")
# Shuffle sequences
indices = np.arange(len(X_seq))
np.random.shuffle(indices)
X_seq = X_seq[indices]
y_seq = y_seq[indices]

n = len(X_seq)
tr_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X_seq[:tr_end], y_seq[:tr_end]
X_val, y_val = X_seq[tr_end:val_end], y_seq[tr_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# 4. Build Model
print("\n[4/6] Building & Training Model...")
inp = tf.keras.Input(shape=(SEQ_LEN, len(feature_cols)))
x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', # STANDARD LOSS
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
]

# Train
t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
train_time = time.time() - t0

# Evaluate
print("\n[5/6] Evaluating on Held-Out Test Set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {acc*100:.2f}%")
print(f"  Test Loss:     {loss:.4f}")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob.flatten() > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"], digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"          Pred_H   Pred_F")
print(f"  True_H  {cm[0][0]:6d}   {cm[0][1]:6d}")
print(f"  True_F  {cm[1][0]:6d}   {cm[1][1]:6d}")

if acc < 0.98:
    print("\n[!] WARNING: Accuracy is below 98%. You may need more epochs or hyperparameter tuning.")
else:
    print("\n[+] SUCCESS: Accuracy is >= 98%!")

# Save Keras Models
# Overwrite both edge and production with this highly accurate model
model.save(os.path.join(BASE_DIR, "fedxai_production_best.keras"))
model.save(os.path.join(BASE_DIR, "fedxai_edge_realistic.keras"))
print("\nSaved Keras models: fedxai_production_best.keras & fedxai_edge_realistic.keras")

# 6. TFLite Export
print("\n[6/6] Exporting to TFLite (Edge Deployment)...")

# Float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_f32 = converter.convert()
p_f32 = os.path.join(EDGE_DIR, "fedxai_f32.tflite")
with open(p_f32, "wb") as f: f.write(tflite_f32)
print(f"  [+] Float32 Model: {os.path.getsize(p_f32)/1024:.1f} KB")

# Define Representative Dataset for Quantization
def rep_data_gen():
    for i in range(min(500, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

# Float16 Quantization
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
converter_f16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter_f16._experimental_lower_tensor_list_ops = False
tflite_f16 = converter_f16.convert()
p_f16 = os.path.join(EDGE_DIR, "fedxai_f16.tflite")
with open(p_f16, "wb") as f: f.write(tflite_f16)
print(f"  [+] Float16 Model: {os.path.getsize(p_f16)/1024:.1f} KB")

# INT8 Quantization
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = rep_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.float32 
converter_int8.inference_output_type = tf.float32
tflite_int8 = converter_int8.convert()
p_int8 = os.path.join(EDGE_DIR, "fedxai_int8.tflite")
with open(p_int8, "wb") as f: f.write(tflite_int8)
print(f"  [+] INT8 Model:    {os.path.getsize(p_int8)/1024:.1f} KB")

# Set INT8 as the primary realistic edge model since it's smallest and fastest for ESP32
p_realistic = os.path.join(EDGE_DIR, "fedxai_realistic.tflite")
with open(p_realistic, "wb") as f: f.write(tflite_int8)
print(f"  [+] Saved fedxai_realistic.tflite (uses INT8)")

# Generate C Array Header for ESP32
print("\nGenerating ESP32 C Header...")
import binascii
def create_c_array(bin_data, var_name):
    hex_array = [f"0x{b:02x}" for b in bin_data]
    hex_str = ", ".join(hex_array)
    # Break into lines
    lines = [hex_str[i:i+80] for i in range(0, len(hex_str), 80)]
    c_str = f"const unsigned char {var_name}[] = {{\n"
    c_str += ",\n".join(lines[:-1])
    if len(lines) > 1: c_str += ",\n"
    c_str += lines[-1] + "\n};\n"
    c_str += f"const unsigned int {var_name}_len = {len(bin_data)};\n"
    return c_str

c_header = create_c_array(tflite_int8, "fedxai_model_tflite")
header_path = os.path.join(EDGE_DIR, "fedxai_model.h")
with open(header_path, "w") as f:
    f.write("#ifndef FEDXAI_MODEL_H\n#define FEDXAI_MODEL_H\n\n")
    f.write(c_header)
    f.write("\n#endif // FEDXAI_MODEL_H\n")
print(f"  [+] Exported C Header: {header_path} ({os.path.getsize(header_path)/1024:.1f} KB)")

print("\n" + "=" * 60)
print(f"RETRAINING COMPLETE in {train_time/60:.1f} minutes!")
print("=" * 60)
