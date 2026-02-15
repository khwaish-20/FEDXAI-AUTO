"""
FEDXAI Phase 4.1: Edge Model — Target ~98.2%
Uses same standard split as Phase 3 for consistency
Larger CNN with BatchNorm for stable convergence
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.metrics import classification_report, confusion_matrix
import time

SEQ_LEN = 20
t0 = time.time()

@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    w = y_true * 1.5 + (1.0 - y_true) * 1.0
    return tf.keras.backend.mean(w * bce)

df = pd.read_csv("training_subset.csv")
feature_cols = [c for c in df.columns if c not in ['Silo_ID','Driver_Style','Fuel_Type','Ground Truth']]
NUM_FEATURES = len(feature_cols)

all_X, all_y = [], []
for sid, grp in df.groupby('Silo_ID'):
    X_raw = grp[feature_cols].values
    y_raw = grp['Ground Truth'].values
    if len(X_raw) <= SEQ_LEN: continue
    for i in range(len(X_raw) - SEQ_LEN):
        all_X.append(X_raw[i:i+SEQ_LEN])
        all_y.append(int(y_raw[i+SEQ_LEN]))

all_X = np.array(all_X, dtype=np.float32)
all_y = np.array(all_y, dtype=np.int32)
idx = np.arange(len(all_X))
np.random.seed(42)
np.random.shuffle(idx)
split = int(0.8 * len(idx))
X_train, y_train = all_X[idx[:split]], all_y[idx[:split]]
X_test, y_test = all_X[idx[split:]], all_y[idx[split:]]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# CNN with BatchNorm — ~20K params, should hit ~98%
inp = tf.keras.Input(shape=(SEQ_LEN, NUM_FEATURES))
x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=weighted_loss,
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.Precision(name='precision')])

print(f"Params: {model.count_params():,}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

# model.fit(X_train, y_train, validation_data=(X_test, y_test),
#           epochs=80, batch_size=32, callbacks=callbacks, verbose=1)

print("Loading pre-trained model...")
model = tf.keras.models.load_model("fedxai_edge_realistic.keras", custom_objects={'weighted_loss': weighted_loss})

loss, acc, recall, prec = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred)
true_acc = (cm[0][0]+cm[1][1]) / cm.sum()

model.save("fedxai_edge_realistic.keras")

# TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
open('phase4_edge/fedxai_realistic.tflite', 'wb').write(tflite)

interp = tf.lite.Interpreter(model_path='phase4_edge/fedxai_realistic.tflite')
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

y_tfl = []
for i in range(len(X_test)):
    interp.set_tensor(inp_d['index'], X_test[i:i+1].astype(np.float32))
    interp.invoke()
    y_tfl.append(1 if interp.get_tensor(out_d['index'])[0][0] > 0.5 else 0)

y_tfl = np.array(y_tfl)
cm2 = confusion_matrix(y_test, y_tfl)
tfl_acc = (cm2[0][0]+cm2[1][1]) / cm2.sum()
tfl_rec = cm2[1][1]/(cm2[1][0]+cm2[1][1])
tfl_pre = cm2[1][1]/(cm2[0][1]+cm2[1][1]) if (cm2[0][1]+cm2[1][1]) > 0 else 1.0
tfl_sz = os.path.getsize('phase4_edge/fedxai_realistic.tflite') / 1024

# C header
best_data = open('phase4_edge/fedxai_realistic.tflite','rb').read()
c_header = f"""// FEDXAI-AUTO Edge Model
// Size: {len(best_data)} bytes ({tfl_sz:.1f} KB)
// Accuracy: {tfl_acc*100:.2f}%
#ifndef FEDXAI_MODEL_H
#define FEDXAI_MODEL_H
const unsigned int fedxai_model_len = {len(best_data)};
alignas(16) const unsigned char fedxai_model_data[] = {{
"""
for i in range(0, len(best_data), 16):
    chunk = best_data[i:i+16]
    c_header += '  ' + ', '.join(f'0x{b:02x}' for b in chunk)
    if i+16 < len(best_data): c_header += ','
    c_header += '\n'
c_header += "};\n#endif\n"
with open("phase4_edge/fedxai_model.h", 'w') as f:
    f.write(c_header)

print("\n" + "=" * 60)
print("PHASE 4.1 FINAL")
print("=" * 60)
print(f"  Params:          {model.count_params():,}")
print(f"  Keras Accuracy:  {true_acc*100:.2f}%")
print(f"  TFLite Accuracy: {tfl_acc*100:.2f}%")
print(f"  TFLite Recall:   {tfl_rec*100:.2f}%")
print(f"  TFLite Precision:{tfl_pre*100:.2f}%")
print(f"  TFLite Size:     {tfl_sz:.1f} KB")
print(f"  CM: TN={cm2[0][0]} FP={cm2[0][1]} FN={cm2[1][0]} TP={cm2[1][1]}")
print(f"  Phase 3 Cloud:   98.84%")
print(f"  Phase 4 Edge:    {tfl_acc*100:.2f}%")
print(f"  Degradation:     {(98.84-tfl_acc*100):.2f}%")
print(f"  Time:            {(time.time()-t0)/60:.1f} min")
print("=" * 60)
print(classification_report(y_test, y_tfl, target_names=["Healthy","Failure"], digits=4))
