import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# Configuration
SEQUENCE_LENGTH = 30
PROCESSED_DATA_PATH = "combined_processed_data.csv"
MODEL_PATH = "fedxai_advanced_global.keras"

# --- 1. Custom Loss Function (Must be defined to load model) ---
def weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_loss(y_true, y_pred):
        b_ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce
        return tf.keras.backend.mean(weighted_b_ce)
    return weighted_loss

# Register or just pass in custom_objects
loss_fn = weighted_binary_crossentropy(1.0, 5.0)

# --- 2. Data Loading Logic (Reused) ---
def split_data_into_clients(df):
    clients = {}
    silo_ids = df['Silo_ID'].unique()
    
    for sid in silo_ids:
        client_df = df[df['Silo_ID'] == sid]
        
        # Features & Target
        # Ensure we drop the same columns as training: Silo_ID, Driver_Style, Fuel_Type, Ground Truth
        features = client_df.drop(columns=['Silo_ID', 'Driver_Style', 'Fuel_Type', 'Ground Truth'])
        target = client_df['Ground Truth'].values
        
        X = features.values
        y = target
        
        # Features List for Plotting
        feature_names = features.columns.tolist()
        
        # Create Sequences (Windowing)
        X_seq, y_seq = [], []
        if len(X) > SEQUENCE_LENGTH:
            for i in range(len(X) - SEQUENCE_LENGTH):
                X_seq.append(X[i:i+SEQUENCE_LENGTH])
                y_seq.append(y[i+SEQUENCE_LENGTH])
            clients[sid] = (np.array(X_seq), np.array(y_seq), feature_names)
            
    return clients

def main():
    print("--- Phase 3.2: XAI Integration (SHAP) ---")
    
    # 1. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'weighted_loss': loss_fn}
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Data
    print("Loading Data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Get a sample of 50 vehicles for analysis
    all_silos = df['Silo_ID'].unique()
    sampled_silos = np.random.choice(all_silos, size=50, replace=False) # 50 cars
    df_sample = df[df['Silo_ID'].isin(sampled_silos)]
    
    client_data = split_data_into_clients(df_sample)
    
    # Flatten all data into one big array for analysis
    all_X = []
    all_y = []
    feature_names = []
    
    for cid, (X, y, feats) in client_data.items():
        all_X.append(X)
        all_y.append(y)
        feature_names = feats # Save once
        
    X_analysis = np.concatenate(all_X)
    y_analysis = np.concatenate(all_y)
    
    print(f"Analysis Dataset Shape: {X_analysis.shape}") # (N, 30, Features)
    
    # 3. Prepare Background Data for SHAP
    # SHAP needs a "background" distribution to compare against.
    # We take 200 random samples.
    background_idx = np.random.choice(X_analysis.shape[0], 200, replace=False)
    background_data = X_analysis[background_idx]
    
    # 4. Initialize SHAP Explainer
    # Since we have a Deep Learning model (LSTM/CNN), we use GradientExplainer or DeepExplainer.
    # GradientExplainer is robust for TF2.
    print("Initializing SHAP GradientExplainer...")
    
    # Note: GradientExplainer expects inputs as a list of tensors if model has multiple inputs.
    # Here we have single input.
    explainer = shap.GradientExplainer(model, background_data)
    
    # 5. Explain a set of Test Samples (e.g., 20 Failures and 20 Healthy)
    print("Calculating SHAP values (this may take a minute)...")
    
    # Find indices of failures (y=1) and healthy (y=0)
    fail_idx = np.where(y_analysis == 1)[0]
    healthy_idx = np.where(y_analysis == 0)[0]
    
    test_idx = np.concatenate([
        fail_idx[:20], # 20 Failures
        healthy_idx[:20] # 20 Healthy
    ])
    
    X_test_explain = X_analysis[test_idx]
    
    shap_values = explainer.shap_values(X_test_explain)
    
    # shap_values for classification is a list of arrays (one for each class).
    # Since we have 1 output node (sigmoid), it returns a list with 1 array or just the array.
    # Usually list of size 1 for binary output.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    print(f"SHAP Values Shape: {shap_values.shape}") # Should be (40, 30, Features)
    
    # 6. Aggregate Time Dimension for Global Importance
    # We have importance for each time step. To see "Sensor Importance", we sum absolute values over time.
    # Shape: (Samples, Features)
    shap_sum_over_time = np.sum(np.abs(shap_values), axis=1)
    
    # 7. Generate Summary Plot
    print("Generating Summary Plot...")
    plt.figure()
    shap.summary_plot(
        shap_sum_over_time, 
        features=X_test_explain.mean(axis=1), # Use mean feature value over time for color
        feature_names=feature_names,
        show=False
    )
    plt.title("Global Feature Importance (Aggregated over Time)")
    # Save
    plt.savefig("shap_summary_plot.png", bbox_inches='tight')
    print("Saved shap_summary_plot.png")
    
    # 8. Local Explanation (Force Plot) for a specific FAILURE case
    print("\n--- Generating Local Explanation for Failure Case ---")
    # Take the first failure case from our test set
    idx_failure = 0 # First sample in X_test_explain is a failure (from fail_idx[:20])
    
    # We need 2D for force_plot (Sensor values vs SHAP values)
    # Feature values for this sample (Mean over time window)
    sample_feature_values = X_test_explain[idx_failure].mean(axis=0)
    
    # Compute base value manually if attribute missing (common in GradientExplainer)
    try:
        base_value = explainer.expected_value
        if isinstance(base_value, list): base_value = base_value[0]
    except AttributeError:
        # Estimate from background samples
        base_probs = model.predict(background_data)
        base_value = base_probs.mean()
        
    print(f"Base Value for Force Plot: {base_value}")
    
    # Restore missing variable
    shap_sample = shap_sum_over_time[idx_failure] # (Features,)
    
    plt.figure()
    # Force plot is usually HTML/JS, but we can try matplotlib output or save text
    # shap.force_plot() returns HTML. We can save it.
    force_plot_html = shap.force_plot(
        base_value,
        shap_sample.reshape(1, -1),
        sample_feature_values.reshape(1, -1),
        feature_names=feature_names,
        matplotlib=False,
        show=False
    )
    shap.save_html("shap_force_plot_failure.html", force_plot_html)
    print("Saved shap_force_plot_failure.html")
    
    print("\nanalysis Complete.")

if __name__ == "__main__":
    main()
