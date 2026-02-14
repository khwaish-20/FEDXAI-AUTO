import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler

RAW_DATA_DIR = "simulated_data"
OUTPUT_RAW_FILE = "combined_raw_data.csv"
OUTPUT_PROCESSED_FILE = "combined_processed_data.csv"

def process_data():
    # 1. Collect all CSV files
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "car_*.csv"))
    
    if not all_files:
        print("No data files found!")
        return

    print(f"Found {len(all_files)} files. Combining...")
    
    # 2. Combine into one DataFrame (Raw)
    # We need to handle the fact that some columns might be different (PSI vs Bar)
    # Pandas concat will align columns, creating NaNs where columns don't exist in a specific file
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Save combined raw data
    combined_df.to_csv(OUTPUT_RAW_FILE, index=False)
    print(f"Saved raw combined data to {OUTPUT_RAW_FILE}")
    print(f"Raw shape: {combined_df.shape}")
    
    # 3. Preprocessing & Cleaning
    print("\n--- Starting Preprocessing ---")
    
    # A. Handling Heterogeneity (Units)
    # Merge 'Fuel Pressure (PSI)' into 'Fuel Pressure (Bar)'
    if 'Fuel Pressure (PSI)' in combined_df.columns:
        print("Detected PSI units. Converting to Bar...")
        # Where Bar is NaN and PSI is not, convert PSI to Bar
        # 1 PSI = 0.0689476 Bar
        
        # Create a temporary series for converted values
        psi_converted = combined_df['Fuel Pressure (PSI)'] * 0.0689476
        
        # Fill missing Bar values with converted PSI values
        combined_df['Fuel Pressure (Bar)'] = combined_df['Fuel Pressure (Bar)'].fillna(psi_converted)
        
        # Drop the PSI column
        combined_df = combined_df.drop(columns=['Fuel Pressure (PSI)'])
    
    # B. Handling Missing Values (Imputation)
    # Check for NaNs
    missing_count = combined_df.isnull().sum().sum()
    print(f"Found {missing_count} missing values.")
    
    if missing_count > 0:
        # Strategy: Forward fill (time-series friendly) then backward fill for any leading NaNs
        # However, since we concatenated multiple cars, simple ffill might bleed data across cars.
        # Ideally, we group by Silo_ID, but for a general cleanup:
        # Mean imputation is safer for general features if we ignore time-series continuity for now,
        # but let's try a smart fill:
        
        # Numeric columns: fill with column mean
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
        
        # Categorical columns (if any NaNs): fill with mode
        # (Driver_Style, Fuel_Type shouldn't be nan based on generator, but good practice)
        cat_cols = combined_df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
             combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
             
    # C. Re-ordering / Selecting Feature Columns
    # Ensure consistent order
    feature_order = [
        "Silo_ID", "Driver_Style", "Fuel_Type", # Metadata first
        "Engine RPM", "Fuel Pressure (Bar)", "Fuel Trim Short-Term (%)", 
        "Fuel Trim Long-Term (%)", "O2 Sensor Voltage (V)", 
        "Coolant Temp (C)", "Intake Air Temp (C)", "Catalyst Temp (C)",
        "Ground Truth" # Target last
    ]
    
    # Filter to ensure we only have expected columns (and handles if some are missing entirely by creating 0s is risky, but robust)
    # Here we assume the generator made these.
    combined_df = combined_df[feature_order]

    # D. Scaling (Min-Max Normalization)
    print("Scaling features...")
    # Select feature columns to scale (exclude ID, categorical, labels)
    cols_to_scale = [
        "Engine RPM", "Fuel Pressure (Bar)", "Fuel Trim Short-Term (%)", 
        "Fuel Trim Long-Term (%)", "O2 Sensor Voltage (V)", 
        "Coolant Temp (C)", "Intake Air Temp (C)", "Catalyst Temp (C)"
    ]
    
    scaler = MinMaxScaler()
    combined_df[cols_to_scale] = scaler.fit_transform(combined_df[cols_to_scale])
    
    # 4. Save Processed Data
    combined_df.to_csv(OUTPUT_PROCESSED_FILE, index=False)
    print(f"Saved CLEANED data to {OUTPUT_PROCESSED_FILE}")
    print(f"Processed shape: {combined_df.shape}")
    print("Sample:\n", combined_df.head())

if __name__ == "__main__":
    process_data()
