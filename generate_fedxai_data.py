import pandas as pd
import numpy as np
import os
import random
from typing import Dict, List

# Configuration
OUTPUT_DIR = "simulated_data"
NUM_SILOS = 10000  # Generating 10000 vehicles as per spec
AI4I_PATH = "ai4i2020.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VehiclePhysicsModel:
    def __init__(self, driver_style, fuel_type, component_target, ai4i_stats):
        self.driver_style = driver_style
        self.fuel_type = fuel_type
        self.component_target = component_target
        self.ai4i_stats = ai4i_stats
        
        # Initial State
        self.pump_wear = 0.0
        self.filter_clog = 0.0
        self.cooling_efficiency = 1.0
        if component_target == 'Heat Dissipation Failure':
            self.cooling_efficiency = 0.6  # Pre-compromised
        
        # Base Parameters derived from AI4I or Spec
        self.base_rpm_mean = ai4i_stats['Rotational speed [rpm]']['mean']
        self.base_rpm_std = ai4i_stats['Rotational speed [rpm]']['std']
        
        # E20 Multiplier
        self.wear_multiplier = 1.5 if fuel_type == 'E20' else 1.0
        
        # Time tracking
        self.simulation_time = 0

    def step(self, dt=1.0):
        # 1. Driver Input (Throttle)
        if self.driver_style == 'Aggressive':
            throttle = np.random.beta(5, 2) # Higher average throttle
        else:
            throttle = np.random.beta(2, 5) # Lower average throttle
            
        # 2. Engine RPM (0-7000)
        # Scale AI4I RPM (mean ~1500) to behave like an engine
        # Idle ~800, Redline ~7000
        base_rpm = 800 + (6200 * throttle)
        noise = np.random.normal(0, 50)
        rpm = np.clip(base_rpm + noise, 0, 7000)
        
        # 3. Fault Injection Logic
        
        # Fault A: Fuel Filter Clogging
        # Rate: 1% per 10 hours (Aggressive) -> 0.001 per hour -> 0.001/3600 per sec
        # Simplified: We are simulating a "drive cycle", so we'll accelerate wear for the sake of data generation
        clog_rate = 0.0001 if self.driver_style == 'Aggressive' else 0.00005
        if self.fuel_type == 'E20':
            clog_rate *= 1.2 # E20 slightly dirtier/cleaning effect clogging filters
        
        if self.component_target == 'Filter':
            self.filter_clog += clog_rate * dt * 100 # Accelerated for detection in short window
            
        # Fault B: Fuel Pump Efficiency Loss
        # Drops 0.1% per day normally. Accelerated 1.5x by E20.
        pump_decay = 0.00001 * self.wear_multiplier
        if self.component_target == 'Pump':
            self.pump_wear += pump_decay * dt * 1000 # Accelerated
            
        # Fault C: Heat
        # Modeled in Temp calculation

        # 4. Sensor Outputs
        
        # Fuel Pressure (Bar)
        # Nominal: 3.5 Bar.
        # Drop due to wear and clog.
        # Pump wear directly reduces max pressure.
        # Filter clog increases pressure drop (Pressure = Source - Drop).
        target_pressure = 3.5 + (0.5 * throttle) # Pressure rises slightly with load
        pressure_loss = (self.pump_wear * 10) + (self.filter_clog * 0.05)
        fuel_pressure = np.clip(target_pressure - pressure_loss, 0, 10)
        
        # Fuel Trim (%)
        # Logic: If pressure is low, Trim goes Positive to compensate.
        # Nominal 0. +/- 25% max.
        long_term_trim = (3.5 - fuel_pressure) * 10 # approximate
        short_term_trim = np.random.normal(0, 1) + long_term_trim
        
        # O2 Sensor Voltage (V)
        # Oscillates 0.1-0.9. Lean (low pressure) -> Low voltage avg. Rich -> High.
        # If Trim is high (compensating), O2 might be erratic.
        o2_bias = -0.3 if fuel_pressure < 2.5 else 0
        o2_volts = np.clip(np.random.normal(0.45 + o2_bias, 0.1) + 0.4 * np.sin(self.simulation_time), 0.1, 0.9)
        
        # Coolant Temp (C)
        # 80-100 Normal.
        # Rises with RPM.
        thermal_load = (rpm / 7000) * 100
        cooling_capacity = 100 * self.cooling_efficiency
        temp_equilibrium = 80 + (thermal_load / cooling_capacity) * 20
        # Simple lag filter
        coolant_temp = temp_equilibrium + np.random.normal(0, 0.5)
        if self.component_target == 'Heat Dissipation Failure' and self.cooling_efficiency < 0.8:
             coolant_temp += 15 # Overheat
        
        # Intake Air Temp (C)
        iat = np.random.normal(25, 2)
        
        # Catalyst Temp (C)
        cat_temp = 400 + (rpm/10) + np.random.normal(0, 10)
        
        self.simulation_time += dt
        
        return {
            "Engine RPM": int(rpm),
            "Fuel Pressure (Bar)": round(fuel_pressure, 2),
            "Fuel Trim Short-Term (%)": round(short_term_trim, 2),
            "Fuel Trim Long-Term (%)": round(long_term_trim, 2),
            "O2 Sensor Voltage (V)": round(o2_volts, 3),
            "Coolant Temp (C)": round(coolant_temp, 1),
            "Intake Air Temp (C)": round(iat, 1),
            "Catalyst Temp (C)": round(cat_temp, 1),
            "Ground Truth": 1 if self.component_target != 'Healthy' else 0
        }

def load_ai4i_stats(path):
    df = pd.read_csv(path)
    stats = {}
    for col in ['Rotational speed [rpm]', 'Torque [Nm]', 'Process temperature [K]']:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    return stats

def generate_silo(silo_id, ai4i_stats):
    # Randomize Profiles
    driver_style = np.random.choice(['Aggressive', 'Conservative'])
    fuel_type = np.random.choice(['E20', 'Standard'], p=[0.6, 0.4])
    target = np.random.choice(['Pump', 'Filter', 'Heat Dissipation Failure', 'Healthy'], p=[0.2, 0.2, 0.1, 0.5])
    
    # Init Physics
    vehicle = VehiclePhysicsModel(driver_style, fuel_type, target, ai4i_stats)
    
    # Run Simulation (e.g., 60 seconds drive cycle at 1Hz)
    data_points = []
    for _ in range(60):
        data = vehicle.step(dt=1.0)
        data_points.append(data)
        
    df = pd.DataFrame(data_points)
    
    # Data Quality Constraints
    
    # 1. Sensor Noise (Added in physics, but adding extra Gaussian layer as requested)
    sensor_cols = ["Fuel Pressure (Bar)", "O2 Sensor Voltage (V)", "Coolant Temp (C)"]
    for col in sensor_cols:
        df[col] += np.random.normal(0, 0.05, len(df))
    
    # 2. Data Heterogeneity (10% files use PSI for Pressure)
    if np.random.random() < 0.1:
        df.rename(columns={"Fuel Pressure (Bar)": "Fuel Pressure (PSI)"}, inplace=True)
        df["Fuel Pressure (PSI)"] = df["Fuel Pressure (PSI)"] * 14.5038
        
    # 3. Missing Values (Random drop)
    if np.random.random() < 0.05: # 5% of files
        mask = np.random.random(len(df)) < 0.005 # Drop 0.5%
        df[mask] = np.nan
        
    # Metadata columns
    df['Silo_ID'] = silo_id
    df['Driver_Style'] = driver_style
    df['Fuel_Type'] = fuel_type
    
    # Save
    filename = f"car_{silo_id}_{target.replace(' ', '_')}.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def main():
    print("Loading AI4I Stats...")
    try:
        stats = load_ai4i_stats(AI4I_PATH)
    except Exception as e:
        print(f"Error loading AI4I dataset: {e}")
        # Fallback stats
        stats = {'Rotational speed [rpm]': {'mean': 1500, 'std': 100}}
        
    print(f"Generating {NUM_SILOS} silos...")
    for i in range(NUM_SILOS):
        generate_silo(i, stats)
        if i % 100 == 0:
            print(f"Generated {i}...")
            
    print(f"Done. Data saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
