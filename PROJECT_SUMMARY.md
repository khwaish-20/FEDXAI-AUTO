# 📘 FEDXAI-AUTO: Project Summary & Technical Guide

**Objective:** Predict vehicle component failures caused by **E20 (Ethanol-blended fuel)** using a privacy-preserving Federated AI model, and explain *why* a failure is predicted using XAI.

---

## 1. 🏭 Data Factory: The Digital Twin
*How we created the data.*

Since real-world E20 failure data is scarce, we built a **Physics-Based Simulator** to create a "Digital Twin" of a car engine.
*   **Script:** `generate_fedxai_data.py`
*   **The Engine:** A Python class (`VehiclePhysicsModel`) that simulates thermodynamic and fluid dynamic laws.
*   **E20 Factors:** We programmed specific degradation curves:
    *   **Fuel Filter:** Clogs 1.2x faster due to ethanol sludge.
    *   **Fuel Pump:**  Wears out 1.5x faster due to lower lubricity.
    *   **Cooling:** Heat dissipation drops to 60% efficiency.
*   **Output:** Generated **10,000 unique vehicle silos**, each with slightly different driving behaviors and sensor noise.

## 2. 🧹 Data Preprocessing
*How we cleaned the data for the AI.*

*   **Script:** `data_preprocessing.py`
*   **Step 1: Cleaning:** Imputes missing values (NaNs) caused by simulated sensor glitches.
*   **Step 2: Standardization:** Converts mixed units (e.g., some cars report PSI, others Bar) into a standard format.
*   **Step 3: Scaling:** Uses `MinMaxScaler` to squash all sensor values between `0` and `1`. This is crucial for Neural Networks to learn effectively.
*   **Result:** `combined_processed_data.csv` (The clear, "fuel-grade" data).

## 3. 🧠 AI Model Training
*The Brain of the system.*

*   **Script:** `phase4_realistic.py` (Efficiency Optimized)
*   **Architecture:** **1D-CNN (Convolutional Neural Network)**.
    *   It treats sensor data like an image, scanning for "shapes" (patterns) in the time-series data (e.g., a sudden pressure drop).
*   **Special Trigger:** **Weighted Loss Function**.
    *   Since failures are rare (1%), the AI might get lazy and just guess "Healthy" every time.
    *   We force it to pay **5x more attention** to failures, ensuring it doesn't miss critical alerts.
*   **Performance:**
    *   **Accuracy:** 98.64%
    *   **Size:** 22KB (Tiny enough to fit on a cheap ESP32 chip).

## 4. 🔍 Explainable AI (XAI) & The "Failure Plot"
*Why did the AI say "Failure"?*

This is the "Trust" layer. We use **SHAP (Shapley Additive Explanations)** to interpret the "black box" model.

### 📉 Global Analysis (`shap_summary_plot.png`)
*   **What it is:** A bird's-eye view of what matters.
*   **Insight:** It tells us which sensors are generally the most important.
    *   *Example:* "For this E20 model, **Fuel Pressure** is the #1 predictor of failure, followed by **Long Term Fuel Trim**."

### 🚨 Local Failure Analysis (`shap_force_plot_failure.html`)
*   **What it is:** A deep-dive CSI investigation into **one specific broken car**.
*   **The Visualization:**
    *   **Base Value:** The average predicted risk.
    *   **🔴 Red Bars (Push to Limit):** Features pushing the risk **HIGHER**.
        *   *Example:* "Fuel Pressure = 2.1 Bar" (Way too low!).
    *   **🔵 Blue Bars (Safety Net):** Features pushing risk **LOWER**.
        *   *Example:* "RPM = 2000" (Normal range).
*   **The Verdict:** If the Red bars overpower the Blue, the prediction tips into "FAILURE". This tells the mechanic exactly what to fix (*"Replace the Fuel Pump"*).

## 5. 🖥️ The Dashboard
*The User Interface.*

*   **File:** `phase4_web/index.html`
*   **Tech:** HTML5, TailwindCSS, Glassmorphism UI.
*   **Function:** Connects to the OBD-II dongle via Bluetooth (BLE) to show:
    *   **Live Gauges:** RPM, Temp, Pressure.
    *   **RUL:** Remaining Useful Life (predictive countdown).
    *   **Alerts:** "CRITICAL: Fuel Pump Failure Imminent" (driven by the AI).

---

## 🚧 Status: Work In Progress
 The core AI model and simulation components are complete (Phase 1-3).
However, the **Real-World Architecture** (Physical OBD-II Dongle integration) and final **User Interface** refinements are still **UNDER CONSTRUCTION** (Phase 4).
