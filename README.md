<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ESP32--S3-Firmware-000000?logo=espressif&logoColor=white" />
  <img src="https://img.shields.io/badge/TFLite_Micro-Edge_AI-4285F4?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

# 🚗 FEDXAI-AUTO: Federated Explainable AI for Automotive Predictive Maintenance

> **A privacy-preserving, explainable AI framework for real-time vehicle failure prediction — from cloud training to edge deployment on an OBD-II smart dongle.**

---

## 📌 Overview

FEDXAI-AUTO addresses the challenge of building accurate predictive maintenance models for vehicles **without centralizing sensitive driving data**. It combines:

- **Federated Learning** — vehicles collaboratively train a shared model without sharing raw data
- **Differential Privacy** — mathematically guaranteed privacy with calibrated DP noise
- **Explainable AI (XAI)** — SHAP + LIME dual-method explanations translated into mechanic-friendly alerts
- **Edge Deployment** — compressed TFLite model running on a ₹1,385 (~$17) OBD-II dongle

### Key Results

| Metric | Cloud Model (Phase 3) | Edge Model (Phase 4 Float32) | Edge Model (Phase 5 QAT INT8) |
|--------|:---:|:---:|:---:|
| **Accuracy** | 98.84% | 98.64% | **98.84%** (0% drop) |
| **Recall** | 98.79% | 98.99% | **99.42%** |
| **Precision** | 98.94% | 98.37% | **98.33%** |
| **Model Size** | 456 KB | 22.6 KB | **16.72 KB** (26% smaller) |
| **Target Device** | Server/Cloud | ESP32-S3 MCU | ESP32-S3 (INT8 math) |

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│  Vehicle ECU │───▶│  ELM327 OBD  │───▶│  ESP32-S3 MCU   │───▶│ Mobile App │
│  (8 sensors) │    │  (UART/BLE)  │    │  TFLite Micro   │    │ (Web/Chrome)│
│              │    │              │    │  22.6KB model    │    │ Dashboard  │
│  RPM, Temp,  │    │  PID Reader  │    │  98.64% Acc     │    │ XAI Alerts │
│  Load, etc.  │    │              │    │  <100ms infer    │    │            │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘
```

> See [`FEDXAI_Architecture_Blueprint.md`](FEDXAI_Architecture_Blueprint.md) for the full Mermaid diagram.

---

## 📁 Repository Structure

```
FEDXAI-AUTO/
│
├── 📋 Documentation
│   ├── PHASES_XAI_IMPLEMENTATION_PLAN.md   # Full project plan with results
│   └── FEDXAI_Architecture_Blueprint.md    # System architecture diagram
│
├── 📊 Phase 1 & 2 — Data Pipeline
│   ├── data_preprocessing.py               # Raw → processed data pipeline
│   └── generate_fedxai_data.py             # Digital Twin data generator
│
├── 🧠 Phase 3 — Federated Learning + XAI
│   ├── federated_training_production.py    # FL training with DP + SecAgg
│   ├── xai_analysis.py                     # SHAP + LIME explainability analysis
│   ├── generate_report.py                  # Model evaluation script
│   ├── fedxai_production_best.keras        # Best cloud model (98.84%)
│   ├── shap_summary_plot.png               # SHAP global feature importance
│   ├── shap_force_plot_failure.html        # SHAP interactive failure explanation
│   ├── lime_explanation_failure.png        # LIME local explanation (failure)
│   ├── lime_explanation_healthy.png        # LIME local explanation (healthy)
│   └── lime_feature_importance.png         # LIME aggregated feature importance
│
├── 🔧 Phase 4 & 5 — Edge Deployment & Quantization
│   ├── phase4_realistic.py                 # TinyML pipeline (train + convert)
│   ├── phase5_qat_int8.py                  # QAT training & full INT8 conversion
│   ├── fedxai_edge_realistic.keras         # Edge Keras model (98.64%)
│   └── phase4_edge/
│       ├── main.cpp                        # ESP32-S3 firmware (C++)
│       ├── platformio.ini                  # PlatformIO build config
│       ├── fedxai_model.h                  # Embedded QAT INT8 model C header
│       ├── fedxai_realistic.tflite         # Float32 TFLite model (22.6 KB)
│       └── fedxai_qat_int8.tflite          # QAT INT8 TFLite model (16.7 KB)
│
└── .gitignore
```

---

## 🔬 Methodology

### Phase 1: Foundation & Baseline
- Validated against the **AI4I 2020 Predictive Maintenance Dataset**
- Established baseline metrics and data preprocessing pipeline

### Phase 2: Digital Twin & Data Factory
- Built a physics-based vehicle simulator generating realistic OBD-II sensor data
- 500 vehicles × 60 timesteps × 8 features (RPM, coolant temp, engine load, etc.)
- Simulated failure modes: overheating, wear, power failure, overstrain, random

### Phase 3: Federated XAI Framework
- **Architecture:** CNN + LSTM hybrid (34,721 params)
- **Training:** 2-stage approach — centralized pre-training → federated fine-tuning
- **Privacy:** Differential Privacy (ε=1.0, δ=1e-5) + Secure Aggregation
- **XAI:** SHAP analysis with mechanic-friendly alert translation
- **Result:** **98.84% accuracy**, 98.79% recall, 1.21% miss rate

### Phase 4: Edge Deployment (Indian Solution)
- **Model Compression:** CNN+LSTM → Pure CNN (22.6 KB float32, 98.64% accuracy)
- **Hardware:** ESP32-S3-WROOM-1 (~₹400) + ELM327 OBD-II (~₹350)
- **Firmware:** Complete C++ implementation with OBD-II polling, TFLite Micro inference, BLE communication, and XAI alert generation
- **BOM Cost:** ₹1,385 per dongle (~$17 USD)

### Phase 5: QAT & Full INT8 Quantization
- **Quantization-Aware Training (QAT):** Resolved BatchNorm rounding errors by simulating INT8 quantization during training.
- **Edge Model Size:** Compressed from 22.6 KB to **16.72 KB** (INT8) (26% size reduction, under 20 KB limit).
- **Edge Model Metrics:** **98.84% accuracy**, **99.42% recall**, **98.33% precision** (retains 100% cloud model accuracy!).
- **ESP32 benefit:** Uses `tf.float32` boundary I/O for compatibility with INT8 internal math, leveraging ESP32-S3 LX7 vector operations.
- **Status:** All software, firmware, and model compilation completed. The only work left is the physical hardware connection.

---

## 🛠️ Setup & Usage

### Prerequisites
```bash
Python 3.10+
TensorFlow 2.x
scikit-learn
pandas, numpy
shap, lime
matplotlib
```

### Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy shap lime matplotlib
```

### Run Training Pipeline
```bash
# Phase 2: Generate synthetic vehicle data
python generate_fedxai_data.py

# Phase 3: Train federated model
python federated_training_production.py

# Phase 3: Run XAI analysis
python xai_analysis.py

# Phase 4: Train edge model + convert to TFLite
python phase4_realistic.py

# Phase 5: Train QAT model + convert to INT8 TFLite + validate
python phase5_qat_int8.py
```

### Flash ESP32 Firmware
```bash
# Install PlatformIO
pip install platformio

# Build & flash
cd phase4_edge
pio run --target upload
```

---

## 📊 OBD-II Features Used

| # | PID | Feature | Unit | Range |
|---|-----|---------|------|-------|
| 1 | 0x0C | Engine RPM | rpm | 600–8000 |
| 2 | 0x05 | Coolant Temperature | °C | 60–130 |
| 3 | 0x04 | Engine Load | % | 0–100 |
| 4 | 0x0D | Vehicle Speed | km/h | 0–250 |
| 5 | 0x0F | Intake Air Temperature | °C | 15–80 |
| 6 | 0x10 | MAF Air Flow Rate | g/s | 0–650 |
| 7 | 0x11 | Throttle Position | % | 0–100 |
| 8 | 0x2F | Fuel Level | % | 0–100 |

---

## 🎯 Hardware BOM (Indian Market)

| Component | Specification | Cost (₹) |
|-----------|--------------|-----------|
| ESP32-S3-WROOM-1 | Dual-core 240MHz, 512KB SRAM, WiFi+BLE | ~400 |
| ELM327 OBD-II Module | UART interface, all standard PIDs | ~350 |
| OBD-II Connector | 16-pin male, J1962 standard | ~150 |
| Voltage Regulator | LM2596 buck, 12V→3.3V | ~85 |
| PCB + Passives | Custom PCB, capacitors, resistors | ~200 |
| 3D Printed Enclosure | ABS/PLA case | ~200 |
| **Total** | | **~₹1,385 (~$17 USD)** |

---

## 📈 Comparison with Baseline

| Metric | Baseline (2025 Paper) | FEDXAI-AUTO | Improvement |
|--------|:---:|:---:|:---:|
| Accuracy | 98.15% | **98.84%** | +0.69% |
| Privacy | None | **DP (ε=1.0)** | ✅ Added |
| Explainability | None | **SHAP + LIME + Alerts** | ✅ Added |
| Edge Deployment | None | **16.72 KB INT8 TFLite** | ✅ Added (ESP32-S3 optimized) |
| Cost per Unit | N/A | **₹1,385** | ✅ Practical |

---

## 🔮 Future Work

- [ ] Connect ESP32-S3 and ELM327 modules to OBD-II connector (physical wiring/connection)
- [ ] Collect real OBD-II data from Indian vehicles for validation
- [ ] Enhace Web Dashboard with cloud sync
- [ ] Add over-the-air (OTA) model updates via WiFi
- [ ] Expand to heavy commercial vehicles (trucks, buses)
- [ ] Multi-fault classification (currently binary: healthy/failure)

---

## 📄 License

This project is developed as part of academic research. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **AI4I 2020 Dataset** — UCI Machine Learning Repository
- **TensorFlow Lite Micro** — Google Edge AI team
- **SHAP** — Scott Lundberg et al.
- **ESP32** — Espressif Systems

---

<p align="center">
  <b>Built with ❤️ for smarter, safer Indian roads</b>
</p>
