# 🏁 How to Run the FINAL AI Training (Full 80 Epochs)

Follow these steps to reproduce the exact 98.6% accuracy model used in our paper.

### 1. Open the File
In VS Code's file explorer (left side), double-click to open:
> **`phase4_realistic.py`**

*This is the full training script that contains the CNN architecture, data loading, and the configuration for 80 training epochs.*

### 2. Open the Terminal
Go to the top menu:
*   **View** -> **Terminal**
*   *Or press* `Ctrl + ~`

### 3. Run the Training Command
We will skip "activating" (which can be error-prone) and run Python directly.
Copy and paste this **single command** into your terminal:

```powershell
.\.venv\Scripts\python.exe phase4_realistic.py
```

### 👁️ What to Expect
This process will take **5-10 minutes** to complete.

1.  **Training**: You will see it run from `Epoch 1/80` to `Epoch 80/80`.
2.  **Evaluation**: It will print the final "accuracy" and "precision" numbers.
3.  **Saving**: It will create `fedxai_edge_realistic.keras` (the brain file) and `phase4_edge/fedxai_model.h` (the C++ header for the car).

**Note:** This will overwrite any existing model files with your new training results.
