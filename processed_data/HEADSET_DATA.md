# HEADSET_DATA Documentation

## 📌 Data Format
The files in this directory are NumPy arrays used for Fine-Tuning.

### Files
- `X_SESSION_NAME.npy`: EEG Signals
- `y_SESSION_NAME.npy`: Labels

### 1. `X` (Signals)
- **Shape:** `(N, 8, 750)` (Strictly enforced by Fine-Tuning)
    - `N`: Number of epochs
    - `8`: Number of EEG Channels
    - `750`: Time points (3 seconds @ 250Hz)

**⚠️ CRITICAL: CHANNEL MAPPING**
The underlying model (`pretrain_eegnet.py` / PhysioNet) was trained on the following channels in this exact order:
1. **FC3**
2. **FC4**
3. **CP3**
4. **Cz**
5. **C3**
6. **C4**
7. **Pz**
8. **CP4**

**Action Required:** Ensure your OpenBCI headset electrodes are placed physically on these locations and connected to pins 1-8 in this order. If your headset produces different channels (e.g., O1, P3...), the model performance will be degraded.

### 2. `y` (Labels)
- **Shape:** `(N,)`
- **Encoding:**
    - `0`: Left Hand
    - `1`: Right Hand
    - `2`: Feet

## ⚙️ Consistency Check
- **Sampling Rate:** 250 Hz
- **Epoch Length:** 
    - The preprocessing scripts typically extract 4.0s epochs.
    - **Confirmed:** The `finetune_eegnet.py` script automatically crops them to the first **3.0s** (750 samples) to match the pre-trained model.
