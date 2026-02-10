# PRETRAIN_DATABASE.npz Documentation

**Filename:** `PRETRAIN_DATABASE.npz`
**Source:** PhysioNet EEG Motor Movement/Imagery Dataset (WARNING: Created from 109 subjects)

## 📌 Data Structure
The file is a compressed NumPy archive (`.npz`) containing two arrays:

### 1. `X` (Signals)
- **Shape:** `(N, 8, 750)`
    - `N`: Number of epochs (Total ~14,769 samples)
    - `8`: Number of EEG Channels
    - `750`: Time points (3 seconds @ 250Hz)

### 2. `y` (Labels)
- **Shape:** `(N,)`
- **Encoding:**
    - `0`: **Left Hand** (Imagery/Real)
    - `1`: **Right Hand** (Imagery/Real)
    - `2`: **Feet** (Imagery/Real)

## 🧠 Channel Mapping (Indices 0-7)
The signals correspond to the following electrodes (Standard 10-20 system):
0. **FC3** (Frontal-Central Left)
1. **FC4** (Frontal-Central Right)
2. **CP3** (Centro-Parietal Left)
3. **Cz**  (Central Midline - Sensory-motor integration)
4. **C3**  (Central Left - Primary Motor Cortex / Right Hand control)
5. **C4**  (Central Right - Primary Motor Cortex / Left Hand control)
6. **Pz**  (Parietal Midline)
7. **CP4** (Centro-Parietal Right)

## ⚙️ Processing Details
- **Sampling Rate:** 250 Hz
- **Filtering:** Bandpass 2-40 Hz
- **Epoch Length:** 3 seconds (0 to 3s relative to marker)
- **Tasks Included:**
    - Motor Imagery (Hands/Feet)
    - Real Motor Execution (Hands/Feet)
