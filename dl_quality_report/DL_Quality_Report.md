# Deep Learning Quality Report

## Zero-Shot Transfer Performance
- **Model**: EEGNet v4 (Pre-trained on BNCI2014001)
- **Test Data**: 180 epochs from OpenBCI sessions.
- **Accuracy**: **35.00%** (Chance Level: 33.3%)
- **Interpretation**:
    - **< 35%**: Random guessing. Likely noise or domain shift (electrode placement mismatch).
    - **35% - 45%**: Weak transfer. Signal contains some motor features.
    - **> 45%**: Strong transfer. High quality, physiological signal.

## Physiological Plausibility
### Spatial Filters (Sensory Motor Cortex)
![Spatial Filters](spatial_filters.png)
*Check if filters focus on C3, C4, Cz (Central Electrodes) or Fp/O (Artifacts).*

### Frequency Saliency (Band Importance)
![Saliency](saliency_map.png)
*Check if Mu/Alpha (8-13Hz) and Beta (13-30Hz) drops cause accuracy loss. High frequency importance usually means EMG noise.*
