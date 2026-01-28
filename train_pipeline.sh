#!/bin/bash

# ==========================================
# 🧠 BCI TRAINING PIPELINE
# ==========================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Full Training Pipeline..."

# 1. PREPARE DATA
echo "------------------------------------------------"
# echo "📦 STEP 1: Process & Cache PhysioNet Data"
# echo "------------------------------------------------"
# (DISABLED: Using static PRETRAIN_DATABASE.npz)
# python tools/cache_physionet.py

echo "------------------------------------------------"
echo "🛠️ STEP 1.5: Extract Epochs (4 Classes)"
echo "------------------------------------------------"
python tools/extract_epochs_4class.py

# 2. PRE-TRAIN
echo "------------------------------------------------"
echo "🏋️ STEP 2: Pre-train EEGNet (Cross-Subject)"
echo "------------------------------------------------"
python EEGnet/pretrain_eegnet.py

# 3. FINE-TUNE
echo "------------------------------------------------"
echo "🎯 STEP 3: Fine-tune EEGNet (Subject-Specific)"
echo "------------------------------------------------"
python EEGnet/finetune_eegnet.py

echo "------------------------------------------------"
echo "✅ PIPELINE COMPLETE!"
echo "------------------------------------------------"
echo "Outputs saved in 'EEGnet/runs/'"
