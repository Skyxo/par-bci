import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import mne

# ==========================================
# 1. CONFIGURATION FOR OPENBCI
# ==========================================
# OpenBCI Cyton sampling rate is 250Hz
SFREQ = 250 

# Define channels relevant for Motor Imagery (MI)
# Note: For MI, we prioritize C3 (Right Hand), C4 (Left Hand), Cz (Feet).
# We ignore Fp1/Fp2 here as they are mostly artifacts/eye blinks.
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
n_channels = len(CH_NAMES)

# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================
print("Loading real EEG data...")

try:
    X = np.load('../data/processed/X.npy')
    y = np.load('../data/processed/y.npy')
    print(f"Loaded X: {X.shape}, y: {y.shape}")
except FileNotFoundError:
    print("Error: Preprocessed data not found. Run 'preprocess_eeg.py' first.")
    exit()

# Check for valid classes
print(f"Classes found: {np.unique(y)}")

# ==========================================
# 3. PREPROCESSING (Already filtered in preprocess_eeg.py)
# ==========================================
# Data is already bandpass filtered (8-30Hz) and epoched.
# We proceed directly to Riemannian classification.

# ==========================================
# 4. RIEMANNIAN GEOMETRY PIPELINE
# ==========================================
# The "Magic" of this algorithm:
# 1. Covariances: Converts time-series window -> Spatial Covariance Matrix (SCM)
# 2. TangentSpace: Projects the curved SCM manifold to a flat Euclidean space
# 3. Classifier: Standard Logistic Regression works great in Tangent Space
clf = make_pipeline(
    Covariances(estimator='lwf'), # 'lwf' (Ledoit-Wolf) is robust for low sample counts
    TangentSpace(),
    LogisticRegression(solver='lbfgs') 
)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf.fit(X_train, y_train)

print(f"Training Complete. Test Accuracy (on random noise): {clf.score(X_test, y_test):.2f}")
print("Note: Accuracy is random here because input data is noise.")

# ==========================================
# 5. REAL-TIME SIMULATION
# ==========================================
def process_real_time_window(buffer_data):
    """
    Simulate receiving a 2-second buffer from the OpenBCI headset.
    buffer_data shape: (n_channels, n_samples)
    """
    # 1. Reshape to (1, n_channels, n_samples) for the classifier
    # In reality, you MUST apply the same bandpass filter (8-30Hz) here first!
    epoch = buffer_data[np.newaxis, :, :]
    
    # 2. Predict
    prediction = clf.predict(epoch)
    
    # 3. Map to Robot Command
    command_map = {1: "ROBOT_MOVE_LEFT", 2: "ROBOT_MOVE_RIGHT", 3: "ROBOT_MOVE_FEET", 10: "ROBOT_REST"}
    return command_map.get(prediction[0], "UNKNOWN")

# Save the model
import joblib
import os
if not os.path.exists('../models'):
    os.makedirs('../models')
    
model_path = '../models/riemann_model.pkl'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Simulate a "live" buffer coming from the headset
print("\n--- Simulating Real-Time Input ---")
live_buffer = np.random.randn(n_channels, int(2.0 * SFREQ)) # 2 seconds of new data
command = process_real_time_window(live_buffer)
print(f"Detected Intention: {command}")
