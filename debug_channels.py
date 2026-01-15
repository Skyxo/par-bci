import mne
import os

# Try to find a file
base_path = os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0")
subject_path = os.path.join(base_path, "S001", "S001R04.edf")

if not os.path.exists(subject_path):
    print(f"File not found at {subject_path}")
    # Try finding any .edf
    import glob
    files = glob.glob(os.path.join(base_path, "*", "*.edf"))
    if files:
        subject_path = files[0]
        print(f"Using found file: {subject_path}")
    else:
        print("No EDF files found.")
        exit()

print(f"Reading {subject_path}...")
raw = mne.io.read_raw_edf(subject_path, preload=False, verbose=False)
print("Channels:", raw.ch_names)
