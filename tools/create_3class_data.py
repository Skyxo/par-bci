
import numpy as np
import os
import glob

INPUT_DIR = "processed_data"
OUTPUT_DIR = "processed_data_3_class"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_files = glob.glob(os.path.join(INPUT_DIR, "X_*.npy"))

print(f"🔄 Filtering data from {INPUT_DIR} to {OUTPUT_DIR} (Removing Class 10/Rest)...")

for f_X in X_files:
    # Derive Y filename
    base_name = os.path.basename(f_X)
    f_Y = os.path.join(INPUT_DIR, base_name.replace("X_", "y_"))
    
    if not os.path.exists(f_Y):
        print(f"⚠️ Missing Y file for {base_name}, skipping.")
        continue
        
    # Load
    X = np.load(f_X)
    y = np.load(f_Y)
    
    # Filter: Keep only where y != 10
    mask = (y != 10)
    X_filt = X[mask]
    y_filt = y[mask]
    
    if len(X_filt) == 0:
        print(f"⚠️ {base_name}: No data left after filtering! (Original: {len(X)})")
        continue

    # Save
    out_X = os.path.join(OUTPUT_DIR, base_name)
    out_y = os.path.join(OUTPUT_DIR, base_name.replace("X_", "y_"))
    
    np.save(out_X, X_filt)
    np.save(out_y, y_filt)
    
    print(f"✅ {base_name}: {len(X)} -> {len(X_filt)} samples saved.")

print("Done.")
