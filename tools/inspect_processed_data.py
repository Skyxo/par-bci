import numpy as np
import glob
import os

DATA_DIR = "processed_data"

def inspect_all():
    print(f"=== 🕵️ INSPECTING {DATA_DIR} ===")
    
    files = glob.glob(os.path.join(DATA_DIR, "X_*.npy"))
    files.sort()
    
    if not files:
        print("❌ No X_*.npy files found.")
        return

    total_epochs = 0
    
    for f in files:
        base_name = os.path.basename(f)
        y_name = base_name.replace("X_", "y_")
        y_path = os.path.join(DATA_DIR, y_name)
        
        print(f"\n📂 File: {base_name}")
        
        try:
            X = np.load(f)
            print(f"   Shape: {X.shape} (Epochs, Chans, Time)")
            
            if os.path.exists(y_path):
                y = np.load(y_path)
                classes, counts = np.unique(y, return_counts=True)
                print(f"   Labels: {dict(zip(classes, counts))}")
                
                # Check for Marker 10
                if 10 in classes:
                    print(f"   ℹ️  Contains 'Rest' (Marker 10)")
                else:
                    print(f"   ℹ️  Action Only (1, 2, 3)")
                    
                total_epochs += X.shape[0]
            else:
                print("   ⚠️  No matching y file found.")
                
        except Exception as e:
            print(f"   ❌ Error reading: {e}")

    print(f"\nΣ TOTAL EPOCHS: {total_epochs}")

if __name__ == "__main__":
    inspect_all()
