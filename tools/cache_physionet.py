import numpy as np
import mne
from mne.datasets import eegbci
import os

# --- CONFIGURATION ---
SUBJECTS = list(range(1, 110)) # Use ALL 109 subjects
# SUBJECTS = list(range(1, 4)) # Debug mode
TARGET_CHANNELS = ['FC3', 'FC4', 'CP3', 'CZ', 'C3', 'C4', 'PZ', 'CP4']
SFREQ = 250
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EEGnet", "physionet_cache_v2.npz")
RUNS_HANDS = [3, 7, 11, 4, 8, 12] # Real + Imag
RUNS_FEET =  [5, 9, 13, 6, 10, 14] # Real + Imag

def robust_load_data(subject, runs, attempts=5, delay=10):
    """Retries downloading/loading data with a delay."""
    for i in range(attempts):
        try:
            # verbose=True shows the download progress which is helpful here
            return eegbci.load_data(subject, runs, update_path=True, verbose=True)
        except Exception as e:
            if i < attempts - 1:
                print(f"\n    ⚠️ Download failed (Attempt {i+1}/{attempts}). Retrying in {delay}s... Error: {e}")
                import time
                time.sleep(delay)
            else:
                raise e

def preprocess_physionet():
    print(f"🚀 Starting PhysioNet Processing & Caching...")
    print(f"   Subjects: {len(SUBJECTS)}")
    print(f"   Target: {OUTPUT_FILE}")
    print(f"   [TIP] Press Ctrl+C at any time to stop and SAVE current progress.")
    
    mne.set_log_level('WARNING')
    
    X_total = []
    y_total = []
    
    for subject in SUBJECTS:
        try:
            print(f"   Processing Subject {subject}...", end='\r')
            
            # --- PART 1: HANDS (Left vs Right) ---
            fnames_hands = robust_load_data(subject, RUNS_HANDS)
            raws_hands = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_hands]
            raw_hands = mne.concatenate_raws(raws_hands)
            
            # --- PART 2: FEET (Feet) ---
            fnames_feet = robust_load_data(subject, RUNS_FEET)
            raws_feet = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_feet]
            raw_feet = mne.concatenate_raws(raws_feet)
            
            # Combine logic
            for raw_tmp, task_type in [(raw_hands, 'hands'), (raw_feet, 'feet')]:
                
                # 1. Standardize Channels
                mne.rename_channels(raw_tmp.info, lambda x: x.strip('.').upper()) 
                
                # Check availability
                available_chs = raw_tmp.ch_names
                missing = [ch for ch in TARGET_CHANNELS if ch not in available_chs]
                if missing: continue

                # 2. Select Channels
                raw_tmp.pick(TARGET_CHANNELS)
                
                # 3. Filter & Resample
                raw_tmp.filter(2., 40., fir_design='firwin', verbose=False)
                if raw_tmp.info['sfreq'] != SFREQ:
                    raw_tmp.resample(SFREQ, npad="auto", verbose=False)
                
                # 4. Epoching
                events, event_id_dict = mne.events_from_annotations(raw_tmp, verbose=False)
                
                current_event_id = {}
                if task_type == 'hands':
                    # T1=Left(0), T2=Right(1)
                    if 'T1' in event_id_dict: current_event_id[event_id_dict['T1']] = 0
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 1
                else:
                    # T2=Feet(2)
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 2
                
                if not current_event_id: continue
                
                valid_events = []
                valid_labels = []
                for ev in events:
                    code = ev[2]
                    if code in current_event_id:
                        valid_events.append(ev)
                        valid_labels.append(current_event_id[code])
                
                if not valid_events: continue
                valid_events = np.array(valid_events)
                
                epochs = mne.Epochs(raw_tmp, valid_events, event_id=None, tmin=0, tmax=3.0, 
                                    proj=False, baseline=None, verbose=False)
                
                data = epochs.get_data() # (N, 8, 751)
                
                # Crop to 750 (3s)
                if data.shape[2] > 750:
                    data = data[:, :, :750]
                elif data.shape[2] < 750:
                    continue # Skip incomplete
                
                X_total.append(data)
                y_total.append(np.array(valid_labels))
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by User! Saving data collected so far...")
            break
        except Exception as e:
            print(f"   ⚠️ Error Subject {subject}: {e}")
            continue
            
    if len(X_total) == 0:
        print("\n❌ Failed to load any data.")
        return

    X = np.concatenate(X_total)
    y = np.concatenate(y_total)
    
    print(f"\n✅ Processing Complete!")
    print(f"   Total Epochs: {X.shape[0]}")
    print(f"   Shape: {X.shape}")
    print(f"   Disk Size: {X.nbytes / 1e6:.2f} MB")
    
    print("💾 Saving to cache file...")
    np.savez_compressed(OUTPUT_FILE, X=X, y=y)
    print(f"✅ Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_physionet()
