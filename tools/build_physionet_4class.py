import numpy as np
import mne
from mne.datasets import eegbci
import os
import shutil

# --- CONFIGURATION ---
SUBJECTS = list(range(1, 110)) # Use ALL 109 subjects
# SUBJECTS = list(range(1, 4)) # Debug mode
TARGET_CHANNELS = ['FC3', 'FC4', 'CP3', 'CZ', 'C3', 'C4', 'PZ', 'CP4']
SFREQ = 250
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EEGnet", "PRETRAIN_DATABASE_4CLASS.npz")

# Runs
RUNS_REST = [1, 2]         # Baseline: Eyes Open / Eyes Closed
RUNS_HANDS = [3, 7, 11, 4, 8, 12] # Moteurs: Left vs Right
RUNS_FEET =  [5, 9, 13, 6, 10, 14] # Moteurs: Feet

def robust_load_data(subject, runs, attempts=5, delay=10):
    """Retries downloading/loading data with a delay."""
    for i in range(attempts):
        try:
            return eegbci.load_data(subject, runs, update_path=True, verbose=False)
        except Exception as e:
            if i < attempts - 1:
                print(f"    ⚠️ Download failed. Retrying in {delay}s...")
                import time
                time.sleep(delay)
            else:
                raise e

def preprocess_physionet_4class():
    print(f"🚀 Starting PhysioNet Processing (4 CLASS: L, R, F, Rest)...")
    print(f"   Subjects: {len(SUBJECTS)}")
    
    mne.set_log_level('WARNING')
    
    X_total = []
    y_total = []
    
    for subject in SUBJECTS:
        try:
            print(f"   Processing Subject {subject}...", end='\r')
            
            # ==========================================
            # PART 1: REST (From Runs 1 & 2) -> Label 3
            # ==========================================
            fnames_rest = robust_load_data(subject, RUNS_REST)
            raws_rest = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_rest]
            # Process each run separately to avoid boundary artifacts
            for raw_tmp in raws_rest:
                mne.rename_channels(raw_tmp.info, lambda x: x.strip('.').upper())
                raw_tmp.pick(TARGET_CHANNELS)
                raw_tmp.filter(2., 40., fir_design='firwin', verbose=False)
                if raw_tmp.info['sfreq'] != SFREQ:
                    raw_tmp.resample(SFREQ, npad="auto", verbose=False)
                
                # Make fixed length events every 3.0s
                # id=3 for Rest
                events = mne.make_fixed_length_events(raw_tmp, id=3, duration=3.0)
                
                epochs = mne.Epochs(raw_tmp, events, event_id=None, tmin=0, tmax=3.0, 
                                    proj=False, baseline=None, verbose=False)
                
                data = epochs.get_data() # (N, 8, 751)
                
                # Crop to 750
                if data.shape[2] > 750: data = data[:, :, :750]
                elif data.shape[2] < 750: continue
                
                X_total.append(data)
                y_total.append(epochs.events[:, -1]) # Should be all 3s

            # ==========================================
            # PART 2: MOTOR (From Runs 3-14) -> Labels 0, 1, 2
            # ==========================================
            # Load Hands (0=Left, 1=Right)
            fnames_hands = robust_load_data(subject, RUNS_HANDS)
            raws_hands = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_hands]
            raw_hands = mne.concatenate_raws(raws_hands)
            
            # Load Feet (2=Feet)
            fnames_feet = robust_load_data(subject, RUNS_FEET)
            raws_feet = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_feet]
            raw_feet = mne.concatenate_raws(raws_feet)
            
            for raw_tmp, task_type in [(raw_hands, 'hands'), (raw_feet, 'feet')]:
                mne.rename_channels(raw_tmp.info, lambda x: x.strip('.').upper()) 
                available_chs = raw_tmp.ch_names
                if any(ch not in available_chs for ch in TARGET_CHANNELS): continue
                
                raw_tmp.pick(TARGET_CHANNELS)
                raw_tmp.filter(2., 40., fir_design='firwin', verbose=False)
                if raw_tmp.info['sfreq'] != SFREQ:
                    raw_tmp.resample(SFREQ, npad="auto", verbose=False)
                
                events, event_id_dict = mne.events_from_annotations(raw_tmp, verbose=False)
                
                # Mapping
                current_event_id = {}
                if task_type == 'hands':
                    if 'T1' in event_id_dict: current_event_id[event_id_dict['T1']] = 0 # Left
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 1 # Right
                else:
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 2 # Feet
                
                if not current_event_id: continue
                
                # Filter events
                valid_events = []
                valid_labels = []
                for ev in events:
                    if ev[2] in current_event_id:
                        valid_events.append(ev)
                        valid_labels.append(current_event_id[ev[2]])
                
                if not valid_events: continue
                valid_events = np.array(valid_events)
                
                # Create epochs
                epochs = mne.Epochs(raw_tmp, valid_events, event_id=None, tmin=0, tmax=3.0, 
                                    proj=False, baseline=None, verbose=False)
                
                data = epochs.get_data()
                if data.shape[2] > 750: data = data[:, :, :750]
                elif data.shape[2] < 750: continue
                
                X_total.append(data)
                y_total.append(np.array(valid_labels))
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted! Saving current progress...")
            break
        except Exception as e:
            print(f"   ⚠️ Error Subject {subject}: {e}")
            continue

    if len(X_total) == 0:
        print("\n❌ Failed to load data.")
        return

    X = np.concatenate(X_total)
    y = np.concatenate(y_total)
    
    print(f"\n✅ Processing Complete!")
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Shape: {X.shape}")
    print(f"   Classes (0=L, 1=R, 2=F, 3=Rest): {unique}")
    print(f"   Counts: {counts}")
    
    print("💾 Saving...")
    np.savez_compressed(OUTPUT_FILE, X=X, y=y)
    print(f"✅ Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_physionet_4class()
