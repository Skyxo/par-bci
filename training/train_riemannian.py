import pygame
import sys
import os
import time
import threading
import numpy as np
import pandas as pd
import mne
import joblib
import glob
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SFREQ = 250
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']

# Colors
BG_COLOR = (15, 15, 20)
TEXT_COLOR = (220, 220, 230)
ACCENT_COLOR = (0, 180, 255)
SUCCESS_COLOR = (50, 200, 80)
ERROR_COLOR = (255, 80, 80)

class TrainingGUI:
    def __init__(self):
        pygame.init()
        self.width, self.height = 600, 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BCI Model Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        self.status_message = "Initializing..."
        self.sub_message = ""
        self.progress = 0.0
        self.finished = False
        self.success = False
        self.accuracy = 0.0
        
        # Threading for non-blocking UI
        self.thread = threading.Thread(target=self.run_training)
        self.thread.start()
        
    def get_all_csvs(self):
        pattern = os.path.join(BASE_DIR, "EEG_Session_*.csv")
        files = glob.glob(pattern)
        return sorted(files)

    def run_training(self):
        time.sleep(1) # Visual pause
        
        # 1. FIND DATA
        self.status_message = "Searching for data..."
        self.progress = 0.1
        csv_files = self.get_all_csvs()
        
        if not csv_files:
            self.status_message = "Error: No CSV file found."
            self.sub_message = "Please run Acquisition first."
            self.finished = True
            self.success = False
            return
            
        self.sub_message = f"Found {len(csv_files)} sessions."
        time.sleep(1)
        
        # 2. LOAD & MERGE
        self.status_message = "Loading & Merging sessions..."
        self.progress = 0.2
        raw_list = []
        
        try:
            for fname in csv_files:
                try:
                    df = pd.read_csv(fname, sep='\t', header=None)
                except:
                    df = pd.read_csv(fname, sep=',', header=None)
                
                # Check shape
                if df.shape[1] < 24: continue

                eeg = df.iloc[:, 1:9].values.T * 1e-6
                markers = df.iloc[:, 23].values
                
                info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
                raw_tmp = mne.io.RawArray(eeg, info, verbose=False)
                
                # Add Stim channel for markers (MNE standard way to preserve markers in concat)
                stim_info = mne.create_info(['STI'], SFREQ, ['stim'])
                stim_raw = mne.io.RawArray(markers.reshape(1, -1), stim_info, verbose=False)
                raw_tmp.add_channels([stim_raw], force_update_info=True)
                
                raw_list.append(raw_tmp)
            
            if not raw_list:
                raise Exception("No valid data found in files")
                
            raw = mne.concatenate_raws(raw_list)
            
            # Extract markers back from Stim channel
            events = mne.find_events(raw, stim_channel='STI', verbose=False)
            
            # Filter
            raw.pick_types(eeg=True) # Drop stim for filtering
            raw.notch_filter([50, 100], fir_design='firwin', verbose=False)
            raw.filter(8., 30., fir_design='firwin', verbose=False)
            
        except Exception as e:
            self.status_message = f"Error Loading: {str(e)[:20]}..."
            print(e)
            self.finished = True
            self.success = False
            return

        # 3. EPOCHING
        self.status_message = "Extracting Epochs..."
        self.progress = 0.5
        
        # Events are already extracted via find_events
        # We need to map them. BrainFlow markers are integers.
        # Check event codes in events array
        
        event_id = {'LEFT': 1, 'RIGHT': 2, 'FEET': 3, 'REST': 10}
        # Filter events to only keep what we want
        # mne.Epochs will ignore events not in event_id if we pass event_id
        
        try:
            # IMPORTANT: Rest phase is only 2.0s long. We must stop before 2.0s to avoid Action overlap.
            # Window 0.5s to 2.0s = 1.5s duration.
            epochs = mne.Epochs(raw, events, event_id, tmin=0.5, tmax=2.0, 
                               proj=False, baseline=None, verbose=False, preload=True)
        except:
            self.status_message = "Error: No valid markers found."
            self.finished = True
            self.success = False
            return
            
        if len(epochs) < 5:
            self.status_message = "Error: Not enough trials."
            self.sub_message = f"Found only {len(epochs)} epochs."
            self.finished = True
            self.success = False
            return

        # 4. TRAINING
        self.status_message = "Training Model..."
        self.progress = 0.7
        
        try:
            clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(), 
                               LogisticRegression(solver='lbfgs', max_iter=1000))
            
            X_train = epochs.get_data()
            y_train = epochs.events[:, -1]
            clf.fit(X_train, y_train)
            
            self.accuracy = clf.score(X_train, y_train)
            
            # --- DEBUG INFO ---
            from sklearn.metrics import classification_report, confusion_matrix
            y_pred = clf.predict(X_train)
            print("\n--- CLASSIFICATION REPORT ---")
            print(classification_report(y_train, y_pred, target_names=[k for k in event_id.keys()]))
            print("--- CONFUSION MATRIX ---")
            print(confusion_matrix(y_train, y_pred))
            print("-----------------------------\n")

            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR)
            joblib.dump(clf, os.path.join(MODELS_DIR, "riemann_model.pkl"))
            
        except Exception as e:
            self.status_message = f"Training Failed: {str(e)}"
            self.finished = True
            self.success = False
            return

        self.status_message = "Training Complete!"
        self.sub_message = f"Accuracy (on train set): {self.accuracy*100:.1f}%"
        self.progress = 1.0
        self.finished = True
        self.success = True

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        # Center Box
        cx, cy = self.width // 2, self.height // 2
        
        # Status Text
        status_surf = self.font.render(self.status_message, True, TEXT_COLOR)
        status_rect = status_surf.get_rect(center=(cx, cy - 40))
        self.screen.blit(status_surf, status_rect)
        
        # Sub Message
        sub_surf = self.font_small.render(self.sub_message, True, (150, 150, 150))
        sub_rect = sub_surf.get_rect(center=(cx, cy))
        self.screen.blit(sub_surf, sub_rect)
        
        # Progress Bar
        bar_w = 400
        bar_h = 10
        pygame.draw.rect(self.screen, (40, 40, 50), (cx - bar_w//2, cy + 40, bar_w, bar_h), border_radius=5)
        
        fill_w = int(self.progress * bar_w)
        if fill_w > 0:
            col = SUCCESS_COLOR if self.success else ACCENT_COLOR
            if self.finished and not self.success: col = ERROR_COLOR
            pygame.draw.rect(self.screen, col, (cx - bar_w//2, cy + 40, fill_w, bar_h), border_radius=5)
            
        # Instruction
        if self.finished:
            instr = "Press SPACE to Close"
            instr_surf = self.font_small.render(instr, True, (100, 100, 100))
            instr_rect = instr_surf.get_rect(center=(cx, self.height - 30))
            self.screen.blit(instr_surf, instr_rect)

    def run(self):
        running = True
        while running:
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if self.finished:
                        running = False
        
        pygame.quit()

if __name__ == "__main__":
    app = TrainingGUI()
    app.run()
