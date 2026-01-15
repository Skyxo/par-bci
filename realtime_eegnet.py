import time
import numpy as np
import torch
import pygame
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import sosfilt, butter, iirnotch

# Import EEGNet
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'EEGnet'))
from pretrain_eegnet import EEGNet

# ================= CONFIGURATION =================
# Hardware
SIMULATION_MODE = False
COM_PORT = "COM10"
BOARD_ID = BoardIds.CYTON_BOARD.value

# Signal Processing
SFREQ = 250
WINDOW_SECONDS = 3.0  # Must match training (3s)
WINDOW_SAMPLES = int(WINDOW_SECONDS * SFREQ) # 750
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7] # 8 Channels

# Filtering (Must match training!)
# Training used: Notch 50Hz, Bandpass 2-40Hz
NOTCH_FREQ = 50.0
L_FREQ = 2.0
H_FREQ = 40.0

# Model
MODEL_PATH = os.path.join("models", "eegnet_best.pth")
CLASSES = ["Gauche", "Droite", "Pieds", "Repos"]
COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (100, 100, 100)] # R, G, B, Gray

# ================= ONLINE FILTER CLASS =================
class OnlineFilter:
    def __init__(self, fs, notch_freq, l_freq, h_freq, n_chan=8):
        self.fs = fs
        self.n_chan = n_chan
        
        # 1. Notch Filter (50Hz)
        b_notch, a_notch = iirnotch(notch_freq, Q=30, fs=fs)
        # Convert to SOS for stability
        self.sos_notch = matrix_sos(b_notch, a_notch) 
        
        # 2. Bandpass Filter (2-40Hz)
        self.sos_bp = butter(4, [l_freq, h_freq], btype='bandpass', fs=fs, output='sos')
        
        # State (zi) for each channel
        # Shape: (n_sections, n_channels, 2)
        self.zi_notch = np.zeros((self.sos_notch.shape[0], n_chan, 2))
        self.zi_bp = np.zeros((self.sos_bp.shape[0], n_chan, 2))

    def process_chunk(self, chunk):
        """
        Apply filters to a small chunk of new data [n_chan, n_samples]
        Updates internal state automatically.
        """
        # Apply Notch
        # sosfilt operates on the last axis by default (time)
        chunk_notch, self.zi_notch = sosfilt(self.sos_notch, chunk, axis=1, zi=self.zi_notch)
        
        # Apply Bandpass
        chunk_bp, self.zi_bp = sosfilt(self.sos_bp, chunk_bp, axis=1, zi=self.zi_bp) # FIX: Use output of notch!
        
        # Correction: The line above used chunk_bp before definition.
        # Should be:
        chunk_filtered, self.zi_bp = sosfilt(self.sos_bp, chunk_notch, axis=1, zi=self.zi_bp)
        
        return chunk_filtered

def matrix_sos(b, a):
    """Helper to convert tf to sos"""
    from scipy.signal import tf2sos
    return tf2sos(b, a)


# ================= MAIN APP =================
def draw_bar(screen, x, y, w, h, val, color, label, font):
    # Background
    pygame.draw.rect(screen, (30, 30, 30), (x, y, w, h))
    pygame.draw.rect(screen, (100, 100, 100), (x, y, w, h), 1)
    
    # Fill
    fill_h = int(h * val)
    pygame.draw.rect(screen, color, (x, y + h - fill_h, w, fill_h))
    
    # Label
    txt = font.render(f"{label}", True, (200, 200, 200))
    screen.blit(txt, (x + (w-txt.get_width())//2, y + h + 5))
    
    # Val
    txt_v = font.render(f"{int(val*100)}%", True, (255, 255, 255))
    screen.blit(txt_v, (x + (w-txt_v.get_width())//2, y - 25))

def main():
    print("--- 🧠 INITIALIZING REAL-TIME EEGNET ---")
    
    # 1. LOAD MODEL
    device = torch.device("cpu") # Real-time usually CPU is fine/safer for low latency on simple models
    if torch.cuda.is_available():
        print("   -> CUDA Available (but using CPU for stability)")
        
    model = EEGNet(nb_classes=4, Chans=8, Samples=WINDOW_SAMPLES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. INIT BOARD
    params = BrainFlowInputParams()
    if SIMULATION_MODE:
        params.board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        params.serial_port = COM_PORT
        params.board_id = BOARD_ID
    
    try:
        board = BoardShim(params.board_id, params)
        board.prepare_session()
        board.start_stream()
        print("✅ Connected to OpenBCI.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 3. INIT FILTERS
    online_filter = OnlineFilter(SFREQ, NOTCH_FREQ, L_FREQ, H_FREQ, n_chan=8)
    
    # 4. INIT PYGAME
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Real-Time EEGNet (4-Class)")
    font = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 40)
    
    # BUFFERS
    # We need a rolling buffer for inference
    # Size: Enough to hold WINDOW_SAMPLES
    data_buffer = np.zeros((8, WINDOW_SAMPLES))
    
    clock = pygame.time.Clock()
    running = True
    
    # Smoothing
    probs_ma = np.zeros(4)
    alpha = 0.1 # Exponential smoothing factor
    
    print("--- 🚀 STARTED STREAMING ---")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        # A. GET DATA
        data = board.get_board_data() # Get all new data
        if data.shape[1] > 0:
            eeg_new = data[CHANNELS, :] # (8, N)
            
            # B. ONLINE FILTERING (Continuous)
            eeg_filtered = online_filter.process_chunk(eeg_new)
            
            # C. UPDATE ROLLING BUFFER
            # Shift buffer left and append new data
            n_new = eeg_filtered.shape[1]
            if n_new >= WINDOW_SAMPLES:
                data_buffer = eeg_filtered[:, -WINDOW_SAMPLES:]
            else:
                data_buffer = np.roll(data_buffer, -n_new, axis=1)
                data_buffer[:, -n_new:] = eeg_filtered
        
        # D. INFERENCE (Every frame? Or throttled?)
        # Let's do every frame for max responsiveness
        
        # Preprocessing matching training:
        # 1. Scale units (uV to Volts? Training used Volts)
        # BCI data is usually uV. Training script checked > 1.0 and scaled by 1e-6.
        # OpenBCI Cyton returns uV.
        # So we must scale by 1e-6 to match Volts.
        input_tensor = data_buffer * 1e-6 
        
        # 2. Reshape (1, 1, 8, 750)
        input_tensor = input_tensor[np.newaxis, np.newaxis, :, :] 
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # E. SMOOTHING
        probs_ma = (1 - alpha) * probs_ma + alpha * probs
        
        # F. VISUALIZATION
        screen.fill((20, 20, 20))
        
        # Title
        title = font_big.render("MOTEUR DE PENSÉE (EEGNet)", True, (255, 255, 255))
        screen.blit(title, (400 - title.get_width()//2, 30))
        
        # Draw Bars
        bar_w = 100
        bar_h = 300
        start_x = 100
        spacing = 50
        
        for i, (name, col) in enumerate(zip(CLASSES, COLORS)):
            x = start_x + i * (bar_w + spacing)
            draw_bar(screen, x, 150, bar_w, bar_h, probs_ma[i], col, name, font)
        
        # Winner
        best_idx = np.argmax(probs_ma)
        if probs_ma[best_idx] > 0.5: # Confidence Threshold
            res_txt = font_big.render(f"DÉTECTION : {CLASSES[best_idx].upper()}", True, COLORS[best_idx])
            pygame.draw.circle(screen, COLORS[best_idx], (400, 530), 20)
        else:
            res_txt = font_big.render("INCERTAIN...", True, (100, 100, 100))
            
        screen.blit(res_txt, (400 - res_txt.get_width()//2, 510))
            
        pygame.display.flip()
        clock.tick(30) # 30 FPS

    # CLEANUP
    board.stop_stream()
    board.release_session()
    pygame.quit()

if __name__ == "__main__":
    main()
