
import time
import numpy as np
import scipy.signal
import pygame
import joblib
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from collections import deque
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
COM_PORT = "COM10"
BOARD_ID = BoardIds.CYTON_BOARD.value

# PIPELINE CONFIG
MODEL_REST_PATH = "models/riemann_rest.pkl"
MODEL_MOTOR_PATH = "models/riemann_motor.pkl"

SFREQ = 250
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
N_CHANS = len(CH_NAMES)

# Windowing
WINDOW_REST_SAMPLES = int(1.0 * SFREQ) # 1s
WINDOW_MOTOR_SAMPLES = int(2.0 * SFREQ) # 2s (Crop 1.0-3.0s)
REST_THRESHOLD = 0.7

# UI Config
WIDTH, HEIGHT = 1400, 800
BG_COLOR = (15, 15, 20)
PANEL_COLOR = (25, 25, 35)
ACCENT_COLOR = (0, 200, 255)
TEXT_COLOR = (220, 220, 230)
REST_COLOR = (100, 100, 100)
ACTIVE_COLOR = (255, 100, 200)

# ================= HELPER CLASSES =================

class OnlineFilter:
    """
    Stateful Filter that maintains continuity between chunks.
    Implements the exact same filtering as training:
    1. Notch 50Hz
    2. Bandpass 1-40Hz
    """
    def __init__(self, sfreq, n_chans):
        self.sfreq = sfreq
        self.n_chans = n_chans
        
        # Design Filters (SOS form is more stable for realtime)
        # 1. Notch 50Hz (Bandstop 48-52Hz)
        self.sos_notch = scipy.signal.iirfilter(4, [48, 52], btype='bandstop', fs=sfreq, output='sos')
        self.zi_notch = scipy.signal.sosfilt_zi(self.sos_notch) # Initial State
        self.zi_notch = np.repeat(self.zi_notch[:, np.newaxis, :], n_chans, axis=1) # (Sections, Chans, 2)
        
        # 2. Bandpass 1-40Hz
        self.sos_bp = scipy.signal.iirfilter(4, [1, 40], btype='bandpass', fs=sfreq, output='sos')
        self.zi_bp = scipy.signal.sosfilt_zi(self.sos_bp)
        self.zi_bp = np.repeat(self.zi_bp[:, np.newaxis, :], n_chans, axis=1)
        
    def process(self, new_samples):
        """
        Apply filters to new chunk of data [n_chans, n_samples].
        Updates internal state automatically.
        """
        if new_samples.size == 0: return new_samples
        
        # Apply Notch
        filtered, self.zi_notch = scipy.signal.sosfilt(self.sos_notch, new_samples, axis=1, zi=self.zi_notch)
        
        # Apply Bandpass
        filtered, self.zi_bp = scipy.signal.sosfilt(self.sos_bp, filtered, axis=1, zi=self.zi_bp)
        
        return filtered

# ================= HELPER FUNCTIONS =================

def draw_signal_trace(screen, signal_data, x, y, w, h, scale=1.0, color=(255, 255, 255)):
    """Draws a simple line trace of the signal."""
    if len(signal_data) < 2: return
    
    mid_y = y + h // 2
    pygame.draw.line(screen, (35, 35, 45), (x, mid_y), (x + w, mid_y))
    
    points = []
    n_points = len(signal_data)
    step = max(1, n_points // w) 
    
    for i in range(0, n_points, step):
        val = signal_data[i]
        px = x + (i / n_points) * w
        py = mid_y - (val * scale)
        py = max(y, min(y + h, py))
        points.append((px, py))
        
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 1)

def main():
    # 1. INIT BRAINFLOW
    params = BrainFlowInputParams()
    params.serial_port = COM_PORT
    board = BoardShim(BOARD_ID, params)
    
    try:
        board.prepare_session()
        board.start_stream()
        print("✅ BrainFlow Stream Started")
    except BrainFlowError as e:
        print(f"❌ Error initializing board: {e}")
        return

    # 2. LOAD MODELS (Riemannian)
    print("📥 Loading Models...")
    try:
        clf_rest = joblib.load(MODEL_REST_PATH)
        clf_motor = joblib.load(MODEL_MOTOR_PATH)
        print("✅ Models loaded: Cascade System Ready.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # 3. INIT ONLINE FILTER & BUFFERS
    online_filter = OnlineFilter(SFREQ, N_CHANS)
    
    # We maintain a long rolling buffer of CLEAN data
    # Size = 5 seconds to be safe
    buffer_len = 5 * SFREQ
    clean_buffer = np.zeros((N_CHANS, buffer_len))
    
    # Filter for Inference (Mu Band 8-13Hz) - We apply this ON TOP of clean data for features
    sos_mu = scipy.signal.iirfilter(4, [8, 13], btype='bandpass', fs=SFREQ, output='sos')

    # 4. INIT PYGAME
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BCI REALTIME - CASCADE SYSTEM")
    
    font_header = pygame.font.SysFont("Arial", 24, bold=True)
    font_large = pygame.font.SysFont("Arial", 60, bold=True)
    font_medium = pygame.font.SysFont("Arial", 22)
    
    clock = pygame.time.Clock()
    
    # State
    last_pred_time = time.time()
    
    # Decisions
    state_rest = "Init..."
    prob_rest = 0.5
    state_motor = "..."
    probs_motor = [0.0, 0.0, 0.0]
    
    scale_uV = 0.5
    
    running = True
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_UP: scale_uV *= 1.2
                if event.key == pygame.K_DOWN: scale_uV /= 1.2
                
        # --- 1. ACQUISITION & FILTERING ---
        # Get new data since last call
        data = board.get_board_data() # Gets all data since last call (clears internal buffer)
        
        if data.shape[1] > 0:
            # Scale uV -> V
            raw_chunk = data[1:9, :] * 1e-6
            
            # ONLINE FILTERING (Crucial Step!)
            clean_chunk = online_filter.process(raw_chunk)
            
            # Update Ring Buffer
            roll_len = clean_chunk.shape[1]
            if roll_len > 0:
                clean_buffer = np.roll(clean_buffer, -roll_len, axis=1)
                clean_buffer[:, -roll_len:] = clean_chunk
        
        # --- 2. CASCADE PREDICTION ---
        now = time.time()
        if now - last_pred_time > 0.1: # 10Hz Prediction Rate
            
            # Are we filled enough?
            # Ideally verify valid data but we assume buffer fills quickly
            
            # STAGE 1: REST CHECK (Last 1.0s)
            window_rest = clean_buffer[:, -WINDOW_REST_SAMPLES:]
            
            # Apply Mu Filter (8-13Hz) for Features
            # Note: We filter the already clean (1-40Hz) data to extract Mu band
            w_rest_filt = scipy.signal.sosfilt(sos_mu, window_rest, axis=1)
            feat_rest = w_rest_filt[np.newaxis, :, :]
            
            # Predict
            try:
                probs_r = clf_rest.predict_proba(feat_rest)[0]
                prob_rest = probs_r[1] # Prob of Rest
                
                if prob_rest > REST_THRESHOLD:
                    state_rest = "REPOS 💤"
                    state_motor = "---"
                    probs_motor = [0.0, 0.0, 0.0]
                else:
                    state_rest = "ACTIF ⚡"
                    
                    # STAGE 2: MOTOR CHECK (Last 2.0s)
                    window_motor = clean_buffer[:, -WINDOW_MOTOR_SAMPLES:]
                    w_motor_filt = scipy.signal.sosfilt(sos_mu, window_motor, axis=1)
                    feat_motor = w_motor_filt[np.newaxis, :, :]
                    
                    probs_m = clf_motor.predict_proba(feat_motor)[0]
                    probs_motor = probs_m
                    
                    idx = np.argmax(probs_m)
                    if idx == 0: state_motor = "GAUCHE ⬅️"
                    elif idx == 1: state_motor = "DROITE ➡️"
                    else: state_motor = "PIEDS 🦶"
                    
            except Exception as e:
                print(f"Prediction Error: {e}")

            last_pred_time = now

        # --- 3. UI DRAWING ---
        
        # SIGNALS (Left)
        # Draw 8 channels vertically
        x_sig, y_sig, w_sig, h_sig = 50, 50, 800, 70
        for i in range(8):
            # Draw trace
            draw_signal_trace(screen, clean_buffer[i], x_sig, y_sig + i * (h_sig + 10), w_sig, h_sig, scale=scale_uV, color=ACCENT_COLOR)
            # Draw label
            label = font_medium.render(CH_NAMES[i], True, (120, 120, 120))
            screen.blit(label, (x_sig - 40, y_sig + i * (h_sig + 10) + 20))

        # RESULTS (Right Panel)
        fx = 900
        
        # State Display
        screen.blit(font_header.render("ETAT ACTUEL:", True, TEXT_COLOR), (fx, 50))
        
        if "REPOS" in state_rest:
            current_state = state_rest
            state_col = REST_COLOR
        else:
            current_state = state_motor
            state_col = ACTIVE_COLOR
            
        screen.blit(font_large.render(current_state, True, state_col), (fx, 90))

        # Probabilities
        y_bar = 250
        classes = ["Gauche", "Droite", "Pieds"]
        current_probs = probs_motor
        font_small = pygame.font.SysFont("Arial", 18)

        for i, cls in enumerate(classes):
            prob = current_probs[i]
            
            name_txt = font_medium.render(cls, True, TEXT_COLOR)
            screen.blit(name_txt, (fx, y_bar))
            
            # Bar bg
            pygame.draw.rect(screen, (40, 40, 60), (fx + 120, y_bar + 5, 250, 20))
            # Bar fg
            bar_w = int(250 * prob)
            pygame.draw.rect(screen, ACCENT_COLOR, (fx + 120, y_bar + 5, bar_w, 20))
            
            perc_txt = font_small.render(f"{prob*100:.1f}%", True, TEXT_COLOR)
            screen.blit(perc_txt, (fx + 380, y_bar + 5))
            
            y_bar += 60
            
        # Info
        info_txt = font_small.render(f"Scale: {scale_uV:.1f} uV", True, (150, 150, 150))
        screen.blit(info_txt, (fx, 600))

        pygame.display.flip()
        clock.tick(30)

    board.stop_stream()
    board.release_session()
    pygame.quit()

if __name__ == "__main__":
    main()
