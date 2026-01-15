import numpy as np
import pandas as pd
import mne
import pygame
import joblib
import scipy.signal
import os
import glob
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# We prioritize the session with Rest markers
TARGET_SESSION = "EEG_Session_2026-01-14_13-35.csv"

# Models
MODEL_REST_PATH = "models/riemann_rest.pkl"
MODEL_MOTOR_PATH = "models/riemann_motor.pkl"

SFREQ = 250
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']

# Windowing (Must match Realtime/Training)
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

LABELS_MAP = {1: "GAUCHE", 2: "DROITE", 3: "PIEDS", 10: "REPOS", 0: "..."}
COLOR_MAP = {1: (255, 80, 80), 2: (80, 255, 80), 3: (80, 80, 255), 10: (150, 150, 150), 0: (50, 50, 50)}

def load_data():
    # Find file
    if not os.path.exists(TARGET_SESSION):
        print(f"⚠️ {TARGET_SESSION} not found. Searching for latest CSV...")
        files = glob.glob("EEG_Session_*.csv")
        if not files:
            raise FileNotFoundError("No EEG CSV files found.")
        filename = sorted(files)[-1] # Latest
    else:
        filename = TARGET_SESSION
    
    print(f"📂 Loading {filename}...")
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
    except:
        df = pd.read_csv(filename, sep=',', header=None)
        
    # Extract Data
    # Columns 1-8 are EEG channels
    raw_data = df.iloc[:, 1:9].values.T # (8, N)
    
    # Scale uV -> V if needed (Check magnitude)
    if np.mean(np.abs(raw_data)) > 100: 
        print("   Detected uV, scaling to Volts...")
        raw_data *= 1e-6 # Convert to Volts
        
    # --- CRITICAL PREPROCESSING (Match Training) ---
    print("   Applying Preprocessing (Notch 50Hz + Bandpass 1-40Hz)...")
    info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.filter(1, 40, verbose=False)
    raw_data = raw.get_data()
    # -----------------------------------------------

    # Extract Markers
    try:
        markers = df.iloc[:, 23].values
    except:
        markers = np.zeros(raw_data.shape[1])
        print("⚠️ Warning: No Marker column found (Index 23)")
    
    return raw_data, markers, os.path.basename(filename)

def predict_cascade(clf_rest, clf_motor, sos_mu, eeg_full, current_idx):
    """Simulate exactly one step of the real-time pipeline."""
    
    # Needs at least 2.5s of history to be safe (motor window)
    if current_idx < WINDOW_MOTOR_SAMPLES:
        return "Buffering...", 0.0, [0.0, 0.0, 0.0], "Wait"

    # --- STAGE 1: REST CHECK (Last 1.0s) ---
    start_rest = current_idx - WINDOW_REST_SAMPLES
    window_rest = eeg_full[:, start_rest:current_idx]
    
    # Filter 8-13Hz (Realtime style)
    w_rest_filt = scipy.signal.sosfilt(sos_mu, window_rest, axis=1)
    feat_rest = w_rest_filt[np.newaxis, :, :] # (1, 8, 250)
    
    # Predict (0=Active, 1=Rest)
    probs_r = clf_rest.predict_proba(feat_rest)[0]
    prob_rest = probs_r[1]
    
    state_rest = "REPOS" if prob_rest > REST_THRESHOLD else "ACTIF"
    
    # --- STAGE 2: MOTOR CHECK (Last 2.0s) ---
    probs_motor = [0.0, 0.0, 0.0]
    pred_motor_str = "..."
    
    if state_rest == "ACTIF":
        start_motor = current_idx - WINDOW_MOTOR_SAMPLES
        window_motor = eeg_full[:, start_motor:current_idx]
        
        # Filter
        w_motor_filt = scipy.signal.sosfilt(sos_mu, window_motor, axis=1)
        feat_motor = w_motor_filt[np.newaxis, :, :]
        
        # Predict (0=Left, 1=Right, 2=Feet)
        probs_m = clf_motor.predict_proba(feat_motor)[0]
        probs_motor = probs_m
        
        idx = np.argmax(probs_m)
        if idx == 0: pred_motor_str = "GAUCHE"
        elif idx == 1: pred_motor_str = "DROITE"
        else: pred_motor_str = "PIEDS"
        
    return state_rest, prob_rest, probs_motor, pred_motor_str

def get_ground_truth(markers, current_idx):
    # Look back 1 sample to see active marker
    if current_idx < len(markers):
        val = int(markers[current_idx])
        if val != 0: return val
    
    # Search distinct marker in last 2 seconds
    lookback = 500
    segment = markers[max(0, current_idx-lookback):current_idx+1]
    uniques = np.unique(segment)
    uniques = uniques[uniques != 0]
    if len(uniques) > 0:
        return int(uniques[-1]) # Return last seen marker
    return 0

def draw_signal(screen, data, x, y, w, h, scale=1.0, color=(255,255,255)):
    if len(data) < 2: return
    mid = y + h//2
    pts = []
    step = max(1, len(data)//w)
    for i in range(0, len(data), step):
        px = x + (i/len(data))*w
        py = mid - (data[i] * scale)
        py = max(y, min(y+h, py))
        pts.append((px, py))
    if len(pts)>1: pygame.draw.lines(screen, color, False, pts, 1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BCI REPLAY - CASCADE SIMULATOR")
    clock = pygame.time.Clock()
    
    # Fonts
    font_xl = pygame.font.SysFont("Arial", 48, bold=True)
    font_l = pygame.font.SysFont("Arial", 32)
    font_m = pygame.font.SysFont("Arial", 20)
    
    # Load Data & Models
    try:
        eeg_data, markers, fname = load_data()
        print("Loading models...")
        clf_rest = joblib.load(MODEL_REST_PATH)
        clf_motor = joblib.load(MODEL_MOTOR_PATH)
        print("Models loaded.")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    # Filter for Visualization (Pre-compute)
    print("Filtering for visualization...")
    sos_vis = scipy.signal.iirfilter(2, [1, 40], btype='bandpass', fs=SFREQ, output='sos')
    eeg_vis = scipy.signal.sosfilt(sos_vis, eeg_data, axis=1)
    
    # Filter for Inference (Mu Band)
    sos_mu = scipy.signal.iirfilter(4, [8, 13], btype='bandpass', fs=SFREQ, output='sos')

    # Playback State
    n_samples = eeg_data.shape[1]
    cursor = 0
    playing = False
    speed = 4 # Fast forward factor
    
    scale_uV = 0.5
    
    running = True
    while running:
        screen.fill(BG_COLOR)
        
        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: playing = not playing
                if e.key == pygame.K_RIGHT: cursor = min(n_samples-1, cursor + SFREQ)
                if e.key == pygame.K_LEFT: cursor = max(0, cursor - SFREQ)
                if e.key == pygame.K_UP: scale_uV *= 1.2
                if e.key == pygame.K_DOWN: scale_uV /= 1.2
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if my > HEIGHT - 50:
                    ratio = mx / WIDTH
                    cursor = int(ratio * n_samples)

        # Update Cursor
        if playing:
            cursor += int(speed * (clock.get_time() / 1000.0) * SFREQ) 
            if cursor >= n_samples:
                cursor = 0
                playing = False

        # --- PROCESS PIPELINE ---
        state_rest, prob_rest, probs_motor, pred_motor = predict_cascade(
            clf_rest, clf_motor, sos_mu, eeg_data, cursor
        )
        
        truth_val = get_ground_truth(markers, cursor)
        truth_str = LABELS_MAP.get(truth_val, "?")
        truth_col = COLOR_MAP.get(truth_val, (255, 255, 255))
        
        # --- UI DRAWING ---
        
        # TIME BAR
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WIDTH, 60))
        t_sec = cursor / SFREQ
        dur_sec = n_samples / SFREQ
        title = font_l.render(f"REPLAY: {fname} | ⏱️ {t_sec:.1f}s / {dur_sec:.1f}s", True, TEXT_COLOR)
        screen.blit(title, (20, 15))
        
        # MAIN LAYOUT
        
        # SIGNALS
        vis_window = 4 * SFREQ 
        start_vis = max(0, cursor - vis_window)
        end_vis = cursor
        
        for i in range(8):
            y_pos = 80 + i * 70
            trace = eeg_vis[i, start_vis:end_vis]
            ch = CH_NAMES[i]
            draw_signal(screen, trace, 50, y_pos, 800, 60, scale=1.0/(scale_uV*1e-6), color=ACCENT_COLOR)
            screen.blit(font_m.render(ch, True, (100,100,100)), (10, y_pos+20))
            
        # Vertical Line at "Now"
        pygame.draw.line(screen, (255, 255, 255), (850, 70), (850, 650), 2)
        screen.blit(font_m.render("NOW", True, (255,255,255)), (835, 660))

        # DASHBOARD (Right Side)
        x_dash = 900
        w_dash = 450
        
        # BLOCK 1: CASCADE STATE
        y_d = 80
        pygame.draw.rect(screen, PANEL_COLOR, (x_dash, y_d, w_dash, 300), border_radius=10)
        
        screen.blit(font_l.render("🧠 AI PREDICTION", True, ACCENT_COLOR), (x_dash+20, y_d+20))
        
        # Stage 1
        screen.blit(font_m.render("Stage 1: Rest/Active", True, TEXT_COLOR), (x_dash+20, y_d+70))
        r_col = REST_COLOR if state_rest == "REPOS" else ACTIVE_COLOR
        pygame.draw.rect(screen, r_col, (x_dash+20, y_d+100, int(prob_rest*300), 20))
        pygame.draw.rect(screen, (50,50,50), (x_dash+20+int(prob_rest*300), y_d+100, 300-int(prob_rest*300), 20), 1)
        screen.blit(font_l.render(state_rest, True, r_col), (x_dash+340, y_d+90))
        
        # Stage 2
        y_s2 = y_d + 150
        screen.blit(font_m.render("Stage 2: Motor Class", True, TEXT_COLOR), (x_dash+20, y_s2))
        
        if state_rest == "ACTIF":
            screen.blit(font_xl.render(pred_motor, True, ACCENT_COLOR), (x_dash+20, y_s2+40))
            
            # Bars
            labs = ["G", "D", "P"]
            for k in range(3):
                p = probs_motor[k]
                py = y_s2 + 40 + k*25
                pygame.draw.rect(screen, (40,40,40), (x_dash+250, py, 100, 20))
                pygame.draw.rect(screen, ACCENT_COLOR, (x_dash+250, py, int(p*100), 20))
                screen.blit(font_m.render(labs[k], True, TEXT_COLOR), (x_dash+220, py))
        else:
             screen.blit(font_xl.render("---", True, (80,80,80)), (x_dash+20, y_s2+40))

        # BLOCK 2: GROUND TRUTH
        y_gt = 400
        pygame.draw.rect(screen, PANEL_COLOR, (x_dash, y_gt, w_dash, 200), border_radius=10)
        
        screen.blit(font_l.render("📋 GROUND TRUTH", True, ACCENT_COLOR), (x_dash+20, y_gt+20))
        screen.blit(font_m.render("Marqueur Réel (CSV)", True, TEXT_COLOR), (x_dash+20, y_gt+60))
        
        screen.blit(font_xl.render(truth_str, True, truth_col), (x_dash+20, y_gt+100))
        screen.blit(font_l.render(f"(Code: {truth_val})", True, (100,100,100)), (x_dash+250, y_gt+110))

        # TIMELINE BOTTOM
        pygame.draw.rect(screen, (30,30,40), (0, HEIGHT-50, WIDTH, 50))
        progress = cursor / n_samples
        pygame.draw.rect(screen, ACCENT_COLOR, (0, HEIGHT-50, int(progress*WIDTH), 50))
        
        screen.blit(font_m.render("[SPACE] Play/Pause  |  [ARROWS] Seek  |  [UP/DOWN] Scale", True, (200,200,200)), (20, HEIGHT-35))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
