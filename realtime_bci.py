import time
import numpy as np
import pygame
import joblib
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter

# ================= CONFIGURATION =================
COM_PORT = "COM10"
BOARD_ID = BoardIds.CYTON_BOARD.value

MODEL_PATH = "models/riemann_model.pkl"
WINDOW_SECONDS = 3.0
UPDATE_INTERVAL = 0.2
SFREQ = 250
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']

# UI Configuration
WIDTH, HEIGHT = 1400, 700
BG_COLOR = (15, 15, 25)
PANEL_COLOR = (25, 25, 40)
TEXT_COLOR = (220, 220, 230)
ACCENT_COLOR = (0, 180, 255)

# Signal plot configuration
SIGNAL_HISTORY_SECONDS = 3
SIGNAL_BUFFER_SIZE = int(SFREQ * SIGNAL_HISTORY_SECONDS)

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

    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    try:
        clf = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        board.stop_stream()
        board.release_session()
        return

    # 3. INIT PYGAME
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BCI - Real-Time Brain Signal Monitor")
    
    font_large = pygame.font.Font(None, 80)
    font_medium = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)
    font_tiny = pygame.font.Font(None, 20)
    
    clock = pygame.time.Clock()
    last_prediction_time = time.time()
    current_prediction = "Accumulating data..."
    current_probs = [0.0, 0.0, 0.0, 0.0]
    
    # Signal buffers (one per channel)
    signal_buffers = [deque(maxlen=SIGNAL_BUFFER_SIZE) for _ in range(8)]
    
    # Wait for initial buffer to fill
    print("Waiting for buffer to fill (3 seconds)...")
    time.sleep(3.5)  # Wait slightly more than WINDOW_SECONDS
    print("Ready!")
    
    running = True
    while running:
        screen.fill(BG_COLOR)
        
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- DATA ACQUISITION ---
        now = time.time()
        
        # Get data for signal display (last 50 samples)
        # Use get_current_board_data to NOT remove from buffer
        display_data = board.get_current_board_data(50)
        if display_data.shape[1] > 0:
            for ch_idx in range(8):
                for sample in display_data[ch_idx + 1]:
                    signal_buffers[ch_idx].append(sample)
        
        # --- PREDICTION LOOP ---
        if now - last_prediction_time > UPDATE_INTERVAL:
            n_samples = int(WINDOW_SECONDS * SFREQ)
            data = board.get_current_board_data(n_samples)
            
            print(f"DEBUG: Requested {n_samples} samples, got {data.shape[1]}")
            
            if data.shape[1] >= n_samples:
                # Filter and scale
                for channel in range(8):
                    DataFilter.perform_bandpass(data[channel+1], SFREQ, 8.0, 30.0, 4, 0, 0)
                
                eeg_data_filtered = data[1:9, :] / 1e6
                epoch = eeg_data_filtered[np.newaxis, :, :]
                
                print(f"DEBUG: Epoch shape: {epoch.shape}")
                
                try:
                    probs = clf.predict_proba(epoch)[0]
                    current_probs = probs
                    
                    print(f"DEBUG: Probabilities: {probs}")
                    
                    pred_idx = np.argmax(probs)
                    classes = clf.classes_
                    pred_label = classes[pred_idx]
                    
                    mapping = {1: "LEFT", 2: "RIGHT", 3: "FEET", 10: "REST"}
                    current_prediction = mapping.get(pred_label, "UNKNOWN")
                    
                    print(f"DEBUG: Prediction: {current_prediction}")
                except Exception as e:
                    print(f"❌ Prediction Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"DEBUG: Not enough data yet ({data.shape[1]}/{n_samples})")
                
            last_prediction_time = now

        # ==================== DRAW UI ====================
        
        # --- LEFT PANEL: Signal Visualization ---
        panel_width = 900
        panel_x = 20
        panel_y = 20
        
        # Title
        title = font_medium.render("Live EEG Signals", True, ACCENT_COLOR)
        screen.blit(title, (panel_x, panel_y))
        
        # Draw 8 mini signal plots (2 rows x 4 columns)
        plot_start_y = panel_y + 60
        plot_width = 200
        plot_height = 100
        plot_gap = 20
        
        for idx in range(8):
            row = idx // 4
            col = idx % 4
            
            plot_x = panel_x + col * (plot_width + plot_gap)
            plot_y = plot_start_y + row * (plot_height + plot_gap + 30)
            
            # Background
            pygame.draw.rect(screen, PANEL_COLOR, (plot_x, plot_y, plot_width, plot_height), border_radius=5)
            
            # Channel label
            label = font_small.render(CH_NAMES[idx], True, TEXT_COLOR)
            screen.blit(label, (plot_x + 5, plot_y - 25))
            
            # Draw signal
            if len(signal_buffers[idx]) > 10:
                buffer = list(signal_buffers[idx])
                
                # Normalize for display
                buffer_array = np.array(buffer[-200:])  # Last 200 samples
                if buffer_array.std() > 0:
                    normalized = (buffer_array - buffer_array.mean()) / (buffer_array.std() * 3)
                    normalized = np.clip(normalized, -1, 1)
                    
                    # Convert to screen coordinates
                    mid_y = plot_y + plot_height // 2
                    points = []
                    for i, val in enumerate(normalized):
                        x = plot_x + int((i / len(normalized)) * plot_width)
                        y = int(mid_y - val * (plot_height // 2 - 5))
                        points.append((x, y))
                    
                    if len(points) > 1:
                        pygame.draw.lines(screen, ACCENT_COLOR, False, points, 2)
        
        # --- RIGHT PANEL: Prediction Display ---
        right_panel_x = panel_width + 60
        right_panel_y = 20
        right_panel_width = WIDTH - right_panel_x - 20
        
        # Prediction Title
        pred_title = font_medium.render("Prediction", True, ACCENT_COLOR)
        screen.blit(pred_title, (right_panel_x, right_panel_y))
        
        # Main prediction display
        pred_y = right_panel_y + 80
        
        # Color based on prediction
        colors = {"LEFT": (255, 80, 80), "RIGHT": (80, 255, 80), 
                 "FEET": (80, 80, 255), "REST": (200, 200, 200)}
        pred_color = colors.get(current_prediction, TEXT_COLOR)
        
        pred_text = font_large.render(current_prediction, True, pred_color)
        pred_rect = pred_text.get_rect(center=(right_panel_x + right_panel_width//2, pred_y))
        screen.blit(pred_text, pred_rect)
        
        # Circular indicator
        indicator_y = pred_y + 100
        pygame.draw.circle(screen, pred_color, 
                          (right_panel_x + right_panel_width//2, indicator_y), 40)
        
        # Probability bars
        bar_start_y = indicator_y + 100
        labels = ["Left", "Right", "Feet", "Rest"]
        bar_colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (200, 200, 200)]
        
        bar_width = 80
        bar_max_height = 200
        bar_gap = 15
        total_width = len(labels) * bar_width + (len(labels) - 1) * bar_gap
        bar_start_x = right_panel_x + (right_panel_width - total_width) // 2
        
        for i, (prob, label, color) in enumerate(zip(current_probs, labels, bar_colors)):
            x = bar_start_x + i * (bar_width + bar_gap)
            
            # Background
            pygame.draw.rect(screen, (40, 40, 50), 
                           (x, bar_start_y, bar_width, bar_max_height), border_radius=5)
            
            # Active bar
            h = int(prob * bar_max_height)
            if h > 0:
                pygame.draw.rect(screen, color, 
                               (x, bar_start_y + bar_max_height - h, bar_width, h), border_radius=5)
            
            # Percentage text
            pct_text = font_small.render(f"{prob*100:.0f}%", True, TEXT_COLOR)
            pct_rect = pct_text.get_rect(center=(x + bar_width//2, bar_start_y + bar_max_height + 15))
            screen.blit(pct_text, pct_rect)
            
            # Label
            lbl_text = font_small.render(label, True, TEXT_COLOR)
            lbl_rect = lbl_text.get_rect(center=(x + bar_width//2, bar_start_y + bar_max_height + 40))
            screen.blit(lbl_text, lbl_rect)
        
        # Bottom info bar
        info_y = HEIGHT - 40
        pygame.draw.rect(screen, PANEL_COLOR, (0, info_y, WIDTH, 40))
        
        info_text = font_tiny.render(
            f"Buffer: {WINDOW_SECONDS}s | Update: {UPDATE_INTERVAL}s | FPS: {int(clock.get_fps())} | Press ESC to quit",
            True, (150, 150, 150)
        )
        screen.blit(info_text, (20, info_y + 12))
        
        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    print("Stopping Stream...")
    board.stop_stream()
    board.release_session()
    pygame.quit()

if __name__ == "__main__":
    main()
