import time
import sys
import os
import numpy as np
import scipy.signal
import pygame
import cv2  # Webcam support
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes

# Configuration
COM_PORT = "COM10"
BOARD_ID = BoardIds.CYTON_BOARD.value
SFREQ = 250
BUFFER_SIZE_SEC = 5  # Keep 5 seconds for analysis/display
UPDATE_INTERVAL = 0.05 

# Channel Names based on User's Montage
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
# Indices for Alpha Check
# FC3=0, FC4=1, CP3=2, Cz=3, C3=4, C4=5, Pz=6, CP4=7
ALPHA_CHANNELS_IDX = [2, 6, 7] 

# GUI Colors
BG_COLOR = (15, 15, 20)
PANEL_COLOR = (30, 30, 45)
TEXT_COLOR = (220, 220, 230)
ACCENT_COLOR = (0, 180, 255)
SUCCESS_COLOR = (50, 200, 80)
WARNING_COLOR = (255, 180, 50)
ERROR_COLOR = (255, 80, 80)
GRID_COLOR = (40, 40, 60)

def compute_alpha_power(data_chunk, sfreq):
    """Computes band power in 8-12 Hz range."""
    nperseg = min(len(data_chunk), int(sfreq * 1.0))
    if nperseg < sfreq // 2: return 0.0
    
    freqs, psd = scipy.signal.welch(data_chunk, fs=sfreq, nperseg=nperseg)
    idx_alpha = np.logical_and(freqs >= 8, freqs <= 12)
    power = np.mean(psd[idx_alpha])
    return power

def check_50hz(data_chunk, sfreq):
    """Returns ratio of 50Hz power to neighbors."""
    nperseg = min(len(data_chunk), int(sfreq * 1.0))
    if nperseg < sfreq: return 0.0 # Need enough data
    
    freqs, psd = scipy.signal.welch(data_chunk, fs=sfreq, nperseg=nperseg)
    
    idx_50 = np.logical_and(freqs >= 48, freqs <= 52)
    idx_bg = np.logical_or(
        np.logical_and(freqs >= 40, freqs <= 45),
        np.logical_and(freqs >= 55, freqs <= 60)
    )
    
    power_50 = np.mean(psd[idx_50]) if np.any(idx_50) else 0
    power_bg = np.mean(psd[idx_bg]) if np.any(idx_bg) else 1e-9
    
    return power_50 / power_bg

def draw_signal(screen, signal_data, x, y, w, h, scale=1.0, color=(255, 255, 255)):
    """Draws a line plot of the signal."""
    if len(signal_data) < 2: return
    
    # Background for plot
    pygame.draw.rect(screen, (20, 20, 25), (x, y, w, h))
    pygame.draw.rect(screen, (40, 40, 50), (x, y, w, h), 1)
    
    # Center line
    mid_y = y + h // 2
    pygame.draw.line(screen, (35, 35, 45), (x, mid_y), (x + w, mid_y))
    
    # Normalize/Scale points
    # We want ~50uV to be meaningful height. 
    # Scale: pixels per uV. 
    # If scale=1.0, 1uV = 1px.
    
    points = []
    n_points = len(signal_data)
    step = max(1, n_points // w) # Simple downsampling for display width
    
    for i in range(0, n_points, step):
        val = signal_data[i]
        
        # Screen Coordinates
        px = x + (i / n_points) * w
        py = mid_y - (val * scale)
        
        # Clip
        py = max(y, min(y + h, py))
        points.append((px, py))
        
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 1)

def main():
    # 1. BRAINFLOW SETUP
    params = BrainFlowInputParams()
    params.serial_port = COM_PORT
    
    board = None
    board_connected = False
    
    def try_connect():
        nonlocal board, board_connected 
        print(f"Connecting to {COM_PORT}...")
        try:
            if board is not None:
                try: 
                    board.stop_stream()
                    board.release_session()
                except: pass
            
            board = BoardShim(BOARD_ID, params)
            board.prepare_session()
            board.start_stream()
            board_connected = True
            print("Connected.")
            return True
        except BrainFlowError as e:
            print(f"Connection Failed: {e}")
            board_connected = False
            return False

    try_connect() # Attempt initial connection

    # 2. INIT PYGAME
    pygame.init()
    # Optimized resolution (Max requested 1500x750)
    WIDTH, HEIGHT = 1450, 750 
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("BCI Signal Verification - Live Dashboard")
    
    font_header = pygame.font.SysFont("Arial", 24, bold=True)
    font_large = pygame.font.SysFont("Arial", 20)
    font_medium = pygame.font.SysFont("Arial", 16)
    font_small = pygame.font.SysFont("Arial", 14)
    
    clock = pygame.time.Clock()
    running = True
    
    # Scale factor for plots (pixels per uV)
    # EEG is usually +/- 50-100 uV. Height is ~80px.
    # So 40px = 100uV => 0.4 px/uV
    scale_uV = 0.5 
    show_filtered = True
    cam_rotation = 0 # 0, 90, 180, 270
    cam_mirror = True # Mirror by default (Self-view standard)
    paused = False # Freeze display
    
    # 3. INIT WEBCAM
    
    # 3. INIT WEBCAM
    print("Opening Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam.")
        cam_enabled = False
    else:
        cam_enabled = True
        print("Webcam Active.")
    
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_f:
                    show_filtered = not show_filtered
                if event.key == pygame.K_UP:
                    scale_uV *= 1.2
                if event.key == pygame.K_DOWN:
                    scale_uV /= 1.2
                # Rotation Controls
                if event.key == pygame.K_LEFT:
                    cam_rotation = (cam_rotation - 90) % 360
                if event.key == pygame.K_RIGHT:
                    cam_rotation = (cam_rotation + 90) % 360
                if event.key == pygame.K_m:
                    cam_mirror = not cam_mirror
                # Pause/Unpause
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r: # Retry connection
                    if not board_connected:
                        try_connect()
                    
            if event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

        if not paused and board_connected:
            # --- DATA PROCESSING ---
            n_samples = int(BUFFER_SIZE_SEC * SFREQ)
            try:
                data = board.get_current_board_data(n_samples)
            except:
                data = None
                
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            
            if data is None or data.shape[1] < SFREQ: 
                # Not enough data yet
                loading_txt = font_large.render("Buffering..." if board_connected else "No Data", True, (100, 100, 100))
                # Just continue to render GUI with zeros or wait?
                # Better to initialize display data with zeros
                pass 
                
            # Display Data (Filtered copy)
            if data is not None and data.shape[1] > 0:
                disp_data = np.copy(data)
                if show_filtered:
                    for ch in eeg_channels:
                        DataFilter.remove_environmental_noise(disp_data[ch], SFREQ, NoiseTypes.FIFTY.value)
                        DataFilter.perform_bandpass(disp_data[ch], SFREQ, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
                else:
                     for ch in eeg_channels:
                         disp_data[ch] = disp_data[ch] - np.mean(disp_data[ch])
            else:
                # Fallback if connected but buffering
                disp_data = np.zeros((32, 1)) # Dummy
                data = np.zeros((32, 1)) # Dummy

        elif not board_connected:
            # Disconnected State - Zero Data
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            n_samples = int(BUFFER_SIZE_SEC * SFREQ)
            data = np.zeros((32, n_samples))
            disp_data = np.zeros((32, n_samples)) + np.random.normal(0, 5, (32, n_samples)) # Little static to show it's alive?

        # --- HEADER ---
        header_h = 60
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WIDTH, header_h))
        
        title = font_header.render(f"Signal Quality ({'Filtered' if show_filtered else 'Raw'})", True, ACCENT_COLOR)
        screen.blit(title, (50, 15)) # Shifted for icon
        
        # Connection Status Icon
        conn_col = SUCCESS_COLOR if board_connected else ERROR_COLOR
        pygame.draw.circle(screen, conn_col, (25, 25), 10)
        
        if not board_connected:
            err_txt = font_medium.render("DISCONNECTED (Press 'R' to Retry)", True, ERROR_COLOR)
            screen.blit(err_txt, (50, 40))
        
        info_txt = font_medium.render(f"Scale: {scale_uV:.2f} px/uV | 'F' Filt | 'M' Mir | SPC Pause | 'R' Retry", True, (150, 150, 150))
        screen.blit(info_txt, (WIDTH - 500, 20))
        
        if paused:
            pause_surf = font_header.render("PAUSED", True, WARNING_COLOR)
            screen.blit(pause_surf, (WIDTH//2 - 50, 20))
        
        # --- CHANNELS ---
        start_y = header_h + 10
        row_height = (HEIGHT - start_y - 20) // 8
        row_height = min(100, row_height)
        
        # Column X positions
        col_name = 20
        col_stat = 80
        col_offset = 180
        col_ptp = 280
        col_50hz = 380
        col_plot = 500 # Moved right to make space
        
        # Responsive Layout
        # Fixed Left Panel: 500px
        # Remaining: Signal Plot (60%) + Webcam (40%)
        avail_w = WIDTH - col_plot - 20
        plot_w = int(avail_w * 0.60)
        
        # Webcam Position
        cam_x = col_plot + plot_w + 10
        cam_y = start_y
        cam_w = WIDTH - cam_x - 10
        # Limit webcam height to avoid overlapping
        cam_h = min(cam_w * 3 // 4, HEIGHT - start_y - 20) 
        
        for i, ch_idx in enumerate(eeg_channels):
            if i >= 8: break
            
            y = start_y + i * row_height
            ch_data_raw = data[ch_idx] # For metrics use RAW data
            ch_data_disp = disp_data[ch_idx] # For plot use Processed data
            
            # --- METRICS (Computed on RAW usually, or raw-ish) ---
            if board_connected and data.shape[1] > 10:
                # PTP (Amplitude/Noise)
                ptp = np.ptp(ch_data_raw)
                # Offset (DC Connection Quality)
                offset_val = np.mean(ch_data_raw)
                # 50Hz (Computed on Raw to detect noise)
                ratio_50 = check_50hz(ch_data_raw, SFREQ)
                
                # Status
                is_railed = (np.mean(np.abs(ch_data_raw)) > 1) and ( (np.abs(ch_data_raw) > 185000).any() or (np.std(ch_data_raw) < 0.1) )
                
                status_text = "OK"
                status_color = SUCCESS_COLOR
                
                if is_railed:
                    status_text = "RAILED"
                    status_color = ERROR_COLOR
                elif ptp > 500: # uV
                    status_text = "UNSTABLE"
                    status_color = ERROR_COLOR
                elif ptp > 150:
                    status_text = "NOISY"
                    status_color = WARNING_COLOR
            else:
                # Dummy metrics when disconnected
                ptp = 0
                offset_val = 0
                ratio_50 = 0
                status_text = "N/C"
                status_color = (100, 100, 100)
                is_railed = False
                
            # Noise 50Hz Labels
            noise_str = f"{ratio_50:.1f}"
            noise_col = SUCCESS_COLOR
            if ratio_50 > 10: noise_col = WARNING_COLOR
            if ratio_50 > 100: noise_col = ERROR_COLOR
            
            # --- DRAW ROW INFO ---
            # Name
            name_s = font_large.render(f"{CH_NAMES[i]}", True, TEXT_COLOR)
            screen.blit(name_s, (col_name, y + row_height//2 - 10))
            
            # Status
            pygame.draw.circle(screen, status_color, (col_stat + 10, y + row_height//2), 6)
            stat_s = font_medium.render(status_text, True, status_color)
            screen.blit(stat_s, (col_stat + 25, y + row_height//2 - 8))
            
            # Offset (Resistance Proxy)
            # Closer to 0 is generally better (or stable). Railed is bad.
            off_col = (180, 180, 180)
            if abs(offset_val) > 150000: off_col = ERROR_COLOR # Near Rail
            
            off_s = font_medium.render(f"DC: {int(offset_val/1000)}k", True, off_col)
            screen.blit(off_s, (col_offset, y + row_height//2 - 8))

            # Amplitude
            ptp_s = font_medium.render(f"Amp: {int(ptp)}", True, (150, 150, 150))
            screen.blit(ptp_s, (col_ptp, y + row_height//2 - 8))
            
            # 50Hz
            noise_s = font_medium.render(f"50Hz: {noise_str}", True, noise_col)
            screen.blit(noise_s, (col_50hz, y + row_height//2 - 8))
            
            # --- PLOT ---
            # Plot height slightly smaller than row
            plot_h = row_height - 10
            draw_signal(screen, ch_data_disp, col_plot, y + 5, plot_w, plot_h, scale=scale_uV, color=ACCENT_COLOR)
            
        # --- DRAW WEBCAM ---
        if cam_enabled and not paused:
            ret, frame = cap.read()
            if ret:
                # 0. Mirror (Flip Horizontal) - BEFORE Rotation usually makes sense for "Self View"
                if cam_mirror:
                    frame = cv2.flip(frame, 1)

                # 1. Handle Rotation
                if cam_rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # +90
                elif cam_rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif cam_rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # -90 / +270
                
                # 2. Resize to fit slot (Preserve aspect ratio is tricky efficiently, 
                # strictly fitting to box is easier for GUI)
                # Note: after rotation, dimensions might have swapped.
                # frame.shape is (h, w, 3)
                
                frame = cv2.resize(frame, (cam_w, cam_h))
                
                # 3. Color & Surface
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.transpose(frame, (1, 0, 2))
                cam_surf = pygame.surfarray.make_surface(frame)
                
                screen.blit(cam_surf, (cam_x, cam_y))
                
                # Add Label
                status_str = f"LIVE ({cam_rotation}°"
                if cam_mirror: status_str += " | Mirrored"
                status_str += ")"
                cv_lbl = font_medium.render(status_str, True, ACCENT_COLOR)
                screen.blit(cv_lbl, (cam_x, cam_y - 25))
                
                # Save last frame for paused state
                last_frame_surf = cam_surf
                last_frame_lbl = cv_lbl
        
        elif cam_enabled and paused and 'last_frame_surf' in locals():
             # Draw cached frame
             screen.blit(last_frame_surf, (cam_x, cam_y))
             screen.blit(last_frame_lbl, (cam_x, cam_y - 25))
            
        pygame.display.flip()
        clock.tick(30) # 30 FPS is enough for viz

    if board_connected:
        try:
            board.stop_stream()
            board.release_session()
        except:
            pass
    if cam_enabled: cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
