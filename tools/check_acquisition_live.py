import time
import sys
import os
import numpy as np
import scipy.signal
import pygame
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
    # 1. INIT BRAINFLOW
    params = BrainFlowInputParams()
    # Try to use the same port as user likely has
    params.serial_port = COM_PORT
    
    print(f"Connecting to {COM_PORT}...")
    try:
        board = BoardShim(BOARD_ID, params)
        board.prepare_session()
        board.start_stream()
        print("Connected.")
    except BrainFlowError as e:
        print(f"Connection Failed: {e}")
        print("Make sure Cyton is ON and Dongle is connected.")
        # Optional: Start synthetic board for testing if real fails? 
        # But this is 'check_acquisition', so failure is informative.
        return

    # 2. INIT PYGAME
    pygame.init()
    # Increased resolution for signals (Reduced per user request)
    WIDTH, HEIGHT = 1100, 750 
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
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

        # --- DATA PROCESSING ---
        # Get all data in buffer
        # get_board_data clears the buffer, so we use it to drain the ringbuffer 
        # but we need to keep a buffer in memory if we want to smooth things out.
        # However, for live plotting, we just want the latest chunk.
        # But wait, metrics need ~1 sec.
        
        # BrainFlow's get_current_board_data(N) returns the last N samples 
        # WITHOUT clearing the buffer. This is perfect.
        
        n_samples = int(BUFFER_SIZE_SEC * SFREQ)
        data = board.get_current_board_data(n_samples)
        eeg_channels = board.get_eeg_channels(BOARD_ID)
        
        if data.shape[1] < SFREQ: 
            # Not enough data yet
            loading_txt = font_large.render("Buffering...", True, (100, 100, 100))
            screen.blit(loading_txt, (WIDTH//2 - 50, HEIGHT//2))
            pygame.display.flip()
            time.sleep(0.05)
            continue
            
        # Display Data (Filtered copy)
        disp_data = np.copy(data)
        if show_filtered:
            # We filter the whole chunk we retrieved. 
            # Note: filtering short chunks at edges might have artifacts. 
            # Ideally we maintain a continuous buffer, but for viz this is usually 'ok'.
            for ch in eeg_channels:
                # Apply 50Hz Notch (Environmental Noise Removal)
                DataFilter.remove_environmental_noise(disp_data[ch], SFREQ, NoiseTypes.FIFTY.value)
                # Apply Bandpass 1-50Hz
                DataFilter.perform_bandpass(disp_data[ch], SFREQ, 1.0, 50.0, 4, 
                                          FilterTypes.BUTTERWORTH.value, 0)
        else:
             for ch in eeg_channels:
                 # Just remove DC offset for viewing raw (Manual Detrend)
                 disp_data[ch] = disp_data[ch] - np.mean(disp_data[ch])

        # --- HEADER ---
        header_h = 60
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WIDTH, header_h))
        
        title = font_header.render(f"Signal Quality Check ({'Filtered' if show_filtered else 'Raw'})", True, ACCENT_COLOR)
        screen.blit(title, (20, 15))
        
        info_txt = font_medium.render(f"Scale: {scale_uV:.2f} px/uV | Press 'F' to filter | Up/Down to scale", True, (150, 150, 150))
        screen.blit(info_txt, (WIDTH - 400, 20))
        
        # --- CHANNELS ---
        start_y = header_h + 10
        row_height = (HEIGHT - start_y - 20) // 8
        row_height = min(100, row_height)
        
        # Column X positions
        col_name = 20
        col_stat = 80
        col_ptp = 180
        col_50hz = 280
        col_plot = 400
        plot_w = WIDTH - col_plot - 20
        
        for i, ch_idx in enumerate(eeg_channels):
            if i >= 8: break
            
            y = start_y + i * row_height
            ch_data_raw = data[ch_idx] # For metrics use RAW data
            ch_data_disp = disp_data[ch_idx] # For plot use Processed data
            
            # --- METRICS (Computed on RAW usually, or raw-ish) ---
            # PTP
            ptp = np.ptp(ch_data_raw)
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
            
            # PTP
            ptp_s = font_medium.render(f"PTP: {int(ptp)}", True, (180, 180, 180))
            screen.blit(ptp_s, (col_ptp, y + row_height//2 - 8))
            
            # 50Hz
            noise_s = font_medium.render(f"50Hz: {noise_str}", True, noise_col)
            screen.blit(noise_s, (col_50hz, y + row_height//2 - 8))
            
            # --- PLOT ---
            # Plot height slightly smaller than row
            plot_h = row_height - 10
            draw_signal(screen, ch_data_disp, col_plot, y + 5, plot_w, plot_h, scale=scale_uV, color=ACCENT_COLOR)
            
        pygame.display.flip()
        clock.tick(30) # 30 FPS is enough for viz

    try:
        board.stop_stream()
        board.release_session()
    except:
        pass
    pygame.quit()

if __name__ == "__main__":
    main()
