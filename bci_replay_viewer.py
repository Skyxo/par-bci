import numpy as np
import pandas as pd
import mne
import pygame
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = "EEG_Session_2026-01-13_16-59.csv"
SFREQ = 250
WINDOW_SIZE = 3.0
STEP_SIZE = 0.5

# UI Config
WIDTH, HEIGHT = 1400, 750  # Reduced height
BG_COLOR = (12, 12, 20)
PANEL_COLOR = (20, 20, 35)
ACCENT_COLOR = (0, 180, 255)
TEXT_COLOR = (220, 220, 230)

def load_and_preprocess(filename):
    """Load, filter (Notch+Bandpass) and extract epochs"""
    print(f"Loading {filename}...")
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
    except:
        df = pd.read_csv(filename, sep=',', header=None)
        
    eeg = df.iloc[:, 1:9].values.T * 1e-6
    markers = df.iloc[:, 23].values
    
    info = mne.create_info(ch_names=['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4'], 
                           sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg, info, verbose=False)
    
    # Filter: Notch 50Hz (Mains) + 100Hz (Harmonic) then Bandpass 8-30Hz
    raw.notch_filter([50, 100], fir_design='firwin', verbose=False)
    raw.filter(8., 30., fir_design='firwin', verbose=False)
    
    return raw, markers

def train_on_file(raw, markers):
    """Train model on the loaded file (Self-Training for visualization)"""
    print("Training model on current file...")
    
    diff_markers = np.diff(markers, prepend=0)
    events_idx = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
    events_vals = markers[events_idx].astype(int)
    events = np.column_stack((events_idx, np.zeros_like(events_idx), events_vals))
    
    epochs = mne.Epochs(raw, events, {'G':1, 'D':2, 'P':3, 'Rest':10}, tmin=0.5, tmax=3.5,
                       proj=False, baseline=None, verbose=False)
    
    clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(), 
                       LogisticRegression(solver='lbfgs'))
    clf.fit(epochs.get_data(), epochs.events[:, -1])
    print("✅ Model trained on current session data")
    return clf

def get_ground_truth(markers, start_idx, window_samples):
    """Determine ground truth for current window"""
    lookback_start = max(0, start_idx - int(1.0 * SFREQ))
    window_markers = markers[lookback_start : start_idx + window_samples]
    valid = window_markers[np.isin(window_markers, [1, 2, 3])]
    
    if len(valid) > 0:
        return int(valid[-1])
    return 0

def predict_window(clf, eeg_data, start_idx, window_samples):
    """Make prediction for current window"""
    window = eeg_data[:, start_idx:start_idx + window_samples]
    X = window[np.newaxis, :, :]
    probs = clf.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred = clf.classes_[pred_idx]
    conf = probs[pred_idx]
    
    full_probs = np.zeros(4)
    for i, cls in enumerate(clf.classes_):
        if cls == 1:
            full_probs[0] = probs[i]
        elif cls == 2:
            full_probs[1] = probs[i]
        elif cls == 3:
            full_probs[2] = probs[i]
        elif cls == 10:
            full_probs[3] = probs[i]
    
    return pred, conf, full_probs

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BCI Signal Viewer - Multi-Channel Display")
    clock = pygame.time.Clock()
    
    font_xlarge = pygame.font.Font(None, 72)
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)
    font_tiny = pygame.font.Font(None, 20)
    
    raw, markers = load_and_preprocess(DATA_FILE)
    eeg_data = raw.get_data()
    clf = train_on_file(raw, markers)
    
    n_samples = eeg_data.shape[1]
    window_samples = int(WINDOW_SIZE * SFREQ)
    step_samples = int(STEP_SIZE * SFREQ)
    max_windows = (n_samples - window_samples) // step_samples
    
    # State
    current_window = 0
    playing = False
    dragging = False
    signal_dragging = False  # New: for dragging on signal area
    zoom_level = 1.0  # 1.0 = normal, 2.0 = zoomed 2x
    
    labels = {0: "REPOS", 1: "GAUCHE", 2: "DROITE", 3: "PIEDS", 10: "REPOS"}
    colors = {0: (150, 150, 150), 1: (255, 80, 80), 2: (80, 255, 80), 3: (80, 80, 255), 10: (150, 150, 150)}
    ch_name = 'Cz'  # Single channel: central motor cortex
    ch_idx = 2  # Index for Cz in original data
    
    running = True
    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_LEFT:
                    current_window = max(0, current_window - 1)
                elif event.key == pygame.K_RIGHT:
                    current_window = min(max_windows - 1, current_window + 1)
                elif event.key == pygame.K_HOME:
                    current_window = 0
                elif event.key == pygame.K_END:
                    current_window = max_windows - 1
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    zoom_level = min(5.0, zoom_level * 1.2)
                elif event.key == pygame.K_MINUS:
                    zoom_level = max(0.1, zoom_level / 1.2)
                elif event.key == pygame.K_PAGEUP:
                    current_window = max(0, current_window - 10)
                elif event.key == pygame.K_PAGEDOWN:
                    current_window = min(max_windows - 1, current_window + 10)
            elif event.type == pygame.MOUSEWHEEL:
                # Mouse wheel for navigation
                current_window -= event.y * 5  # 5 windows per wheel tick
                current_window = max(0, min(max_windows - 1, current_window))  # Min 0.1x = view 10x more data
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                # Zoom buttons
                zoom_plus_rect = pygame.Rect(WIDTH - 150, 10, 60, 35)
                zoom_minus_rect = pygame.Rect(WIDTH - 80, 10, 60, 35)
                
                # Navigation buttons
                nav_left_rect = pygame.Rect(WIDTH - 320, 10, 60, 35)
                nav_right_rect = pygame.Rect(WIDTH - 240, 10, 60, 35)
                
                if zoom_plus_rect.collidepoint(mx, my):
                    zoom_level = min(5.0, zoom_level * 1.2)
                elif zoom_minus_rect.collidepoint(mx, my):
                    zoom_level = max(0.1, zoom_level / 1.2)
                elif nav_left_rect.collidepoint(mx, my):
                    current_window = max(0, current_window - 10)
                elif nav_right_rect.collidepoint(mx, my):
                    current_window = min(max_windows - 1, current_window + 10)
                
                # Signal area drag (for navigation)
                signals_x = 390
                signals_y = 60
                signals_width = WIDTH - 2 * signals_x
                signals_height = HEIGHT - 120
                if signals_x <= mx <= signals_x + signals_width and signals_y <= my <= signals_y + signals_height:
                    signal_dragging = True
                
                # Timeline drag
                timeline_y = HEIGHT - 50
                if 50 <= mx <= WIDTH - 50 and timeline_y - 10 <= my <= timeline_y + 30:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                signal_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = pygame.mouse.get_pos()
                    ratio = (mx - 50) / (WIDTH - 100)
                    ratio = max(0, min(1, ratio))
                    current_window = int(ratio * max_windows)
                elif signal_dragging:
                    mx, my = pygame.mouse.get_pos()
                    signals_x = 390
                    signals_width = WIDTH - 2 * signals_x
                    # Map mouse X position to signal position
                    if signals_x <= mx <= signals_x + signals_width:
                        ratio = (mx - signals_x) / signals_width
                        current_window = int(ratio * max_windows)
                        current_window = max(0, min(max_windows - 1, current_window))
        
        if playing:
            current_window += 1
            if current_window >= max_windows:
                current_window = 0
        
        start_idx = current_window * step_samples
        current_time = start_idx / SFREQ
        
        pred, conf, probs = predict_window(clf, eeg_data, start_idx, window_samples)
        truth = get_ground_truth(markers, start_idx, window_samples)
        
        # --- TITLE BAR ---
        total_time = n_samples / SFREQ
        title = font_medium.render(f"📊 Visualiseur BCI  |  ⏱️ {current_time:.1f}s / {total_time:.0f}s  |  🔍 Zoom {zoom_level:.1f}x", True, TEXT_COLOR)
        screen.blit(title, (20, 10))
        
        # Zoom buttons
        zoom_plus_rect = pygame.Rect(WIDTH - 150, 10, 60, 35)
        zoom_minus_rect = pygame.Rect(WIDTH - 80, 10, 60, 35)
        
        # Navigation buttons
        nav_left_rect = pygame.Rect(WIDTH - 320, 10, 60, 35)
        nav_right_rect = pygame.Rect(WIDTH - 240, 10, 60, 35)
        
        # Draw navigation buttons
        pygame.draw.rect(screen, (80, 80, 200), nav_left_rect, border_radius=5)
        pygame.draw.rect(screen, (80, 80, 200), nav_right_rect, border_radius=5)
        
        pygame.draw.rect(screen, (60, 120, 200), zoom_plus_rect, border_radius=5)
        pygame.draw.rect(screen, (60, 120, 200), zoom_minus_rect, border_radius=5)
        
        # Button labels
        nav_left_text = font_medium.render("◄", True, (255, 255, 255))
        nav_right_text = font_medium.render("►", True, (255, 255, 255))
        plus_text = font_medium.render("+", True, (255, 255, 255))
        minus_text = font_medium.render("-", True, (255, 255, 255))
        
        nav_left_rect_center = nav_left_text.get_rect(center=nav_left_rect.center)
        nav_right_rect_center = nav_right_text.get_rect(center=nav_right_rect.center)
        plus_rect = plus_text.get_rect(center=zoom_plus_rect.center)
        minus_rect = minus_text.get_rect(center=zoom_minus_rect.center)
        
        screen.blit(nav_left_text, nav_left_rect_center)
        screen.blit(nav_right_text, nav_right_rect_center)
        screen.blit(plus_text, plus_rect)
        screen.blit(minus_text, minus_rect)
        
        # --- LEFT PANEL: AI Prediction ---
        pred_y = 60
        pred_width = 350
        pred_height = 340  # Reduced to give more space to signal
        pygame.draw.rect(screen, PANEL_COLOR, (20, pred_y, pred_width, pred_height), border_radius=10)
        
        # Title
        pred_label = font_medium.render("🤖 Ce que l'IA détecte", True, ACCENT_COLOR)
        screen.blit(pred_label, (40, pred_y + 10))
        
        # Subtitle
        subtitle = font_tiny.render("Intention détectée par le modèle", True, (150, 150, 150))
        screen.blit(subtitle, (40, pred_y + 45))
        
        # Main prediction
        pred_text = font_xlarge.render(labels[pred], True, colors[pred])
        pred_rect = pred_text.get_rect(center=(20 + pred_width//2, pred_y + 130))
        screen.blit(pred_text, pred_rect)
        
        # Confidence
        conf_text = font_medium.render(f"Confiance: {conf:.0%}", True, TEXT_COLOR)
        conf_rect = conf_text.get_rect(center=(20 + pred_width//2, pred_y + 200))
        screen.blit(conf_text, conf_rect)
        
        # Probabilities with labels
        prob_y = pred_y + 250
        prob_labels = ["G", "D", "P", "R"]  # Shortened labels
        prob_icons = ["👈", "👉", "🦶", "😌"]
        
        for i, (label_id, prob, label_txt, icon) in enumerate(zip([1, 2, 3, 0], probs, prob_labels, prob_icons)):
            y_pos = prob_y + i * 20  # Reduced spacing
            
            # Icon and label
            icon_text = font_small.render(f"{icon} {label_txt}", True, TEXT_COLOR)
            screen.blit(icon_text, (40, y_pos))
            
            # Progress bar
            bar_x = 100
            bar_width = 180
            bar_height = 15
            
            # Background
            pygame.draw.rect(screen, (40, 40, 50), (bar_x, y_pos, bar_width, bar_height), border_radius=3)
            
            # Filled portion
            fill_width = int(prob * bar_width)
            if fill_width > 0:
                pygame.draw.rect(screen, colors[label_id], (bar_x, y_pos, fill_width, bar_height), border_radius=3)
            
            # Percentage
            pct_text = font_tiny.render(f"{prob:.0%}", True, TEXT_COLOR)
            screen.blit(pct_text, (bar_x + bar_width + 5, y_pos))
        
        # --- RIGHT PANEL: Ground Truth + Legend ---
        truth_x = WIDTH - pred_width - 20
        pygame.draw.rect(screen, PANEL_COLOR, (truth_x, pred_y, pred_width, pred_height), border_radius=10)
        
        # Title
        truth_label = font_medium.render("📋 Intention réelle", True, ACCENT_COLOR)
        screen.blit(truth_label, (truth_x + 20, pred_y + 10))
        
        # Subtitle
        subtitle2 = font_tiny.render("Ce qui était demandé", True, (150, 150, 150))
        screen.blit(subtitle2, (truth_x + 20, pred_y + 45))
        
        # Ground truth
        truth_text = font_xlarge.render(labels[truth], True, colors[truth])
        truth_rect = truth_text.get_rect(center=(truth_x + pred_width//2, pred_y + 130))
        screen.blit(truth_text, truth_rect)
        
        # --- LEGEND for markers ---
        legend_y = pred_y + pred_height - 140  # Position at bottom of panel
        legend_title = font_small.render("🎨 Légende", True, ACCENT_COLOR)
        screen.blit(legend_title, (truth_x + 20, legend_y))
        
        marker_info = [
            ("🔴 Main gauche", (255, 80, 80)),
            ("🟢 Main droite", (80, 255, 80)),
            ("🔵 Pieds", (80, 80, 255)),
            ("⚪ Repos", (150, 150, 150))
        ]
        
        for i, (text, color) in enumerate(marker_info):
            y = legend_y + 35 + i * 35
            # Color square
            pygame.draw.rect(screen, color, (truth_x + 30, y, 20, 20), border_radius=3)
            # Label
            label = font_small.render(text, True, TEXT_COLOR)
            screen.blit(label, (truth_x + 60, y - 2))
        
        # --- SINGLE SIGNAL DISPLAY (Full height) ---
        signals_x = 390
        signals_y = pred_y + pred_height + 20  # Start below side panels
        signals_width = WIDTH - 2 * signals_x
        signals_height = HEIGHT - signals_y - 120  # To timeline
        
        pygame.draw.rect(screen, PANEL_COLOR, (signals_x, signals_y, signals_width, signals_height), border_radius=10)
        
        # Calculate view window based on zoom
        n_samples = eeg_data.shape[1]
        samples_to_show = int(window_samples / zoom_level)
        view_center = start_idx + window_samples // 2
        view_start = view_center - samples_to_show // 2
        view_end = view_start + samples_to_show
        
        # Ensure indices stay within valid bounds
        view_start = max(0, view_start)
        view_end = min(n_samples, view_end)
        
        # Adjust if we don't have enough samples
        if view_end - view_start < 10:
            view_start = max(0, start_idx - 50)
            view_end = min(n_samples, start_idx + window_samples + 50)
        
        # Single channel display
        plot_x = signals_x + 60
        plot_width = signals_width - 80
        plot_y = signals_y + 20
        plot_height = signals_height - 40
        plot_mid_y = plot_y + plot_height // 2
        
        # Background
        pygame.draw.rect(screen, (25, 25, 40), (plot_x, plot_y, plot_width, plot_height))
        
        # --- DRAW CORRECTNESS OVERLAY ZONES ---
        # Calculate visible windows and their correctness
        visible_start_window = max(0, view_start // step_samples)
        visible_end_window = min(max_windows, (view_end // step_samples) + 1)
        
        # Only draw overlays if plot area is valid
        if plot_height > 0 and plot_width > 0:
            for window_idx in range(visible_start_window, visible_end_window):
                window_start_sample = window_idx * step_samples
                window_end_sample = window_start_sample + window_samples
                
                # Get prediction and truth for this window
                if window_start_sample + window_samples <= n_samples:
                    window_pred, _, _ = predict_window(clf, eeg_data, window_start_sample, window_samples)
                    window_truth = get_ground_truth(markers, window_start_sample, window_samples)
                    
                    # Determine correctness
                    if window_truth != 0:
                        is_window_correct = (window_pred == window_truth)
                        
                        # Map to screen coordinates
                        overlay_start = max(view_start, window_start_sample)
                        overlay_end = min(view_end, window_end_sample)
                        
                        rel_start = (overlay_start - view_start) / max(1, view_end - view_start)
                        rel_end = (overlay_end - view_start) / max(1, view_end - view_start)
                        
                        x_start = plot_x + int(rel_start * plot_width)
                        x_end = plot_x + int(rel_end * plot_width)
                        width = max(1, x_end - x_start)
                        
                        # Draw colored zone only if dimensions are valid
                        if width > 0:
                            # Draw colored zone
                            if is_window_correct:
                                overlay_color = (0, 200, 0, 60)  # Green semi-transparent
                            else:
                                overlay_color = (200, 0, 0, 60)  # Red semi-transparent
                            
                            overlay_surface = pygame.Surface((width, plot_height), pygame.SRCALPHA)
                            overlay_surface.fill(overlay_color)
                            screen.blit(overlay_surface, (x_start, plot_y))
        
        # Channel label
        lbl = font_small.render(ch_name, True, ACCENT_COLOR)
        screen.blit(lbl, (signals_x + 10, plot_mid_y - 10))
        
        # Extract signal
        sig = eeg_data[ch_idx, view_start:view_end]
        if len(sig) > 10:
            norm_sig = (sig - sig.mean()) / (sig.std() + 1e-9)
            norm_sig = np.clip(norm_sig, -4, 4)
            
            points = []
            for j in range(len(norm_sig)):
                x = plot_x + int((j / len(norm_sig)) * plot_width)
                y = int(plot_mid_y - norm_sig[j] * (plot_height // 2.5))
                # Clip Y coordinate to stay within signal box
                y = np.clip(y, plot_y, plot_y + plot_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, ACCENT_COLOR, False, points, 2)
        
        # Draw markers - more visible
        marker_colors = {1: (255, 80, 80), 2: (80, 255, 80), 3: (80, 80, 255), 10: (150, 150, 150)}
        for idx in range(view_start, min(view_end, len(markers))):
            m = markers[idx]
            if m in [1, 2, 3, 10]:
                rel_pos = (idx - view_start) / max(1, view_end - view_start)
                x = plot_x + int(rel_pos * plot_width)
                color = marker_colors.get(m, (255, 255, 255))
                
                # Draw semi-transparent background for better visibility
                if plot_height > 0:
                    marker_bg = pygame.Surface((8, plot_height), pygame.SRCALPHA)
                    marker_bg.fill((*color, 80))  # Semi-transparent
                    screen.blit(marker_bg, (x - 4, plot_y))
                
                # Draw thick line
                pygame.draw.line(screen, color, (x, plot_y), (x, plot_y + max(1, plot_height)), 6)
        
        # --- TIMELINE ---
        timeline_y = HEIGHT - 50
        pygame.draw.rect(screen, PANEL_COLOR, (50, timeline_y - 10, WIDTH - 100, 50), border_radius=5)
        
        progress = current_window / max(1, max_windows)
        progress_width = WIDTH - 100
        pygame.draw.rect(screen, (50, 50, 60), (50, timeline_y, progress_width, 20))
        pygame.draw.rect(screen, ACCENT_COLOR, (50, timeline_y, int(progress * progress_width), 20))
        
        cursor_x = 50 + int(progress * progress_width)
        pygame.draw.circle(screen, (255, 255, 255), (cursor_x, timeline_y + 10), 8)
        
        # Controls - More visible and clear
        control_y = timeline_y + 25
        
        # Make controls more prominent with icons
        controls_parts = [
            ("📍 Drag signal", "Naviguer"),
            ("⏯️ ESPACE", "Play/Pause"),
            ("🔍 +/-", "Zoom"),
            ("⬅️➡️", "Avancer/Reculer"),
            ("🚪 ESC", "Quitter")
        ]
        
        x_offset = 50
        for label, desc in controls_parts:
            control_text = font_small.render(f"{label}: {desc}", True, (200, 200, 210))
            screen.blit(control_text, (x_offset, control_y))
            x_offset += control_text.get_width() + 25
        
        pygame.display.flip()
        clock.tick(30 if not playing else 10)
    
    pygame.quit()

if __name__ == "__main__":
    main()

