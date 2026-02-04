import pygame
import sys
import os
import subprocess
import time

# Configuration
WIDTH, HEIGHT = 1000, 600
BG_COLOR = (15, 15, 20)
BTN_COLOR = (40, 40, 60)
BTN_HOVER_COLOR = (60, 60, 90)
BTN_TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 180, 255)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_CHECK = os.path.join(BASE_DIR, "tools", "check_acquisition_live.py")
SCRIPT_ACQ = os.path.join(BASE_DIR, "acquisition_protocol.py")
SCRIPT_TRAIN = os.path.join(BASE_DIR, "training", "train_riemannian.py")
SCRIPT_LIVE = os.path.join(BASE_DIR, "realtime_bci.py")

class Button:
    def __init__(self, x, y, w, h, text, action_script, description, is_console=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.script = action_script
        self.desc = description
        self.is_console = is_console # If True, don't hide console output
        self.hovered = False

    def draw(self, screen, font, font_small):
        color = BTN_HOVER_COLOR if self.hovered else BTN_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT_COLOR, self.rect, 2, border_radius=10)
        
        # Text
        txt_surf = font.render(self.text, True, BTN_TEXT_COLOR)
        txt_rect = txt_surf.get_rect(center=(self.rect.centerx, self.rect.centery - 10))
        screen.blit(txt_surf, txt_rect)
        
        # Description
        desc_surf = font_small.render(self.desc, True, (150, 150, 150))
        desc_rect = desc_surf.get_rect(center=(self.rect.centerx, self.rect.centery + 20))
        screen.blit(desc_surf, desc_rect)

    def check_hover(self, mx, my):
        self.hovered = self.rect.collidepoint(mx, my)

    def click(self):
        if self.script:
            print(f"Launching {self.text}...")
            # We use subprocess to launch. 
            # On Windows, python w/o 'w' keeps console.
            cmd = [sys.executable, self.script]
            
            # We want to wait or run async? 
            # For BCI flow, usually blocking is fine or hiding the menu.
            # But let's keep menu open in background? 
            # Actually, because of BrainFlow board lock, we CANNOT have two scripts trying to access board.
            # But here, the STUDIO does not access the board. So it is safe to keep open.
            
            try:
                # Minimize Studio or just overlay?
                # Let's run blocking so user returns to studio after closing the tool
                pygame.display.iconify() 
                subprocess.run(cmd, cwd=BASE_DIR)
                # Restore
                pygame.display.init()
                screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Re-init sometimes needed
                pygame.display.set_caption("BCI Studio - Main Menu")
            except Exception as e:
                print(f"Error launching script: {e}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BCI Studio - Main Menu")
    clock = pygame.time.Clock()
    
    font_title = pygame.font.SysFont("Arial", 50, bold=True)
    font_btn = pygame.font.SysFont("Arial", 30)
    font_desc = pygame.font.SysFont("Arial", 18)
    
    # Layout
    btn_w, btn_h = 400, 100
    cx = WIDTH // 2 - btn_w // 2
    start_y = 150
    gap = 20
    
    buttons = [
        Button(cx, start_y, btn_w, btn_h, 
               "1. CHECK SIGNALS", SCRIPT_CHECK, 
               "Verify electrodes, impedance, and noise (Console)", is_console=True),
        
        Button(cx, start_y + (btn_h + gap), btn_w, btn_h, 
               "2. ACQUISITION", SCRIPT_ACQ, 
               "Run protocol to record training data"),
        
        Button(cx, start_y + (btn_h + gap)*2, btn_w, btn_h, 
               "3. TRAIN MODEL", SCRIPT_TRAIN, 
               "Train AI on latest recording (Console)"),
        
        Button(cx, start_y + (btn_h + gap)*3, btn_w, btn_h, 
               "4. PREDICT (LIVE)", SCRIPT_LIVE, 
               "Real-time classification from trained model")
    ]
    
    running = True
    while running:
        screen.fill(BG_COLOR)
        mx, my = pygame.mouse.get_pos()
        
        # Title
        title_surf = font_title.render("BCI WORKSTATION", True, ACCENT_COLOR)
        title_rect = title_surf.get_rect(center=(WIDTH//2, 80))
        screen.blit(title_surf, title_rect)
        
        for btn in buttons:
            btn.check_hover(mx, my)
            btn.draw(screen, font_btn, font_desc)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for btn in buttons:
                        if btn.hovered:
                            btn.click()
                            # Re-set caption after return
                            pygame.display.set_caption("BCI Studio - Main Menu")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
