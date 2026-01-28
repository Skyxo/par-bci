import time
import datetime
import numpy as np
import pygame
import random
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter

# ================= CONFIGURATION DU PROTOCOLE =================
# Mettre à False pour connecter le vrai casque OpenBCI
SIMULATION_MODE = False 
COM_PORT = "COM10" # Windows: 'COM3', Linux: '/dev/ttyUSB0', Mac: '/dev/cu.usbserial...'
RECORD_REST_MARKERS = True # Re-enabled as requested

# --- MARQUEURS (CORRIGÉS) ---
# IMPORTANT : J'ai remplacé 0 par 10 pour la Baseline car BrainFlow interdit le 0
MARKER_BASELINE = 10 
MARKER_LEFT = 1 
MARKER_RIGHT = 2 
MARKER_FEET = 3 

# --- TIMINGS ---
# --- TIMINGS (ACCELERATED) ---
t_BASELINE = 1.25      # (Reference only, used randomized below)
t_ACTION = 4.0         # Maintained (Brain needs time to generate pattern)
t_RELAX = 1.0          # Reduced from 2.0
t_REST_MIN = 1.5       # Reduced from 2.5
t_REST_MAX = 2.5       # Reduced from 4.0

# --- STRUCTURE ---
# Pour un bon entraînement BCI: minimum 30 essais par classe
TRIALS_PER_CLASS_PER_RUN = 6  # 6 essais par classe par run
NUMBER_OF_RUNS = 5  # 5 runs = 30 essais par classe au total

class BCIProtocol:
    def __init__(self):
        self.running = True
        self.screen = None
        self.width, self.height = 800, 600
        self.bg_color = (30, 30, 30)
        
        self.params = BrainFlowInputParams()
        if SIMULATION_MODE:
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            self.board_id = BoardIds.CYTON_BOARD.value
        self.params.serial_port = COM_PORT
        
        self.board = BoardShim(self.board_id, self.params)

        self.connected = False # Connection state flag
        
    def init_hardware(self):
        try:
            if self.board.is_prepared():
                self.board.release_session()
            self.board.prepare_session()
            self.board.start_stream()
            print("--- FLUX EEG DÉMARRÉ ---")
            self.connected = True
            time.sleep(2)
        except BrainFlowError as e:
            print(f"ERREUR CONNEXION: {e}")
            self.connected = False
            # self.running = False # No longer kill app, just flag it

    def init_graphics(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Acquisition EEG - Standard Graz Protocol")
        self.font = pygame.font.SysFont('Arial', 50)
        self.font_small = pygame.font.SysFont('Arial', 30)

    # --- NOUVELLE FONCTION INTELLIGENTE ---
    def smart_sleep(self, duration):
        """
        Remplace time.sleep(). Attend 'duration' secondes,
        MAIS vérifie en continu si l'utilisateur veut quitter.
        Renvoie False si l'utilisateur a annulé, True sinon.
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            # Vérification des événements pendant l'attente
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\n!!! ARRÊT D'URGENCE DEMANDÉ PAR L'UTILISATEUR !!!")
                        return False
            
            # Petite pause pour ne pas surcharger le processeur
            time.sleep(0.01) 
        return True

    def draw_fixation(self):
        self.screen.fill(self.bg_color)
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.line(self.screen, (200, 200, 200), (cx - 20, cy), (cx + 20, cy), 4)
        pygame.draw.line(self.screen, (200, 200, 200), (cx, cy - 20), (cx, cy + 20), 4)
        pygame.display.flip()

    def draw_cue(self, direction):
        self.screen.fill(self.bg_color)
        cx, cy = self.width // 2, self.height // 2
        color = (0, 255, 127) 
        
        if direction == 'left':
            points = [(cx - 100, cy), (cx + 50, cy - 80), (cx + 50, cy + 80)]
            label = "MAIN GAUCHE"
            pygame.draw.polygon(self.screen, color, points)
        elif direction == 'right':
            points = [(cx + 100, cy), (cx - 50, cy - 80), (cx - 50, cy + 80)]
            label = "MAIN DROITE"
            pygame.draw.polygon(self.screen, color, points)
        elif direction == 'feet':
            points = [(cx, cy + 100), (cx - 80, cy - 50), (cx + 80, cy - 50)]
            label = "PIEDS"
            pygame.draw.polygon(self.screen, color, points)
        elif direction == 'rest':
            # Circle for Rest
            pygame.draw.circle(self.screen, (100, 100, 255), (cx, cy), 60)
            label = "REPOS (NE RIEN FAIRE)"
        
        text = self.font.render(label, True, (255, 255, 255))
        text_rect = text.get_rect(center=(cx, cy - 150))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def draw_text_screen(self, text_lines):
        """Affiche un écran de texte multiline et attend."""
        self.screen.fill(self.bg_color)
        cx, cy = self.width // 2, self.height // 2
        
        # Start drawing from a bit higher up
        start_y = cy - (len(text_lines) * 40) // 2
        
        for i, line in enumerate(text_lines):
            # Render text
            if i == 0 or "RUN" in line: # Title or Important
                font = self.font
                color = (255, 255, 0) # Yellow for titles
                if "ERREUR" in line: color = (255, 50, 50) # Red for errors
            else:
                font = self.font_small
                color = (255, 255, 255)
            
            text = font.render(line, True, color)
            rect = text.get_rect(center=(cx, start_y + i * 40))
            self.screen.blit(text, rect)
            
        pygame.display.flip()

    def run_experiment(self):
        self.init_hardware()
        self.init_graphics()

        # LOOP UNTIL CONNECTED
        while not self.connected:
            self.draw_text_screen([
                "ERREUR DE CONNEXION AU CASQUE",
                "",
                f"Port: {COM_PORT}",
                "Vérifiez que le dongle est branché et le casque allumé.",
                "",
                "[ESPACE] : Réessayer la connexion",
                "[ECHAP] : Quitter"
            ])
            if not self.wait_for_retry(): return

        self.draw_text_screen([
            "PROTOCOLE D'IMAGERIE MOTRICE - GRAZ ACTIVE REST",
            "",
            "3 PHASES par essai :",
            "1. Croix (+): Attention (1-1.5s)",
            "2. Action/Repos : Imaginez OU Reposez-vous (4s)",
            "3. Relâchement : Pause courte (2s)",
            "",
            "Le 'REPOS' est maintenant une action comme les autres.",
            "Restez concentré du début à la fin !",
            "",
            "Appuyez sur ESPACE pour commencer."
        ])
        if not self.wait_for_start(): return

        for run_idx in range(NUMBER_OF_RUNS):
            print(f"--- DÉBUT RUN {run_idx + 1}/{NUMBER_OF_RUNS} ---")
            
            # 1. LONG SETTLING (5s Black Screen)
            self.screen.fill((0,0,0))
            pygame.display.flip()
            print("Période de stabilisation (5s)...")
            if not self.smart_sleep(5.0): return

            # TACHES: Inclut maintenant le REPOS comme une classe active
            trials = [MARKER_LEFT] * TRIALS_PER_CLASS_PER_RUN + \
                     [MARKER_RIGHT] * TRIALS_PER_CLASS_PER_RUN + \
                     [MARKER_FEET] * TRIALS_PER_CLASS_PER_RUN + \
                     [MARKER_BASELINE] * TRIALS_PER_CLASS_PER_RUN # REPOS
            
            random.shuffle(trials)

            for i, marker in enumerate(trials):
                # -------------------------------------------------
                # PHASE 1: FIXATION / PRE-CUE (Black Screen + Cross)
                # -------------------------------------------------
                # Randomize duration (1.0s to 1.5s) to avoid rhythm locking
                t_fixation = random.uniform(1.0, 1.5)
                
                self.draw_fixation()
                
                # NOTE: We do NOT insert a marker here anymore. 
                # Rest is now an explicit class in Phase 2.
                
                print(f"Run {run_idx+1} | Essai {i+1}: Fixation ({t_fixation:.1f}s)")
                if not self.smart_sleep(t_fixation): 
                    self.save_and_quit()
                    return

                # -------------------------------------------------
                # PHASE 2: ACTIVE TASK (Action OR Rest) (4.0s)
                # -------------------------------------------------
                dir_str = 'unknown'
                if marker == MARKER_LEFT: dir_str = 'left'
                elif marker == MARKER_RIGHT: dir_str = 'right'
                elif marker == MARKER_FEET: dir_str = 'feet'
                elif marker == MARKER_BASELINE: dir_str = 'rest'
                
                self.draw_cue(dir_str)
                self.board.insert_marker(marker)
                print(f"  -> TACHE: {dir_str} (4s)")
                
                if not self.smart_sleep(t_ACTION):
                    self.save_and_quit()
                    return

                # -------------------------------------------------
                # PHASE 3: RELAXATION / INTERVAL (2.0s)
                # -------------------------------------------------
                # Mandatory break between trials to allow ERS rebound
                self.screen.fill((0,0,0)) 
                pygame.display.flip()
                print("  -> Relâchement (Inter-Essai) (2s)")
                if not self.smart_sleep(t_RELAX):
                    self.save_and_quit()
                    return
                
                # NO PHASE 4 (Systematic Rest) anymore!
                # We loop directly to next Fixation.

            # Fin du Run - Prompt Utilisateur
            if run_idx < NUMBER_OF_RUNS - 1:
                self.draw_text_screen([
                    f"FIN DU RUN {run_idx + 1}", 
                    "", 
                    "[ESPACE] : Continuer (Prochain Run)", 
                    "[ECHAP] : Arrêter et Sauvegarder"
                ])
                # Insert End of Run Marker
                self.board.insert_marker(99)
                if not self.wait_for_start(): return

        self.save_and_quit()

    def wait_for_retry(self):
        """Menu boucle pour réessayer la connexion"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_and_quit()
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.init_hardware() # Retry
                        return True # Break loop to check self.connected
                    if event.key == pygame.K_ESCAPE:
                        self.save_and_quit()
                        return False
            time.sleep(0.05)
        return True

    def wait_for_start(self):
        """Attend Espace pour continuer ou Echap pour quitter"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_and_quit()
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        self.save_and_quit()
                        return False
            time.sleep(0.05)
        return True

    def save_and_quit(self):
        print("\n--- ARRÊT DU PROTOCOLE ---")
        print("Sauvegarde des données en cours...")
        
        # Récupération sécurisée des données
        try:
            data = self.board.get_board_data()
            self.board.stop_stream()
            self.board.release_session()
        except:
            print("Avertissement: Session déjà fermée ou pas de données.")
            data = None

        if data is not None and data.size > 0:
            filename = f"EEG_Session_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
            DataFilter.write_file(data, filename, 'w')
            print(f"✅ SUCCÈS : Fichier sauvegardé : {filename}")
            print(f"Échantillons récupérés : {data.shape[1]}")
        else:
            print("❌ Aucune donnée à sauvegarder.")

        pygame.quit()
        exit()

if __name__ == "__main__":
 app = BCIProtocol()
 app.run_experiment()