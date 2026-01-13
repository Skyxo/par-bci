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

# --- MARQUEURS (CORRIGÉS) ---
# IMPORTANT : J'ai remplacé 0 par 10 pour la Baseline car BrainFlow interdit le 0
MARKER_BASELINE = 10 
MARKER_LEFT = 1 
MARKER_RIGHT = 2 
MARKER_FEET = 3 

# --- TIMINGS ---
t_BASELINE = 2.0 # Repos avec croix
# La flèche restera affichée pendant t_CUE + t_IMAGERY
t_CUE = 1.0 
t_IMAGERY = 3.0 
t_TOTAL_ACTION = t_CUE + t_IMAGERY # = 4.0 secondes

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

    def init_hardware(self):
        try:
            self.board.prepare_session()
            self.board.start_stream()
            print("--- FLUX EEG DÉMARRÉ ---")
            time.sleep(2)
        except BrainFlowError as e:
            print(f"ERREUR CRITIQUE: {e}")
            self.running = False

    def init_graphics(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Acquisition EEG - Appuyez sur ECHAP pour arrêter et sauver")
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
        elif direction == 'right':
            points = [(cx + 100, cy), (cx - 50, cy - 80), (cx - 50, cy + 80)]
            label = "MAIN DROITE"
        elif direction == 'feet':
            points = [(cx, cy + 100), (cx - 80, cy - 50), (cx + 80, cy - 50)]
            label = "PIEDS"
        
        pygame.draw.polygon(self.screen, color, points)
        text = self.font.render(label, True, (255, 255, 255))
        text_rect = text.get_rect(center=(cx, cy - 150))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def draw_text_screen(self, text_lines):
        self.screen.fill(self.bg_color)
        cy = self.height // 2 - (len(text_lines) * 20)
        for line in text_lines:
            txt_surf = self.font_small.render(line, True, (255, 255, 255))
            rect = txt_surf.get_rect(center=(self.width//2, cy))
            self.screen.blit(txt_surf, rect)
            cy += 40
        pygame.display.flip()

    def run_experiment(self):
        self.init_hardware()
        self.init_graphics()

        if not self.running: return

        self.draw_text_screen([
            "PROTOCOLE D'IMAGERIE MOTRICE - 30 ESSAIS PAR CLASSE",
            "",
            "IMPORTANT: Imaginez INTENSÉMENT le mouvement.",
            "Main gauche: Serrez le poing gauche mentalement.",
            "Main droite: Serrez le poing droit mentalement.",
            "Pieds: Bougez les orteils mentalement.",
            "",
            "Appuyez sur ESPACE pour commencer."
        ])
        if not self.wait_for_start(): return

        for run_idx in range(NUMBER_OF_RUNS):
            print(f"--- DÉBUT RUN {run_idx + 1}/{NUMBER_OF_RUNS} ---")
            
            trials = [MARKER_LEFT] * TRIALS_PER_CLASS_PER_RUN + \
                     [MARKER_RIGHT] * TRIALS_PER_CLASS_PER_RUN + \
                     [MARKER_FEET] * TRIALS_PER_CLASS_PER_RUN
            random.shuffle(trials)

            for i, marker in enumerate(trials):
                # 1. BASELINE (Repos)
                self.draw_fixation()
                try:
                    self.board.insert_marker(MARKER_BASELINE)
                except Exception:
                    pass # Evite crash si erreur marker
                
                print(f"Run {run_idx+1} | Essai {i+1}: Repos")
                # Si smart_sleep renvoie False, on sort de la boucle TOTALEMENT
                if not self.smart_sleep(t_BASELINE): 
                    self.save_and_quit()
                    return

                # 2. STIMULUS + IMAGERIE (Flèche maintenue)
                if marker == MARKER_LEFT: dir_str = 'left'
                elif marker == MARKER_RIGHT: dir_str = 'right'
                else: dir_str = 'feet'
                
                self.draw_cue(dir_str)
                self.board.insert_marker(marker)
                print(f"Run {run_idx+1} | Essai {i+1}: ACTION ({dir_str})")
                
                # On attend toute la durée (Cue + Imagery) sans effacer l'écran
                if not self.smart_sleep(t_TOTAL_ACTION):
                    self.save_and_quit()
                    return

                # 3. PAUSE NOIRE (Random)
                self.screen.fill((0,0,0)) 
                pygame.display.flip()
                inter_trial = random.uniform(1.5, 2.5)
                if not self.smart_sleep(inter_trial):
                    self.save_and_quit()
                    return

            # Fin du Run
            if run_idx < NUMBER_OF_RUNS - 1:
                self.draw_text_screen([f"FIN DU RUN {run_idx + 1}", "Appuyez sur ESPACE pour continuer."])
                self.board.insert_marker(99)
                if not self.wait_for_start(): return

        self.save_and_quit()

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