import pygame
import sys
import os
import time
import datetime
import threading
import numpy as np
import pandas as pd
import glob
import joblib
import itertools

# --- FIX MATPLOTLIB (Indispensable sur Mac pour éviter le crash GUI) ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Traitement du signal et ML
import mne
from scipy.signal import butter, lfilter
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# PyRiemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# --- CONFIGURATION ---
BASE_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),"EEG_session_2026-02-03")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "resultsDelayEmbedding")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

SFREQ = 250
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']

# Couleurs Interface
BG_COLOR = (15, 15, 20)
TEXT_COLOR = (220, 220, 230)
ACCENT_COLOR = (0, 180, 255)
SUCCESS_COLOR = (50, 200, 80)
ERROR_COLOR = (255, 80, 80)

# --- FONCTIONS DE TRAITEMENT (Stateless pour scikit-learn) ---
def apply_bandpass(X, lowcut, highcut, sfreq=250):
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, X, axis=2)

def apply_logvar(X):
    var = np.var(X, axis=2)
    return np.log(var + 1e-6)


class DelayEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, delays=[1]):
        """
        delays: liste des décalages en 'samples'. 
        Exemple: [4] signifie qu'on ajoute t-4 (environ 16ms à 250Hz)
        """
        self.delays = delays

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = X.shape
        
        # On va stocker les signaux originaux + les signaux décalés
        X_aug = [X]
        
        for d in self.delays:
            if d > 0:
                # Décalage vers le passé (t-d)
                # On pad avec des zéros au début pour garder la même taille
                shifted = np.roll(X, shift=d, axis=2)
                shifted[:, :, :d] = 0 
                X_aug.append(shifted)
            elif d < 0:
                # Décalage vers le futur (t+d) (si on veut t+1)
                # Attention, en temps réel pur, le futur n'existe pas !
                # Mais en analyse de fenêtre glissante (buffer), on peut le faire.
                shifted = np.roll(X, shift=d, axis=2)
                shifted[:, :, d:] = 0
                X_aug.append(shifted)
                
        # On concatène sur l'axe des canaux (axis 1)
        # Résultat: (n_epochs, n_channels * (1 + n_delays), n_times)
        return np.concatenate(X_aug, axis=1)

# --- INTERFACE GRAPHIQUE ---
class TrainingGUI:
    def __init__(self, timeWindow=2.0, delays_mu=[4,8], delays_beta_low=[4,8], delays_beta_high=[4,8]):
        pygame.init()
        self.width, self.height = 900, 700 
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BCI Training (Sliding Window + Viz)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        self.status_message = "Initializing..."
        self.sub_message = ""
        self.progress = 0.0
        self.finished = False
        self.success = False
        self.accuracy = 0.0
        self.viz_image = None
        self.saved_files = []
        self.windowTime = timeWindow
        self.delays_mu = delays_mu
        self.delays_beta_low = delays_beta_low
        self.delays_beta_high = delays_beta_high
        
        self.thread = threading.Thread(target=self.run_training)
        self.thread.start()
        
    def get_all_csvs(self):
        pattern = os.path.join(BASE_DIR, "EEG_Session_*.csv")
        files = glob.glob(pattern)
        return sorted(files)

    def generate_separate_viz(self, clf, X, y, y_pred, y_proba, class_names):
        self.status_message = "Generating Graphs..."
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.saved_files = []

        # ---------------------------------------------------------
        # --- IMAGE 1 : MATRICE DE CONFUSION ---
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
        
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        # AJOUT DE L'ACCURACY DANS LE TITRE
        plt.title(f"Matrice de Confusion (CV Acc: {self.accuracy*100:.1f}%)")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        thresh = cm_norm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            plt.text(j, i, f"{cm_norm[i, j]*100:.0f}%\n({cm[i, j]})", 
                     horizontalalignment="center", color=text_color)
        plt.ylabel('Vraie Classe')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        
        file_matrix = os.path.join(RESULTS_DIR, f"confusion_matrix_{self.windowTime}_delay{self.delays_mu}_{self.delays_beta_low}_{self.delays_beta_high}_{timestamp}.png")
        plt.savefig(file_matrix, dpi=100)
        plt.close(fig1)
        self.saved_files.append(file_matrix)

        # ---------------------------------------------------------
        # --- IMAGE 2 : PCA (Zones) ---
        # ---------------------------------------------------------
        try:
            fig2 = plt.figure(figsize=(8, 6))
            # On transforme les données mais sans la dernière étape (le classifieur)
            X_feats = clf[:-1].transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_feats)
            
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            
            from sklearn.neighbors import KNeighborsClassifier
            # On utilise plus de voisins car on a plus de points (fenêtre glissante)
            knn_viz = KNeighborsClassifier(n_neighbors=5)
            knn_viz.fit(X_pca, y)
            Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            colors = ['#FF9999', '#66B3FF', '#99FF99'][:len(class_names)]
            cmap_light = ListedColormap(['#FFE5E5', '#E5F2FF', '#E5FFE5'][:len(class_names)])
            
            plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
            for i, color in zip(range(1, len(class_names)+1), colors):
                idx = np.where(y == i)
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=class_names[i-1], edgecolor='k', s=40, alpha=0.6)
            
            plt.title(f"Zones de Décision PCA (CV Acc: {self.accuracy*100:.1f}%)")
            plt.legend()
            plt.tight_layout()
            
            file_pca = os.path.join(RESULTS_DIR, f"pca_zones_{self.windowTime}_delay{self.delays_mu}_{self.delays_beta_low}_{self.delays_beta_high}_{timestamp}.png")
            plt.savefig(file_pca, dpi=100)
            plt.close(fig2)
            self.saved_files.append(file_pca)
        except Exception as e:
            print(f"PCA Viz warning: {e}")
        # ---------------------------------------------------------
        # --- IMAGE 3 : TRIANGLE (Simplex) ---
        # ---------------------------------------------------------
        if len(class_names) == 3:
            fig3 = plt.figure(figsize=(8, 8))
            corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
            triangle = plt.Polygon(corners, fill=None, edgecolor='k', linewidth=2)
            plt.gca().add_patch(triangle)
            
            plt.text(-0.05, -0.05, class_names[0], ha='right', fontweight='bold')
            plt.text(1.05, -0.05, class_names[1], ha='left', fontweight='bold')
            plt.text(0.5, np.sqrt(3)/2 + 0.05, class_names[2], ha='center', fontweight='bold')
            
            X_tri = y_proba[:, 1] + y_proba[:, 2] * 0.5
            Y_tri = y_proba[:, 2] * (np.sqrt(3)/2)
            
            for i, color in zip(range(1, 4), colors):
                idx = np.where(y == i)
                plt.scatter(X_tri[idx], Y_tri[idx], c=color, label=f"Vrai {class_names[i-1]}", edgecolor='k', s=40, alpha=0.5)
            
            plt.xlim(-0.1, 1.1); plt.ylim(-0.1, 1.0)
            plt.axis('off')
            plt.title("Triangle de Certitude")
            plt.legend(loc='upper right')
            
            file_tri = os.path.join(RESULTS_DIR, f"triangle_{self.windowTime}_delay{self.delays_mu}_{self.delays_beta_low}_{self.delays_beta_high}_{timestamp}.png")
            plt.savefig(file_tri, dpi=100)
            plt.close(fig3)
            self.saved_files.append(file_tri)

        # Charger l'image de la matrice pour l'affichage Pygame
        if os.path.exists(file_matrix):
            img = pygame.image.load(file_matrix)
            target_w = self.width - 100
            scale = target_w / img.get_width()
            target_h = int(img.get_height() * scale)
            if target_h > (self.height - 150):
                target_h = self.height - 150
                scale = target_h / img.get_height()
                target_w = int(img.get_width() * scale)
            self.viz_image = pygame.transform.scale(img, (target_w, target_h))

    def run_training(self):
        time.sleep(1)
        
        # 1. CHARGEMENT
        self.status_message = "Searching for data..."
        self.progress = 0.1
        csv_files = self.get_all_csvs()
        if not csv_files:
            self.status_message = "Error: No CSV file found."; self.finished = True; return
        
        self.status_message = "Loading & Merging..."
        self.progress = 0.2
        raw_list = []
        try:
            for fname in csv_files:
                try: df = pd.read_csv(fname, sep='\t', header=None)
                except: df = pd.read_csv(fname, sep=',', header=None)
                if df.shape[1] < 24: continue
                eeg = df.iloc[:, 1:9].values.T * 1e-6; markers = df.iloc[:, 23].values
                info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
                raw_tmp = mne.io.RawArray(eeg, info, verbose=False)
                stim_info = mne.create_info(['STI'], SFREQ, ['stim'])
                stim_raw = mne.io.RawArray(markers.reshape(1, -1), stim_info, verbose=False)
                raw_tmp.add_channels([stim_raw], force_update_info=True)
                raw_list.append(raw_tmp)
            if not raw_list: raise Exception("No valid data found")
            raw = mne.concatenate_raws(raw_list)
            events = mne.find_events(raw, stim_channel='STI', verbose=False)
            
            raw.pick(['eeg'])
            raw.notch_filter([50, 100], fir_design='firwin', verbose=False)
            raw.filter(4., 40., fir_design='firwin', verbose=False)
        except Exception as e:
            self.status_message = f"Error Loading: {str(e)[:20]}..."; print(e)
            self.finished = True; self.success = False; return

        # ---------------------------------------------------------
        # 3. EPOCHING (AVEC AUGMENTATION / SLIDING WINDOW)
        # ---------------------------------------------------------
        self.status_message = "Augmenting Data (Sliding Window)..."
        self.progress = 0.4
        event_id = {'LEFT': 1, 'RIGHT': 2, 'FEET': 3}
        class_names = ['LEFT', 'RIGHT', 'FEET']
        
        try:
            # Paramètres de la fenêtre glissante
            window_duration = self.windowTime  # Durée de chaque analyse (s)
            overlap = 0.2          # Décalage entre fenêtres (s)
            t_start = 0.5          # Ignorer le début (réaction)
            t_end_action = 3.5     # Fin de l'action
            
            aug_events_list = []
            
            for event in events:
                time_sample = event[0]
                type_code = event[2]
                
                if type_code in event_id.values():
                    current_start = t_start
                    # BOUCLE WHILE : On crée plusieurs fenêtres pour un seul événement
                    while (current_start + window_duration) <= t_end_action:
                        shift_samples = int(current_start * SFREQ)
                        new_event = [time_sample + shift_samples, 0, type_code]
                        aug_events_list.append(new_event)
                        current_start += overlap
            
            aug_events = np.array(aug_events_list)
            if len(aug_events) == 0: raise Exception("No events found")
            
            # Tri chronologique (important pour MNE)
            aug_events = aug_events[aug_events[:, 0].argsort()]

            # Création des epochs sur les données augmentées
            epochs = mne.Epochs(raw, aug_events, event_id, tmin=0, tmax=window_duration, 
                                proj=False, baseline=None, verbose=False, preload=True)
            
            self.sub_message = f"Augmented to {len(epochs)} epochs."
            
        except Exception as e:
            self.status_message = f"Epoching Error: {str(e)}"
            self.finished = True; self.success = False; return

        # 4. TRAINING & CROSS-VALIDATION
        self.status_message = "Training & Cross-Validation..."
        self.progress = 0.6
        try:
            cov_mu = make_pipeline(
                FunctionTransformer(apply_bandpass, kw_args={'lowcut':8, 'highcut':12, 'sfreq':SFREQ}, validate=False),
                DelayEmbedding(delays=self.delays_mu),
                Covariances(estimator='oas'), 
                TangentSpace())
            cov_beta_low = make_pipeline(
                FunctionTransformer(apply_bandpass, kw_args={'lowcut':12, 'highcut':20, 'sfreq':SFREQ}, validate=False),
                DelayEmbedding(delays=self.delays_beta_low),
                Covariances(estimator='oas'), 
                TangentSpace())
            cov_beta_high = make_pipeline(
                FunctionTransformer(apply_bandpass, kw_args={'lowcut':20, 'highcut':30, 'sfreq':SFREQ}, validate=False),
                DelayEmbedding(delays=self.delays_beta_high),
                Covariances(estimator='oas'), TangentSpace())
            """feat_logvar = make_pipeline(
                FunctionTransformer(apply_bandpass, kw_args={'lowcut':8, 'highcut':30, 'sfreq':SFREQ}, validate=False),
                FunctionTransformer(apply_logvar, validate=False))"""
            
            feats = FeatureUnion([('mu', cov_mu), ('beta_low', cov_beta_low), ('beta_high', cov_beta_high)]) #, ('logvar', feat_logvar)])
            clf = make_pipeline(feats, StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced'))
            
            X_train = epochs.get_data() 
            y_train = epochs.events[:, -1]
            
            n_samples = len(y_train)
            if n_samples < 6:
                print("Warning: Not enough data for CV. Fitting directly.")
                clf.fit(X_train, y_train)
                self.accuracy = 1.0; y_pred = y_train
                y_proba = np.zeros((len(y_train), 3))
                y_proba[np.arange(len(y_train)), y_train-1] = 1.0 
            else:
                n_splits = 5 if n_samples > 20 else 3
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                try:
                    y_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
                    y_proba = cross_val_predict(clf, X_train, y_train, cv=cv, method='predict_proba')
                    
                    # --- CALCUL DE L'ACCURACY ---
                    self.accuracy = accuracy_score(y_train, y_pred)
                    
                except Exception as cv_e:
                    print(f"CV Error: {cv_e}")
                    clf.fit(X_train, y_train)
                    self.accuracy = 0.0; y_pred = y_train
                    y_proba = np.zeros((len(y_train), 3))

            clf.fit(X_train, y_train)
            
            if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
            joblib.dump(clf, os.path.join(MODELS_DIR, "riemann_model.pkl"))
            
            self.progress = 0.9
            self.generate_separate_viz(clf, X_train, y_train, y_pred, y_proba, class_names)
            
        except Exception as e:
            self.status_message = f"Critical Error: {str(e)}"; print(e)
            self.finished = True; self.success = False; return

        self.status_message = "Done!"
        self.sub_message = f"Cross-Val Accuracy: {self.accuracy*100:.1f}% ({len(epochs)} samples)"
        self.progress = 1.0
        self.finished = True
        self.success = True

    def draw(self):
        self.screen.fill(BG_COLOR)
        cx, cy = self.width // 2, self.height // 2
        
        if self.finished and self.success and self.viz_image:
            img_rect = self.viz_image.get_rect(center=(cx, cy - 30))
            self.screen.blit(self.viz_image, img_rect)
            
            info_txt = f"Saved 3 graphs in 'results' folder."
            info_surf = self.font_small.render(info_txt, True, (100, 200, 100))
            self.screen.blit(info_surf, info_surf.get_rect(center=(cx, self.height - 70)))
        else:
            if not self.finished:
                bar_w, bar_h = 400, 10
                pygame.draw.rect(self.screen, (40, 40, 50), (cx - bar_w//2, cy + 40, bar_w, bar_h), border_radius=5)
                fill_w = int(self.progress * bar_w)
                if fill_w > 0:
                    pygame.draw.rect(self.screen, ACCENT_COLOR, (cx - bar_w//2, cy + 40, fill_w, bar_h), border_radius=5)
            
            status_surf = self.font.render(self.status_message, True, TEXT_COLOR)
            self.screen.blit(status_surf, status_surf.get_rect(center=(cx, cy - 40)))
            sub_surf = self.font_small.render(self.sub_message, True, (150, 150, 150))
            self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx, cy)))

        if self.finished:
            if self.success and not self.viz_image:
                 score_surf = self.font.render(f"Cross-Val Accuracy: {self.accuracy*100:.1f}%", True, SUCCESS_COLOR)
                 self.screen.blit(score_surf, score_surf.get_rect(center=(cx, 30)))
            
            instr = "Press SPACE to Close"
            col = SUCCESS_COLOR if self.success else ERROR_COLOR
            instr_surf = self.font_small.render(instr, True, col)
            self.screen.blit(instr_surf, instr_surf.get_rect(center=(cx, self.height - 30)))

    def run(self):
        running = True
        while running:
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and self.finished:
                    if event.key == pygame.K_SPACE: running = False
        pygame.quit()

if __name__ == "__main__":
    app = TrainingGUI(timeWindow=2.75, delays_mu=[4], delays_beta_low=[2,4,8], delays_beta_high=[2,4,8])
    app.run()