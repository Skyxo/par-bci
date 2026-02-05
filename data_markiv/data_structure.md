# Structure des Données EEG (`.csv`)

Ce document décrit le format des fichiers bruts générés par le script d'acquisition (ex: `EEG_Session_YYYY-MM-DD_HH-MM.csv`).

## 1. Format du Fichier
*   **Type** : CSV (BrainFlow format), séparateur TAB (`\t`).
*   **Source** : OpenBCI Cyton (8 canaux).
*   **Fréquence** : 250 Hz (1 échantillon toutes les 4 ms).

### Colonnes (Indices 0-23)
| Index | Nom | Description | Unité |
| :--- | :--- | :--- | :--- |
| **0** | `Sample Index` | Compteur d'échantillons (0-255 en boucle) | - |
| **1-8** | `EEG Channels` | **Données EEG brutes** (FC3, FC4, CP3, Cz, C3, C4, Pz, CP4) | µV (Microvolts) |
| **9-11** | `Accel` | Accéléromètre (X, Y, Z) | g |
| **12-21** | `Other` | Données internes OpenBCI (rarement utilisées) | - |
| **22** | `Timestamp` | Horodatage UNIX précis | Secondes |
| **23** | `Marker` | **Marqueurs d'événements** (Triggers) | Entier (Float) |

---

## 2. Marqueurs (Trigger Codes)
Les marqueurs apparaissent dans la **dernière colonne (23)**. Ils indiquent le début d'une tâche.

| Code | Classe / Événement | Description |
| :--- | :--- | :--- |
| **1.0** | **Main Gauche** (Left) | L'utilisateur imagine bouger sa main gauche. |
| **2.0** | **Main Droite** (Right) | L'utilisateur imagine bouger sa main droite. |
| **3.0** | **Pieds** (Feet) | L'utilisateur imagine bouger ses deux pieds. |
| **10.0** | **Repos** (Rest) | **Nouvelle classe active**. L'utilisateur ne fait rien (état neutre). |
| **99.0** | Fin de Run | Indique la pause entre deux blocs d'acquisition. |
| **0.0** | Rien | Pas d'événement à cet instant. |

> **Note importante** : Le marqueur est une "impulsion". Il n'apparaît que sur **une seule ligne** au début de l'action. Tout ce qui suit pendant 4 secondes appartient à cette classe.

---

## 3. Protocole d'Acquisition (Timing)
Chaque essai suit cette chronologie précise :

1.  **Fixation** (Croix `+`) : 2.0s à 3.0s (Aléatoire). Pas de marqueur.
2.  **ACTION (Tâche)** : **4.0s fixes**.
    *   Le marqueur (1, 2, 3 ou 10) est envoyé au début (t=0s).
    *   C'est la fenêtre à utiliser pour l'entraînement.
3.  **Relachement** (Écran noir) : 3.0s à 4.0s (Aléatoire). Repos inter-essai.

---

## 4. Résumé des Sessions Enregistrées
Voici le détail du contenu de chaque fichier CSV détecté :

| Fichier | Main G (1) | Main D (2) | Pieds (3) | Repos (10) | Total Essais | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **2026-01-14_13-35** | 30 | 30 | 30 | **90** | 180 | Session complète (mais déséquilibre Repos) |
| **2026-01-14_14-01** | 6 | 6 | 6 | 0 | 18 | Session partielle (Test ?) |
| **2026-01-14_14-06** | 6 | 6 | 6 | 0 | 18 | Session partielle (Test ?) |
| **2026-02-03_18-06** | 13 | 16 | 12 | 12 | 53 | **Valide mais atypique**. Contient 2 Runs Complets (2x24) et 1 Run avorté au début (5 essais). Une pause de 20min a eu lieu après le Run 2. |
| **2026-02-03_19-21** | **30** | **30** | **30** | **30** | **120** | **Session Parfaite (Équilibrée)** |

> [!TIP]
> Pour l'entraînement, privilégiez la session du **03/02 à 19h21**, car elle contient exactement 30 exemples par classe, parfaitement équilibrés.
