# 🚀 Installation sur Serveur GPU (Remote)

Ce guide explique comment installer l'environnement complet sur votre serveur distant (Linux/Windows) pour utiliser le GPU avec PyTorch.

## 1. Pré-requis : Miniconda
Si Conda n'est pas installé sur le serveur, téléchargez et installez Miniconda.

**Commande Linux (Terminal) :**
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 2. Créer l'environnement (Recommandé)
Utilisez le fichier `environment.yml` que j'ai créé. Il contient tout le nécessaire (CUDA, PyTorch, MNE, etc.).

```bash
# Aller dans le dossier du projet
cd /chemin/vers/par-bci

# Créer l'environnement 'bci_env'
conda env create -f environment.yml

# Activer l'environnement
conda activate bci_env
```

## 3. Option Alternative (Pip uniquement)
Si vous ne pouvez pas utiliser Conda, utilisez `pip` avec le fichier `requirements_server.txt` (configuré pour CUDA 12.1).

```bash
# (Optionnel) Créer un venv
python -m venv venv
source venv/bin/activate  # Sur Linux
# venv\Scripts\activate   # Sur Windows

# Installer les dépendances
pip install -r requirements_server.txt
```

## 4. Vérification
Pour vérifier que le GPU est bien détecté :

```bash
python -c "import torch; print(f'CUDA Reference: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 5. Lancer l'entraînement
Vous pouvez maintenant lancer vos scripts d'entraînement :

```bash

## 6. Transfert de Données (PC -> Serveur)

Pour envoyer vos enregistrements EEG (les fichiers `.csv`) de votre PC vers le serveur, n'utilisez pas Git. Utilisez la commande `scp` (Secure Copy) depuis le terminal de votre PC Windows.

**Syntaxe :**
```powershell
scp -P [PORT] [FICHIER_LOCAL] [UTILISATEUR]@[IP]:[DESTINATION]
```

**Exemple concret :**
Si vous voulez envoyer un fichier de session vers le dossier `par-bci` du serveur :

```powershell
# Commande à lancer depuis VOTRE PC (pas sur le serveur)
scp -P 2222 EEG_Session_2026-01-14_13-35.csv projet11@delorean1:~/par-bci/
```

*Note : Remplacez `2222` par le vrai port SSH s'il est différent, et vérifiez l'IP.*
