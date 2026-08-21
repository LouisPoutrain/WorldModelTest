# 🧠 Deep JEPA GridWorld

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

**Implémentation complète de l'architecture d'Intelligence Autonome à 6 modules de Yann LeCun**, appliquée à un environnement de planification complexe (GridWorld procédural avec labyrinthes, pièges en U et gestion d'énergie).

Ce projet s'inspire directement du papier [*A Path Towards Autonomous Machine Intelligence*](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022) et de l'implémentation [le-wm](https://github.com/lucas-maes/le-wm.git) pour la régularisation SIGReg.

---

## ✨ Features Principales

- **Espace Latent Auto-Supervisé** — L'encodeur CNN (`Perception`) compresse la grille 10×10 en un vecteur dense de 32 dimensions, régularisé par **SIGReg** (Sketch Isotropic Gaussian Regularizer) pour prévenir le *Latent Collapse* sans heuristique EMA.
- **World Model Prédictif (RNN)** — Un `GRUCell` avec prédiction résiduelle (`WorldModel`) simule la physique du monde "dans l'imagination" de l'agent, avec un *Rollout Drift* inférieur à 0.5 MSE sur 10 pas.
- **Planification par Cross-Entropy Method (CEM)** — L'`Actor` échantillonne des centaines de trajectoires futures dans le World Model et sélectionne la meilleure via un algorithme élitiste avec lissage de Laplace.
- **Critique N-Step TD Learning** — Le réseau `Cost` apprend la valeur à long terme des états via l'équation de Bellman sur N pas, avec un Target Network stabilisé par mise à jour Polyak (τ).
- **Entraînement Asynchrone Modulaire** — Les modules peuvent être entraînés de manière isolée (gel de la `Perception` et du `WorldModel` pour muscler le `Cost`).

---

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/LouisPoutrain/WorldModelTest.git
cd WorldModelTest

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- `torch` (PyTorch 2.x)
- `numpy`
- `matplotlib`
- `scikit-learn` (pour l'évaluation Linear Probing)
- `tqdm`

---

## ⚡ Quickstart

### Entraînement complet (GPU recommandé)
```bash
python3 main_colab.py
```

### Entraînement isolé du Critique (CPU suffit)
```bash
python3 train_critic_only.py
```

### Évaluation
```bash
# Diagnostic complet
python3 diagnostic.py

# Évaluations individuelles
python3 eval/eval_perception.py    # Linear Probing (R² score)
python3 eval/eval_dynamics.py      # Rollout Drift (MSE)
python3 eval/eval_behavior.py      # Généralisation ID / OOD
```

### Visualisation
```bash
python3 visualize.py               # Voir l'agent naviguer en temps réel
```

---

## 📁 Structure du Projet

```
WorldModelTest/
├── modules/                    # Les 6 modules cognitifs (LeCun)
│   ├── perception.py           #   Encodeur CNN (ResNet) → espace latent
│   ├── world_model.py          #   ARPredictor RNN (GRUCell)
│   ├── cost.py                 #   Critique (TD-Learning)
│   ├── actor.py                #   Planificateur CEM
│   ├── configurator.py         #   Gestionnaire de buts (énergie / cible)
│   ├── memory.py               #   Replay Buffer (ShortTermMemory)
│   └── sigreg.py               #   Régulariseur SIGReg
├── env/
│   └── gridworld.py            # Environnement GridWorld procédural
├── eval/                       # Suite d'évaluation scientifique
│   ├── eval_perception.py      #   Test 1 : Linear Probing
│   ├── eval_dynamics.py        #   Test 2 : Rollout Drift
│   └── eval_behavior.py        #   Test 3 : Généralisation OOD
├── checkpoints/                # Poids sauvegardés des modèles
├── logs/                       # Courbes d'entraînement (CSV + PNG)
├── docs/                       # Documentation technique
├── main_colab.py               # Script d'entraînement principal
├── train_critic_only.py        # Entraînement isolé du Critique
├── diagnostic.py               # Diagnostic exhaustif de l'agent
└── visualize.py                # Visualisation en temps réel
```

---

## 📖 Documentation

| Document | Description |
|---|---|
| [`docs/1_ARCHITECTURE.md`](docs/1_ARCHITECTURE.md) | Théorie des 6 modules cognitifs de LeCun |
| [`docs/2_MODULES_API.md`](docs/2_MODULES_API.md) | API technique de chaque module Python |
| [`docs/3_TRAINING_EVAL.md`](docs/3_TRAINING_EVAL.md) | Protocoles d'entraînement et d'évaluation |
| [`docs/4_CONFIGURATION.md`](docs/4_CONFIGURATION.md) | Guide des hyperparamètres |

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Dimensions latentes actives | **32/32** (pas de collapse) |
| Rollout Drift à t+10 | **0.49 MSE** |
| Succès In-Distribution | **~50%** |
| Succès OOD (Piège en U) | **0%** (limite théorique du CEM) |
