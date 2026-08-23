# Implémentation de l'Architecture d'Intelligence Autonome (Yann LeCun) dans un Gridworld

Ce document détaille le plan d'implémentation pour le développement de l'architecture d'intelligence autonome de Yann LeCun appliquée à un environnement Gridworld 2D. 

L'objectif est de mettre en place les 6 modules (Perception, Configurateur, World Model, Cost, Actor, Short-Term Memory) et l'environnement Gridworld, ainsi que la boucle principale (Mode-2) d'exécution et d'apprentissage en utilisant PyTorch.

## User Review Required

> [!IMPORTANT]
> L'architecture est très riche et inclut à la fois un apprentissage auto-supervisé (World Model) et un apprentissage par renforcement (Critic). 
> Pouvez-vous valider que cette structuration modulaire (un fichier par module) et les dimensions proposées pour l'espace latent (ex. 16 ou 32) conviennent à votre cas d'usage ?

## Open Questions

> [!WARNING]
> 1. Souhaitez-vous que l'entraînement du World Model et du Critic se fasse en ligne (à chaque pas de temps) ou par batch réguliers (ex. tous les N pas) à partir du Replay Buffer ?
> 2. Pour le Mode-1 de l'Actor (réactif), souhaitez-vous l'implémenter et l'entraîner via imitation du Mode-2 (Behavioral Cloning) ou devons-nous nous concentrer uniquement sur le Mode-2 (Planification) dans un premier temps ?
> 3. Quelles sont les dimensions exactes souhaitées pour la grille (par défaut 10x10) et la position initiale de la cible et des stations de recharge ?

## Proposed Changes

Nous proposons de structurer le projet en plusieurs fichiers Python clairs et modulaires. Tous les composants utiliseront `torch` et `torch.nn`. Le code sera placé dans votre répertoire de projet (WorldModelTest).

### Structure du projet

- `env/gridworld.py` : L'environnement (état de la grille, position de l'agent, énergie, génération de la vision locale 5x5).
- `modules/perception.py` : L'encodeur (CNN + MLP) transformant la vision locale $5 \times 5$ en état latent $s_t$.
- `modules/configurator.py` : Module déterminant les poids des coûts et le sous-objectif (cible ou station de recharge) en fonction de l'énergie.
- `modules/world_model.py` : Le modèle prédictif (JEPA) prédisant $\hat{s}_{t+1}$ à partir de $s_t$ et $a_t$.
- `modules/cost.py` : Contient le coût intrinsèque (fonction mathématique) et le Critic (réseau de neurones prédisant la valeur à long terme).
- `modules/actor.py` : Implémente le planificateur Mode-2 (MPC - Model Predictive Control) générant les séquences d'actions simulées.
- `modules/memory.py` : Le Replay Buffer stockant les transitions et les trajectoires.
- `main.py` : La boucle principale liant tous les modules ensemble.

---

### Détails des Composants

#### 1. Environnement (`env/gridworld.py`)
- **Classe `GridWorldEnv`** : 
  - Matrice 2D (ex. 10x10).
  - Éléments: Agent, Cible, Stations de recharge, Obstacles statiques/dynamiques.
  - Fonction `step(action)` : Met à jour la position de l'agent, diminue l'énergie de 1, gère les collisions, retourne la nouvelle perception locale $5 \times 5$, le reward/coût, et un booléen `done`.
  - Fonction `get_local_observation()` : Extrait un tenseur de dimension $(C, 5, 5)$ centré sur l'agent (où $C$ représente différents canaux : obstacles, cible, station, agent).

#### 2. Perception (`modules/perception.py`)
- **Classe `Perception(nn.Module)`** :
  - **Entrée** : Tenseur $(B, C, 5, 5)$.
  - **Architecture** : 2 couches de convolutions suivies d'une couche linéaire.
  - **Sortie** : Vecteur latent $s_t$ de dimension $d_{latent} = 32$.

#### 3. Configurateur (`modules/configurator.py`)
- **Classe `Configurator`** :
  - Reçoit l'état interne (Énergie $E$, position globale des stations si supposées connues du configurateur, ou utilise une représentation fixe pour les objectifs).
  - **Sorties** : $w_{energy}$, $w_{collision}$, $w_{goal}$ et $s_{goal}$.
  - Logique : Si $E \le 30$, $s_{goal} = s_{station}$ et $w_{energy}$ est augmenté. Sinon, $s_{goal} = s_{cible}$.

#### 4. World Model (`modules/world_model.py`)
- **Classe `WorldModel(nn.Module)`** :
  - **Entrée** : $s_t$ (dim 32) concaténé avec $a_t$ (one-hot, dim 4).
  - **Architecture** : MLP à 2 ou 3 couches avec activations ReLU.
  - **Sortie** : $\hat{s}_{t+1}$ (dim 32).
  - Entraîné via une perte MSE : $\mathcal{L} = || \hat{s}_{t+1} - s_{t+1} ||^2$.

#### 5. Coût et Critique (`modules/cost.py`)
- **Fonction `intrinsic_cost`** :
  - Calcule la distance entre $s_t$ et $s_{goal}$, pénalise les collisions (si détectables dans $s_t$) et le manque d'énergie.
- **Classe `Critic(nn.Module)`** :
  - **Entrée** : $s_t$ (dim 32).
  - **Sortie** : Valeur scalaire $V(s_t)$.
  - Entraîné par TD-Learning (ex. TD(0)) : $\mathcal{L}_{critic} = \text{MSE}(V(s_t), C_{intr} + \gamma V(s_{t+1}))$.

#### 6. Acteur (`modules/actor.py`)
- **Classe `Actor`** (Mode-2) :
  - Paramètres : Nombre de séquences $N$ (ex. 20), Horizon $H$ (ex. 5).
  - Fonctionnalité : Boucle sur $N$ séquences de $H$ actions aléatoires. Pour chaque séquence, simule les états futurs via le `WorldModel`, calcule le coût total via `intrinsic_cost` et la pénalité finale via le `Critic`.
  - Retourne la séquence optimale et, plus spécifiquement, la première action $a_0$ à exécuter.

#### 7. Mémoire (`modules/memory.py`)
- **Classe `ShortTermMemory`** :
  - Buffer circulaire avec une capacité maximale.
  - Méthodes `push(transition)` et `sample(batch_size)`.

#### 8. Boucle Principale (`main.py`)
- Intégration de la logique étape par étape de la boucle Mode-2 (Perception $\to$ Configurator $\to$ Planification $\to$ Exécution $\to$ Apprentissage).

## Verification Plan

### Test unitaire et Vérification Manuelle
- Écrire un script de test simple pour chaque module (`test_env.py`, `test_modules.py`) afin de s'assurer que les dimensions des tenseurs (entrées/sorties) correspondent parfaitement à travers les 6 modules.
- Lancer la boucle principale sur quelques épisodes pour valider que :
  - L'agent planifie et prend des actions (Mode-2).
  - Le World Model et le Critic mettent à jour leurs poids sans provoquer d'erreur.
  - Le Configurateur réagit correctement à la baisse de l'énergie (changement de $s_{goal}$).
