# ⚙️ Configuration & Hyperparamètres

> Guide pour ajuster le comportement de l'agent sans modifier le code Python.

---

## Hyperparamètres du Planificateur CEM (`Actor`)

Le planificateur CEM est instancié dans les scripts avec des arguments nommés :

```python
actor = Actor(
    action_dim=4,           # Nombre d'actions possibles (Haut, Bas, Gauche, Droite)
    num_sequences=500,      # N : Trajectoires simulées par itération
    horizon=10,             # H : Profondeur de simulation (nombre de pas)
    cem_iterations=10,      # M : Itérations de raffinement
    elite_size=50,          # K : Séquences élites retenues
    gamma=0.9               # Discount factor
)
```

### Impact des paramètres CEM

| Paramètre | ↑ Augmenter | ↓ Diminuer |
|---|---|---|
| `num_sequences` (N) | Meilleure couverture de l'espace d'actions. Plans plus intelligents. | Plus rapide mais plans sous-optimaux. |
| `horizon` (H) | Vision à plus long terme. Peut anticiper les pièges. | Vision myope. Réagit au dernier moment. |
| `cem_iterations` (M) | Distribution d'actions plus concentrée. Plans plus précis. | Plans plus bruités. |
| `elite_size` (K) | Plus de diversité dans les séquences retenues. | Convergence plus agressive vers la meilleure trajectoire. |

**⚠️ Règle d'or :** Le `horizon` d'évaluation ne doit **jamais** dépasser le `horizon` d'entraînement. Le World Model n'a pas été entraîné à imaginer au-delà de 10 pas ; au-delà, ses prédictions dérivent (hallucination), et les plans du CEM deviennent incohérents.

### Configurations recommandées

| Profil | `num_sequences` | `horizon` | `cem_iterations` | `elite_size` | Usage |
|---|---|---|---|---|---|
| **Rapide** | 50 | 5 | 3 | 5 | Debug, exploration bruitée pour le Replay Buffer |
| **Standard** | 500 | 10 | 10 | 50 | Évaluation et visualisation |
| **Haute qualité** | 2000 | 10 | 15 | 100 | Benchmark final (lent mais optimal) |

---

## Hyperparamètres d'Apprentissage du Critique

Ces paramètres se trouvent dans `train_critic_only.py` :

```python
num_episodes = 5000       # Nombre d'épisodes d'exploration
batch_size = 128          # Taille du batch pour le TD-Learning
gamma = 0.90              # Discount factor
epsilon = 0.5             # Taux d'exploration ε-greedy
seq_len = 3               # N dans "N-Step TD" (profondeur de la somme de Bellman)
tau = 0.05                # Coefficient de mise à jour Polyak du Target Cost
```

### Impact des paramètres d'apprentissage

| Paramètre | ↑ Augmenter | ↓ Diminuer |
|---|---|---|
| `gamma` | Horizon de "vision" plus long. Le Critique voit loin dans le futur. Convergence plus lente. | Vision à court terme. Convergence rapide mais rate les impasses lointaines. |
| `epsilon` | Plus d'exploration aléatoire. L'agent tombe dans tous les pièges → données riches. | L'agent suit le CEM → données biaisées vers les chemins "faciles". |
| `seq_len` | Propagation des récompenses plus profonde. Le Critique anticipe mieux. | Plus réactif mais moins de contexte temporel. |
| `tau` | Le Target Network se met à jour vite. Apprentissage instable mais adaptatif. | Target très stable. Apprentissage lent mais régulier. |
| `lr` (optimizer) | Apprentissage agressif. Risque d'oscillation. | Apprentissage stable. Nécessite plus d'épisodes. |

### Recommandations

**Pour une grille 10×10 avec des épisodes de 20-50 pas :**
- `gamma = 0.90` est optimal. Un γ de 0.99 propage la valeur trop loin et lisse excessivement les récompenses.
- `epsilon = 0.5` est le bon compromis. En dessous de 0.3, l'agent n'explore pas assez les impasses.
- `seq_len = 3` avec `gamma = 0.90` donne un horizon effectif de ~10 pas (suffisant pour voir l'entrée d'un piège en U).

---

## Configuration de l'Environnement (`GridWorldEnv`)

```python
env = GridWorldEnv(
    size=10,                # Taille de la grille (size × size)
    max_energy=100,         # Énergie maximale de l'agent
    procedural=True,        # True = grilles aléatoires, False = grille fixe
    obstacle_density=0.15   # Proportion d'obstacles (15% de la grille)
)
```

### Système de récompenses

| Événement | Récompense |
|---|---|
| Pas normal | -1.0 |
| Collision (mur ou bord) | -5.0 |
| Station de recharge atteinte | +10.0 |
| Cible atteinte | +100.0 |

### Observation

L'observation est un tenseur `(4, size, size)` à 4 canaux binaires :

| Canal | Contenu |
|---|---|
| 0 | Obstacles (1.0 = mur) |
| 1 | Position de la cible |
| 2 | Position de la station de recharge |
| 3 | Position de l'agent |

---

## Configuration du Configurator

Le `Configurator` ne prend qu'un seul argument à l'initialisation :

```python
configurator = Configurator(latent_dim=32)
```

Il nécessite ensuite un appel à `set_goals(s_target, s_station)` pour enregistrer les vecteurs latents des objectifs (obtenus en passant des observations synthétiques dans la `Perception`).

### Seuil d'énergie

Le basculement entre la cible et la station se fait à **30 unités d'énergie** (codé en dur dans `get_configuration(energy)`). En dessous de 30, l'agent priorise la recharge avec `w_goal = 2.0` et `w_energy = 5.0`.

---

## Chemins des Checkpoints

| Fichier | Contenu |
|---|---|
| `checkpoints/agent_checkpoint.pth` | Checkpoint principal (25 000 épisodes). Contient `perception`, `world_model` et `cost`. |
| `checkpoints/agent_critic_nstep.pth` | Checkpoint avec le Critique ré-entraîné (N-Step TD). Même `perception` et `world_model`, mais `cost` mis à jour. |

Les checkpoints sont des dictionnaires PyTorch :

```python
checkpoint = torch.load("checkpoints/agent_checkpoint.pth")
# Clés disponibles : 'perception', 'world_model', 'cost'

perception.load_state_dict(checkpoint['perception'])
world_model.load_state_dict(checkpoint['world_model'])
cost.load_state_dict(checkpoint['cost'])
```
