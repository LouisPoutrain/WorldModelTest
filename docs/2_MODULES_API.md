# 📦 API des Modules (`modules/`)

> Référence technique pour développeurs. Ce document détaille la signature, les entrées/sorties et le fonctionnement interne de chaque composant Python du dossier `modules/`.

---

## `perception.py` — Encodeur CNN

### Classe `ResidualBlock(nn.Module)`

Bloc résiduel standard avec skip-connection.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        # Conv2d(channels, channels, 3x3, padding=1) → ReLU → Conv2d → skip-connection
    
    def forward(self, x):
        # x: (B, C, H, W) → (B, C, H, W)
        return F.relu(x + self.block(x))
```

### Classe `Perception(nn.Module)`

Compresse une observation brute `(B, 4, 10, 10)` en vecteur latent `(B, 32)`.

```python
class Perception(nn.Module):
    def __init__(self, in_channels=4, latent_dim=32):
        # Architecture : Conv2d(4→32) → 2×ResidualBlock(32) → Conv2d(32→64, stride=2) 
        #              → 2×ResidualBlock(64) → Flatten → Linear(64*5*5, 128) → Linear(128, 32)

    def forward(self, x):
        """
        x: (B, 4, 10, 10) — Observation de la grille (4 canaux : obstacles, cible, station, agent)
        
        Returns:
            s_t: (B, latent_dim) — Vecteur latent compressé
        """
```

**Notes :**
- Si `x` est de dimension 3 (pas de batch), `unsqueeze(0)` est appliqué automatiquement.
- Le `stride=2` du second `Conv2d` réduit la résolution spatiale de 10×10 à 5×5.
- Pas de `BatchNorm` finale : la régularisation est déléguée à `SIGReg`.

---

## `world_model.py` — Prédicteur Temporel (RNN)

### Classe `WorldModel(nn.Module)`

Prédit l'état latent futur en utilisant un `GRUCell` comme mémoire temporelle.

```python
class WorldModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=128):
        # input_proj: Linear(latent_dim + action_dim, hidden_dim) + ReLU
        # rnn_cell:   GRUCell(hidden_dim, hidden_dim)
        # predictor:  Linear(128→128) → ReLU → Linear(128→128) → ReLU → Linear(128→32)
```

#### `init_hidden(batch_size, device) → h_0`
Retourne un état caché initial rempli de zéros : `torch.zeros(batch_size, hidden_dim)`.

#### `forward_step(s_t, a_t, h_t) → (s_next, h_next)`
Effectue un pas de prédiction temporelle (inférence / planification).

```python
def forward_step(self, s_t, a_t, h_t):
    """
    s_t:  (B, latent_dim)  — État latent courant
    a_t:  (B, action_dim)  — Action one-hot (4 classes)
    h_t:  (B, hidden_dim)  — État caché du GRU

    Returns:
        s_next: (B, latent_dim)  — État prédit (s_t + delta)
        h_next: (B, hidden_dim)  — Nouvel état caché
    """
    x = torch.cat([s_t, a_t], dim=-1)     # (B, 36)
    x_proj = self.input_proj(x)            # (B, 128)
    h_next = self.rnn_cell(x_proj, h_t)    # (B, 128)
    delta = self.predictor(h_next)         # (B, 32)
    s_next = s_t + delta                   # Prédiction résiduelle
    return s_next, h_next
```

#### `forward_seq(s_0, a_seq, h_0=None) → (s_preds, h_seq)`
Traite une séquence complète de T pas (entraînement BPTT).

```python
def forward_seq(self, s_0, a_seq, h_0=None):
    """
    s_0:   (B, latent_dim)        — État initial
    a_seq: (B, T, action_dim)     — Séquence d'actions one-hot
    h_0:   (B, hidden_dim) | None — État caché initial (zéros si None)

    Returns:
        s_preds: (B, T, latent_dim) — Prédictions pour t=1 à T
        h_seq:   (B, T, hidden_dim) — États cachés pour t=1 à T
    """
```

> **Mode autorégressif :** chaque prédiction `s_next` est réinjectée comme entrée du pas suivant, forçant le modèle à apprendre une dynamique robuste sur le long terme.

---

## `actor.py` — Planificateur CEM

### Classe `Actor`

> **Attention :** `Actor` n'hérite PAS de `nn.Module`. C'est un algorithme de planification pur, sans paramètres apprenables.

```python
class Actor:
    def __init__(self, action_dim=4, num_sequences=200, horizon=12, 
                 cem_iterations=5, elite_size=30, gamma=0.9):
        self.action_dim = action_dim
        self.N = num_sequences       # Nombre de trajectoires échantillonnées
        self.H = horizon             # Profondeur de simulation (nombre de pas)
        self.M = cem_iterations      # Itérations de raffinement CEM
        self.K = elite_size          # Nombre de séquences "élites" retenues
        self.gamma = gamma
```

#### `plan(s_t, h_t, world_model, cost_module, s_goal, w_goal=1.0) → (action, sequence, cost)`

```python
def plan(self, s_t, h_t, world_model, cost_module, s_goal, w_goal=1.0):
    """
    s_t:         (1, latent_dim)  — État latent actuel
    h_t:         (1, hidden_dim)  — État caché RNN actuel
    world_model: WorldModel       — Le simulateur interne
    cost_module: Cost             — Le réseau Critique
    s_goal:      (1, latent_dim)  — Cible dans l'espace latent
    w_goal:      float            — Poids de la boussole (distance au goal)

    Returns:
        action:   int              — La première action de la meilleure séquence
        sequence: Tensor (H,)      — La séquence d'actions complète
        cost:     float            — Le coût de la meilleure séquence
    """
```

**Calcul du coût :**
```python
# Boussole : distance MSE entre l'état simulé final et le goal
dist_to_goal = F.mse_loss(s_sim, s_goal.expand(N, -1), reduction='none').sum(dim=1)

# Critique : évaluation apprise de la qualité de l'état
c_critic = cost_module(s_sim).squeeze(-1)

# Combinaison
total_cost = (w_goal * dist_to_goal) + c_critic
```

**Lissage de Laplace** (pour éviter que les probabilités convergent vers 0) :
```python
new_probs[t] = (counts + 0.1) / (self.K + 0.1 * self.action_dim)
```

---

## `cost.py` — Réseau Critique

### Classe `Cost(nn.Module)`

```python
class Cost(nn.Module):
    def __init__(self, latent_dim=32):
        # critic: Linear(32, 32) → ReLU → Linear(32, 1)

    def forward(self, s_t, *args, **kwargs):
        """
        s_t: (B, latent_dim) — État(s) latent(s) à évaluer

        Returns:
            V(s_t): (B, 1) — Coût futur attendu (plus c'est bas, mieux c'est)
        """
        return self.critic(s_t)
```

---

## `configurator.py` — Gestionnaire de Buts

### Classe `Configurator`

> **Attention :** `Configurator` n'hérite PAS de `nn.Module`. C'est un automate logique pur.

```python
class Configurator:
    def __init__(self, latent_dim=32):
        self.s_target = -torch.ones(1, latent_dim)   # Placeholder
        self.s_station = -torch.ones(1, latent_dim)   # Placeholder

    def set_goals(self, s_target, s_station):
        """Enregistre les vecteurs latents de la cible et de la station."""

    def get_configuration(self, energy):
        """
        energy: int — Énergie restante de l'agent

        Returns:
            s_goal:      (1, latent_dim) — Vecteur latent cible
            w_energy:    float           — Poids du coût énergétique
            w_collision: float           — Poids du coût de collision (toujours 1.0)
            w_goal:      float           — Poids de la distance au goal
        """
```

---

## `memory.py` — Replay Buffer Épisodique

### Classe `ShortTermMemory`

```python
class ShortTermMemory:
    def __init__(self, capacity=10000):
        self.buffer = []       # Buffer circulaire
        self.position = 0      # Curseur d'écriture

    def push(self, x_t, a_t, x_next, reward, done):
        """Stocke une transition (x_t, a_t, x_next, reward, done) sur CPU."""

    def sample_transitions(self, batch_size):
        """
        Échantillonnage classique de transitions individuelles.
        
        Returns:
            x_batch:      (B, C, H, W)
            a_batch:      (B,)
            x_next_batch: (B, C, H, W)
            reward_batch: (B, 1)
            done_batch:   (B, 1)
        """

    def sample_sequences(self, batch_size, seq_len=8):
        """
        Extraction de séquences temporelles contiguës pour BPTT et N-Step TD.
        Vérifie que les séquences ne traversent pas un `done=True` ni le curseur de wrap.
        
        Returns:
            x_0_batch:         (B, C, H, W)      — Observation initiale
            a_seq_batch:       (B, T)             — Actions t=0..T-1
            x_next_seq_batch:  (B, T, C, H, W)   — Observations t=1..T
            reward_seq_batch:  (B, T)             — Récompenses
            done_seq_batch:    (B, T)             — Flags de fin d'épisode
        
        Raises:
            ValueError: Si le buffer ne contient pas assez de séquences valides.
        """
```

---

## `sigreg.py` — Régulariseur SIGReg

### Classe `SIGReg(nn.Module)`

```python
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        # t:       linspace(0, 3, 17)     — Points d'évaluation
        # phi:     exp(-t²/2)             — Fonction caractéristique gaussienne
        # weights: coefficients de quadrature trapézoïdale

    def forward(self, proj):
        """
        proj: (B, D) ou (T, B, D) — Vecteurs latents à régulariser

        Returns:
            loss: scalar — Statistique d'Epps-Pulley (à minimiser)
        """
```

> Source : [le-wm](https://github.com/lucas-maes/le-wm.git)
