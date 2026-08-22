# 🧠 Spécification Technique Détaillée : Architecture JEPA & Agent Autonome (World Model)

Ce document de référence est conçu pour vous accompagner dans l'implémentation, le débogage et l'extension de votre agent autonome basé sur l'architecture **JEPA (Joint-Embedding Predictive Architecture)** de Yann LeCun. Il fait le pont entre la théorie mathématique des papiers de recherche (LeCun 2022, I-JEPA, V-JEPA, LeJEPA/SIGReg 2025) et le code PyTorch de votre projet.

---

## 1. 🏛 Fondations Théoriques et Principes Clés

### 1.1. Intelligence Autonome et EBM (Energy-Based Models)
L'approche de LeCun s'éloigne des LLM génératifs (qui prédisent des pixels ou des tokens) au profit des **Modèles Basés sur l'Énergie (EBM)**. L'agent cherche constamment à minimiser une fonction d'énergie, qui correspond à la fois à "l'incohérence" de ses prédictions (lors de l'apprentissage) et au "coût" de ses actions (lors de la planification).

### 1.2. Mode-1 vs Mode-2 (Le double système)
- **Mode-1 (Réactif)** : C'est le réflexe. L'agent observe $x_t$, le convertit en état latent $s_t$, puis utilise une politique réactive (Policy Network) pour agir instantanément (non utilisé ou secondaire dans votre architecture de base).
- **Mode-2 (Planification délibérative)** : L'agent "s'arrête pour réfléchir". C'est là qu'interviennent l'**Acteur CEM** et le **World Model**. Il simule mentalement des scénarios, évalue leurs coûts, et exécute la meilleure action. C'est le cœur de votre code.

---

## 2. 🧩 Les 6 Modules Fondamentaux : Deep Dive Code & Tenseurs

### 2.1. L'Environnement (`env/gridworld.py`)
- **Rôle** : Moteur physique gérant la grille réelle, l'énergie (batterie de l'agent) et générant l'observation partielle (POV).
- **Entrées / Sorties PyTorch** :
  - **In** : Action discrète $a_t \in \{0, 1, 2, 3\}$ (Haut, Bas, Gauche, Droite).
  - **Out** : Vision $x_t$ Tensor `[Batch=1, Channels=1, H=5, W=5]`, Énergie $e_t$ (float), Reward $r_t$ (float), Done (bool).

### 2.2. Perception / Encodeur (`modules/perception.py`)
- **Rôle** : Réseau neuronal (CNN suivi de couches Linéaires) qui mappe les pixels bruts ($x_t$) vers un espace abstrait continu ($s_t$).
- **La dualité de l'Encodeur (Crucial pour JEPA)** :
  L'architecture JEPA exige **DEUX** encodeurs pour éviter l'effondrement (*collapse*) :
  1. **Encodeur Principal** (Paramètres $\theta$) : Entraîné par descente de gradient.
  2. **Encodeur Cible (Target Encoder)** (Paramètres $\bar{\theta}$) : Mis à jour par moyenne mobile exponentielle (EMA) : $\bar{\theta} = \tau\bar{\theta} + (1-\tau)\theta$ (avec $\tau \approx 0.99$).
- **Dimensions Tenseur** : 
  - **In** : `[B, 1, 5, 5]`
  - **Out** : $s_t$ Tensor `[B, 128]` (si `latent_dim = 128`).
- **Prévention du Collapse (`modules/sigreg.py`)** : 
  Dans les derniers papiers (LeJEPA, 2025), la méthode **SIGReg** (Sketched Isotropic Gaussian Regularization) ou **VICReg** (Variance-Invariance-Covariance) est ajoutée à la perte pour forcer l'espace latent à être structuré et informatif :
  $\mathcal{L}_{V} = \max(0, 1 - \text{Var}(s_t))$

### 2.3. Configurateur (`modules/configurator.py`)
- **Rôle** : Le cortex préfrontal. Fixe l'objectif de l'Acteur en fonction des signaux vitaux de l'environnement.
- **Logique Opérationnelle** :
  Si `current_energy < seuil_critique` :
      $s_{goal} \leftarrow \text{Encodeur}(x_{\text{recharge}})$
      Pondérations : $w_{energy} = 1.0, w_{goal} = 0.5$
  Sinon :
      $s_{goal} \leftarrow \text{Encodeur}(x_{\text{cible}})$
      Pondérations : $w_{energy} = 0.1, w_{goal} = 1.0$
- **Sorties** : $s_{goal}$ `[1, 128]`, et tenseur de poids $w$.

### 2.4. Le World Model (`modules/world_model.py`)
- **Rôle** : Réseau de neurones récurrent ou feed-forward (MLP/Transformer) qui prédit les conséquences d'une action, **strictement dans l'espace latent**.
- **Équation Principale** : $\hat{s}_{t+1} = \text{Pred}_\phi(s_t, z_t, a_t)$ où $z_t$ est la variable latente stochastique gérant l'incertitude (optionnelle en gridworld déterministe).
- **Perte d'Apprentissage (JEPA Loss)** : 
  La prédiction $\hat{s}_{t+1}$ est comparée au véritable état encodé par l'**Encodeur Cible** $s'_{t+1}$, **jamais** aux pixels.
  $\mathcal{L}_{WM} = \mathbb{E}[\Vert{}\hat{s}_{t+1} - s'_{t+1}\Vert{}_2^2]$

### 2.5. L'Acteur CEM (`modules/actor.py`)
- **Rôle** : Utilise le **Model Predictive Control (MPC)** couplé à la méthode **Cross-Entropy (CEM)** pour simuler et affiner un plan.
- **L'Algorithme CEM (Étape par Étape)** :
  1. Initialiser une distribution $P(a) = \text{Uniform}(0, 1)$ pour chaque pas de l'horizon $H$ (ex: $H=5$).
  2. Répéter $M$ itérations d'optimisation (ex: $M=3$) :
     - Échantillonner $N$ séquences aléatoires (ex: $N=100$) depuis $P(a)$. Shape: `[100, 5, 4]`.
     - Simuler chaque séquence via le World Model : `WM(WM(WM(s_t, a_1), a_2)...)`.
     - Évaluer le coût cumulé de chaque séquence via l'Évaluateur de Coût.
     - Trier les séquences par coût, isoler les $K$ séquences d'élite (ex: $K=10$).
     - Mettre à jour $P(a)$ en calculant la moyenne des actions dans les $K$ élites.
  3. Retourner la première action de la distribution $P(a)$ optimisée finale.

### 2.6. L'Évaluateur de Coût (`modules/cost.py`)
- **Rôle** : Combine la tâche imposée par le Configurateur et la "prudence" mathématique.
- **Composante 1 : Le Coût Intrinsèque (Boussole)** : 
  $c_{intr} = \Vert{} \hat{s}_{t+1} - s_{goal} \Vert{}_2$ (Distance Euclidienne).
- **Composante 2 : Le Critique (Value Function $V_{\psi}$)** :
  Réseau neuronal (MLP) prédisant la douleur/plaisir à long terme pour éviter les murs locaux.
  $c_{total} = w_{goal} \cdot c_{intr} + V_{\psi}(\hat{s}_{t+1})$
- **Apprentissage du Critique (TD-Learning)** : 
  $\mathcal{L}_{Critic} = (V_\psi(s_t) - (c_{intr\_reel} + \gamma V_\psi(s_{t+1})))^2$

---

## 3. 🔄 Dynamique Globale et Boucle d'Entraînement (`main.py`)

L'entraînement complet (hors ligne ou itératif) se déroule sur des lots (batchs) extraits de la **Mémoire Replay Buffer**.
Les optimiseurs PyTorch se coordonnent ainsi :

```python
# 1. Forward Pass Encodeur
s_t = encoder(x_t)
with torch.no_grad():
    s_next_target = target_encoder(x_next)  # Cible stable

# 2. Prédiction World Model
s_next_pred = world_model(s_t, a_t)

# 3. Calcul des Pertes (Losses)
loss_jepa = F.mse_loss(s_next_pred, s_next_target)
loss_sigreg = compute_sigreg_variance_loss(s_t) # Empêche s_t de finir à (0,0)

loss_total_enc_wm = loss_jepa + lambda_reg * loss_sigreg

# 4. Rétropropagation
opt_encoder.zero_grad()
opt_wm.zero_grad()
loss_total_enc_wm.backward()
opt_encoder.step()
opt_wm.step()

# 5. Mise à jour de l'Encodeur Cible (EMA)
for param, target_param in zip(encoder.parameters(), target_encoder.parameters()):
    target_param.data.mul_(tau).add_(param.data, alpha=1 - tau)