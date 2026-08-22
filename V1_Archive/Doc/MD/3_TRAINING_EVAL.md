# 🎓 Entraînement & Évaluation

> Ce document décrit les protocoles d'entraînement du modèle et les 3 métriques d'évaluation scientifique utilisées pour mesurer la véritable intelligence de l'agent.

---

## Stratégie d'Entraînement

### Phase 1 : Entraînement Joint (`main_colab.py`)

Le script principal entraîne simultanément la `Perception`, le `WorldModel` et le `Cost` sur des épisodes de GridWorld procédural.

**Boucle d'entraînement :**

1. L'agent explore l'environnement avec le planificateur `Actor` (CEM).
2. Chaque transition `(x_t, a_t, x_next, reward, done)` est stockée dans `ShortTermMemory`.
3. Quand le buffer contient assez de données, on extrait des séquences via `memory.sample_sequences(batch_size, seq_len=8)`.
4. **Perception + World Model** : L'encodeur transforme `x_0` en `s_0`, puis `world_model.forward_seq(s_0, a_seq)` prédit les `s_preds`. La loss est la MSE entre les prédictions et les cibles encodées par le Target Encoder (EMA).
5. **SIGReg** : La perte `SIGReg()(s_t)` est ajoutée pour régulariser l'espace latent.
6. **Cost (TD-Learning)** : `V(s_t)` est entraîné contre la cible `r_t + γ × V_target(s_{t+1})`, où `target_cost` est une copie stabilisée par EMA (τ = 0.005).

**Hyperparamètres d'entraînement :**

| Paramètre | Valeur |
|---|---|
| `latent_dim` | 32 |
| `hidden_dim` | 128 |
| `seq_len` | 8 |
| `batch_size` | 32 |
| `lr` (Perception + WM) | 3e-4 |
| `lr` (Cost) | 1e-4 |
| γ (discount) | 0.9 |
| τ (EMA Target Encoder) | 0.005 |
| τ (EMA Target Cost) | 0.005 |
| `ShortTermMemory.capacity` | 10 000 |

---

### Phase 2 : Entraînement Isolé du Critique (`train_critic_only.py`)

**Motivation :** Le diagnostic a révélé que le `Cost` entraîné en Phase 1 était trop faible pour évaluer correctement les impasses complexes. La Perception et le World Model fonctionnent parfaitement — il serait catastrophique de les réentraîner et risquer de corrompre l'espace latent.

**Principe : Le Gel (Freezing)**

```python
# Charger le meilleur checkpoint
checkpoint = torch.load("checkpoints/agent_checkpoint.pth")
perception.load_state_dict(checkpoint['perception'])
world_model.load_state_dict(checkpoint['world_model'])

# Geler les modèles stables
for param in perception.parameters():
    param.requires_grad = False
for param in world_model.parameters():
    param.requires_grad = False

# Seul le Cost est entraînable (poids aléatoires frais)
optimizer = torch.optim.Adam(cost.parameters(), lr=1e-3)
```

**Avantage :** La boucle d'entraînement est extrêmement rapide car PyTorch ne calcule plus les gradients à travers le CNN ni le RNN.

#### Le N-Step TD Learning

Au lieu de la mise à jour classique à 1 pas :

$$V(s_t) = r_t + \gamma \cdot V(s_{t+1})$$

On utilise la somme cumulée sur N=3 pas :

$$V(s_t) = -r_t + \gamma(-r_{t+1}) + \gamma^2(-r_{t+2}) + \gamma^3 \cdot V_{target}(s_{t+3})$$

> **Inversion du signe** : Le CEM minimise le coût. Les récompenses (+100 pour la cible) deviennent des coûts négatifs (-100), et les pénalités (-5 pour collision) deviennent des coûts positifs (+5).

**Stabilisation par Target Network :**

Un `target_cost` (copie du `Cost`) est mis à jour lentement par Polyak averaging :

```python
# Après chaque batch
for target_param, param in zip(target_cost.parameters(), cost.parameters()):
    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
```

**Politique d'exploration :** ε-greedy (ε = 0.5) combiné au CEM. 50% du temps, l'agent prend une action aléatoire pour tomber dans les pièges et alimenter le buffer en "erreurs riches".

**Environnement mixte :** La classe `MixedGridWorldEnv` dans `train_critic_only.py` génère :
- 25% de grilles avec piège en U
- 25% de grilles avec labyrinthe zig-zag
- 50% de grilles aléatoires classiques (In-Distribution)

**Sauvegarde :** `checkpoints/agent_critic_nstep.pth` (ne touche jamais au checkpoint de base).

---

## Protocoles d'Évaluation

### Test 1 : Linear Probing (`eval/eval_perception.py`)

**Objectif :** Vérifier que l'espace latent encode des informations géométriques exploitables.

**Protocole :**
1. Générer 5 000 grilles aléatoires via `GridWorldEnv`.
2. Encoder chaque grille avec `perception(obs)` pour obtenir `s_t` de dimension 32.
3. Entraîner une **Régression Linéaire** (scikit-learn `LinearRegression`) pour prédire les coordonnées `(row, col)` de l'agent à partir de `s_t`.
4. Mesurer le score **R²** sur un jeu de test.

**Interprétation :**
- R² ≈ 1.0 → L'espace latent encode les positions de manière linéairement séparable.
- R² ≈ 0.2 → L'information est présente mais de manière non-linéaire (normal pour SIGReg).
- R² ≈ 0.0 → Latent Collapse (les vecteurs sont tous identiques).

**Résultat observé :** R² = 0.22 — L'espace est fonctionnel mais non-linéaire.

---

### Test 2 : Rollout Drift (`eval/eval_dynamics.py`)

**Objectif :** Mesurer la stabilité des prédictions du World Model sur le long terme.

**Protocole :**
1. Collecter 500 transitions réelles dans l'environnement.
2. À chaque transition, encoder l'état réel avec `perception(x_t)`.
3. En parallèle, faire avancer le World Model "à l'aveugle" via `world_model.forward_step(s_imag, a_t, h_t)`.
4. Mesurer la MSE entre le vecteur imaginé `s_imag` et le vecteur réel `s_real` à chaque pas.
5. Regrouper les erreurs par profondeur temporelle (t+1, t+5, t+10).

**Interprétation :**
- Erreur < 0.5 à t+10 → Excellent. Le World Model est fiable pour le CEM.
- Erreur > 2.0 à t+5 → Le modèle hallucine. Les plans du CEM seront incohérents.

**Résultat observé :** 0.10 → 0.33 → 0.49 — Le World Model est très performant.

---

### Test 3 : Généralisation OOD (`eval/eval_behavior.py`)

**Objectif :** Tester la capacité de l'agent à résoudre des environnements jamais rencontrés pendant l'entraînement.

**Protocole :**
1. **In-Distribution (100 épisodes)** : Grilles aléatoires générées par `GridWorldEnv(size=10)`.
2. **OOD Piège en U (20 épisodes)** : Grille fixe avec un couloir en U entre l'agent et la cible. Pour réussir, l'agent doit d'abord s'éloigner de la cible.
3. **OOD Labyrinthe Zig-Zag (20 épisodes)** : Grille avec des murs horizontaux imposant un parcours en zigzag.

**Métriques :**
- **Taux de succès** : Pourcentage d'épisodes où l'agent atteint `target_pos`.
- **Pas moyens** : Nombre moyen de pas avant `done` (100 = échec par épuisement).

**Résultat observé :** ID ≈ 50% / OOD = 0% — Le CEM est efficace en espace ouvert mais bloqué par les minimas locaux des labyrinthes.
