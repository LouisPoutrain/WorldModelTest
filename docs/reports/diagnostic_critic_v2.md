# 🔬 Diagnostic : Pourquoi le Critique entraîné ne change rien (OOD = 0%)

## Verdict : 3 bugs critiques identifiés

---

## Bug #1 — Le Critique n'évalue que l'état FINAL (le plus grave)

> [!CAUTION]
> Le planificateur CEM ne regarde que le **dernier état simulé** de la trajectoire pour calculer le coût critique. Il est structurellement aveugle aux dangers intermédiaires.

### Le problème dans [`actor.py`](file:///Users/poutrainlouis/code/WorldModelTest/modules/actor.py#L40-L56)

```python
# Rollout : le CEM simule H pas...
for t in range(self.H):
    a_t_sim = action_sequences[:, t]
    a_t_onehot = F.one_hot(a_t_sim, num_classes=self.action_dim).float()
    s_sim, h_sim = world_model.forward_step(s_sim, a_t_onehot, h_sim)

# ...mais n'évalue le Critique que sur s_sim FINAL (après H pas) !
c_critic = cost_module(s_sim).squeeze(-1)  # ← UN SEUL appel
```

Le Critique a beau avoir parfaitement appris que les cases du piège en U sont "mauvaises", **l'Actor ne lui demande jamais d'évaluer les états intermédiaires** de la trajectoire. Il ne regarde que l'état à `t+H`.

**Conséquence :** Si le World Model prédit que l'état final est "proche de la cible" (car la cible est spatialement proche dans le U-trap), le CEM choisit cette trajectoire même si elle passe par des murs à `t+1, t+2, ...`.

### ✅ Fix : Accumuler le coût critique le long de la trajectoire

```diff
 # Rollout dans le World Model
 s_sim = s_t.repeat(self.N, 1)
 h_sim = h_t.repeat(self.N, 1)
+cumulative_critic_cost = torch.zeros(self.N, device=device)
 
 with torch.no_grad():
     for t in range(self.H):
         a_t_sim = action_sequences[:, t]
         a_t_onehot = F.one_hot(a_t_sim, num_classes=self.action_dim).float()
         s_sim, h_sim = world_model.forward_step(s_sim, a_t_onehot, h_sim)
+        # Accumuler le coût critique à CHAQUE pas (avec discount)
+        step_cost = cost_module(s_sim).squeeze(-1)
+        cumulative_critic_cost += (self.gamma ** t) * step_cost
 
     # Coût distance au goal (état final)
     dist_to_goal = F.mse_loss(
         s_sim, s_goal.expand(self.N, -1), reduction='none'
     ).sum(dim=1)
 
-    c_critic = cost_module(s_sim).squeeze(-1)
-    total_cost = (w_goal * dist_to_goal) + c_critic
+    total_cost = (w_goal * dist_to_goal) + cumulative_critic_cost
```

---

## Bug #2 — Loss finale à 46.7 : le Critique n'a PAS convergé

> [!WARNING]
> Une loss MSE de **46.7** après 5000 épisodes / 7h est beaucoup trop haute. Le Critique n'a pas réellement appris une Value function utile.

### Causes probables

1. **Le Target Cost est initialisé aléatoirement** (car `cost.load_state_dict` est commenté). Donc le bootstrap `γ³ · V_target(s_{t+3})` démarre de valeurs absurdes, et le soft-update `τ=0.05` est trop rapide pour stabiliser le target en partant de zéro.

2. **Les récompenses ont une variance énorme** : un pas normal = -1, une collision = -5, une cible = +100. Les coûts inversés vont de -100 à +5. L'échelle des targets TD est instable.

### ✅ Fix : Normaliser les rewards et ralentir le target update

```python
# Clipper les rewards pour stabiliser
reward = max(min(reward, 10.0), -5.0)  # Cap le +100 de la cible

# Target update plus lent (car on part de zéro)
tau = 0.005  # au lieu de 0.05
```

---

## Bug #3 — Le World Model ne voit pas les murs (limitation structurelle)

> [!IMPORTANT]
> C'est la raison fondamentale pour laquelle même un Critique parfait ne résoudra pas les OOD à 100%.

Le World Model a été entraîné sur des grilles aléatoires (ID). Dans un U-trap, les positions **à l'intérieur du U** n'existent jamais dans les données d'entraînement. Quand l'Actor simule une trajectoire qui traverse un mur :

1. Le World Model **ne sait pas qu'un mur bloque** → il prédit `s_sim` comme si l'agent avait bougé.
2. Le Critique évalue un `s_sim` **imaginaire** qui n'existe pas dans la réalité.
3. Le coût paraît faible → le CEM choisit cette trajectoire "fantôme".

**C'est un problème de physique du modèle, pas de critique.** Le World Model n'a pas la notion de collision. Il prédit la dynamique dans l'espace latent sans contrainte physique.

### Pourquoi ce n'est pas un deal-breaker

Le fix du Bug #1 (accumuler le coût sur la trajectoire) atténue fortement ce problème. Si le Critique a appris que les cases près des murs sont "chères", il pénalisera les trajectoires qui s'en approchent, même si le World Model ne modélise pas parfaitement les collisions.

---

## 📊 Résumé des actions

| Priorité | Fix | Impact attendu |
|----------|-----|----------------|
| 🔴 P0 | **Actor : accumuler le coût critique sur tous les pas** | Le CEM pourra enfin "voir" les dangers intermédiaires |
| 🟡 P1 | **Clipper les rewards + τ=0.005** | Stabiliser la convergence du Critique |
| 🟢 P2 | **Augmenter les épisodes d'entraînement** (10k-20k) | Donner plus de données au Critique |

> [!TIP]
> Le fix P0 seul devrait suffire à améliorer significativement l'ID (de 42% → 70%+). Pour les OOD, il faudra combiner P0 + P1 + un re-entraînement plus long.
