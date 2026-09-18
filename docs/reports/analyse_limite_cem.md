# 🔍 Analyse Honnête — Pourquoi le Critique ne débloquera pas les OOD

## Historique des tentatives

| Tentative | ID | U-Trap | ZigZag | Conclusion |
|-----------|-----|--------|--------|------------|
| Baseline (boussole seule) | **49%** | 0% | 0% | Référence |
| Critique N-Step TD (7h) | 42% | 0% | 0% | Loss=46.7, pas convergé |
| Critique N-Step + accumulation (w=1.0) | 8% | 0% | 0% | Critique noie la boussole |
| Critique N-Step + accumulation (w=0.1) | 26% | 5% | 0% | Signal faible |
| Critique Monte Carlo (7h) | 38% | 0% | 0% | Discrimination 5/10 = hasard |

> [!CAUTION]
> **Aucune méthode d'entraînement du Critique n'a amélioré les résultats.** Le problème n'est pas l'algorithme d'entraînement — c'est une limitation structurelle de l'architecture.

---

## La vraie raison : le CEM ne voit pas assez loin

Le problème fondamental n'est **pas** le Critique. C'est le **horizon du planificateur CEM**.

### Le Piège en U — Anatomie du problème

```
Agent (5,3)                   Cible (5,6)
    ·                            ★
    ·   ████████  ← mur haut
    ·   █      █
    ·   █      █  ← murs verticaux
    ·   █      █
    ·   ████████  ← mur bas
```

Pour contourner le U, l'agent doit :
1. Monter jusqu'à la rangée 2 (3 pas)
2. Traverser vers la droite, colonnes 5→8 (3 pas)
3. Descendre vers la rangée 5 (3 pas)  
4. Revenir vers la cible en colonne 6 (2 pas)

**Total : ~15-20 pas de détour**, dont les 5 premiers **éloignent** l'agent de la cible.

### Ce que le CEM voit avec horizon=10

Le CEM échantillonne 500 trajectoires de **10 pas**. Après 10 pas :
- Trajectoire "directe" : l'agent se cogne au mur mais `dist_to_goal` est faible → **élu comme "meilleure"**
- Trajectoire "détour" : l'agent monte et contourne mais `dist_to_goal` est élevée → **rejetée**

Même avec un **Critique parfait** qui dit "le mur est dangereux", le CEM ne peut pas voir que les 10 prochains pas (monter → traverser) mèneront à la cible à `t+15`. Il est **structurellement aveugle** au-delà de son horizon.

---

## Pourquoi le Critique ne peut pas compenser

Le Critique `V(s_t)` est un scalaire par état. Pour compenser la myopie du CEM, il devrait encoder :

> "Cet état est à 20 pas de la cible via un détour, mais seulement 3 pas en ligne droite (bloquée)"

C'est un problème de **crédit temporel à longue portée** — exactement ce que le TD-Learning est censé résoudre. Mais :

1. **Le Critique est trop petit** (32→32→1 = 1089 paramètres) pour encoder un paysage de valeur complexe sur un espace latent 32D non-linéaire
2. **Chaque grille est différente** — le Critique doit généraliser à des configurations d'obstacles jamais vues
3. **Le taux de succès exploration est de 28%** — dans 72% des épisodes, le Critique n'apprend rien d'utile car l'agent n'atteint jamais la cible

---

## 3 Pistes pour avancer

### Piste A — Augmenter le horizon CEM (simple, lent)
```python
actor = Actor(horizon=25, num_sequences=1000, cem_iterations=15)
```
**Pro :** Le CEM pourrait "voir" le contournement du U en 25 pas.  
**Con :** ~6x plus lent à l'inférence (déjà ~5s/pas → ~30s/pas).

### Piste B — Planification hiérarchique (complexe, élégant)
Deux niveaux de planification :
1. **Macro-planificateur** : choisit des sous-objectifs (waypoints) dans l'espace latent
2. **Micro-planificateur** (CEM actuel) : atteint chaque waypoint en 10 pas

Le Critique servirait à évaluer les waypoints, pas les trajectoires individuelles.

### Piste C — Politique apprise (RL classique)
Remplacer le CEM par un réseau de politique entraîné par PPO/SAC qui apprend directement `π(a|s)`. Le World Model servirait de simulateur pour le Dyna-RL (model-based RL).

**Pro :** Les politiques apprises n'ont pas de limitation d'horizon.  
**Con :** Abandon partiel de l'architecture JEPA pure de LeCun.

---

## Recommandation

> [!IMPORTANT]
> **Piste A** est la plus rapide à tester (1 ligne de code). Si `horizon=25` résout le U-Trap, ça prouve que le problème est bien l'horizon et pas le Critique. C'est le test le plus informatif à faire maintenant.

On pourra ensuite décider si l'on investit dans B ou C selon vos objectifs.
