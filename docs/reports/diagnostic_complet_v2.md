# 🔬 Rapport de Diagnostic Complet : Architecture V2 (JEPA)

Ce rapport compile les résultats de tous les scripts d'évaluation de la V2 (`eval_perception.py`, `eval_dynamics.py`, `eval_behavior.py`, `plot_eval.py`).

## 1. Perception (L'encodeur CNN)
- **Outil** : `eval_perception.py` (Linear Probing)
- **Score R²** : **0.155** (0 = aléatoire, 1 = parfait)
- **Verdict** : ❌ **ÉCHEC (Collapse partiel / Non-linéarité extrême)**
- **Analyse** : L'espace latent de 32 dimensions généré par le JEPA ne contient **pas** linéairement les coordonnées de l'agent et de la cible. L'architecture encode bien l'image, mais de manière totalement intriquée. On ne peut pas simplement tracer une droite pour extraire la position `(x, y)`.

## 2. World Model (La Physique Latente)
- **Outil** : `eval_dynamics.py` (Rollout Drift)
- **Erreur à t+1** : 0.103
- **Erreur à t+10** : 0.545
- **Verdict** : ✅ **SUCCÈS (Physique apprise)**
- **Analyse** : Malgré le fait que l'espace latent soit illisible pour nous (voir point 1), le World Model (GRU) **le comprend parfaitement**. Il est capable de prédire l'impact d'une action dans ce sous-espace avec une très grande précision. L'erreur dérive lentement, ce qui est normal, mais reste exploitable sur un horizon de 10 pas.

## 3. Le Critique (Apprentissage par Renforcement TD)
- **Outil** : `train_critic_only.py` & `plot_eval.py`
- **Coût prédit moyen** : ~156.0 (Constant partout)
- **Verdict** : ❌ **ÉCHEC TOTAL**
- **Analyse** : Le réseau `Cost` (32 -> 32 -> 1) est beaucoup trop petit. On lui a demandé de regarder un espace latent complètement intriqué (R² = 0.15) et d'en déduire mentalement un algorithme de contournement de murs (Geodesic Distance). Face à la difficulté et au bruit du TD-Learning, le réseau a paniqué et appris à prédire une valeur moyenne constante (~156) pour tous les états.

## 4. Le Planificateur (Cross-Entropy Method)
- **Outil** : `eval_behavior.py` (Comportement final)
- **Succès ID** : 26% (avec Critique) / 60% (Boussole pure)
- **Succès U-Trap** : **0%**
- **Succès ZigZag** : **0%**
- **Verdict** : ❌ **ÉCHEC DE GÉNÉRALISATION**
- **Analyse** : 
  - **Avec la Boussole Pure** : Le CEM utilise la distance Euclidienne. Il fonce en ligne droite et s'empale au fond du U-Trap (Minimum local).
  - **Avec le Critique** : Puisque le Critique prédit 156.0 pour 100% des trajectoires imaginées, le CEM ne voit aucune différence entre aller dans un mur ou avancer vers la sortie. Il se déplace donc au hasard complet (Mouvement Brownien).

---

## 🎯 Conclusion & Solution (Phase 3)

La fondation de la V2 (World Model) est excellente, mais la couche de décision (Critique) est dysfonctionnelle car le Reinforcement Learning est trop bruité pour apprendre des géométries complexes sur un espace compressé.

**La solution finale (Supervised Geodesic Critic)** :
Puisque nous sommes dans un environnement connu, nous allons abandonner le Reinforcement Learning instable. Nous allons :
1. Calculer mathématiquement le vrai "plus court chemin" (avec un algorithme BFS) pour nos 10 000 labyrinthes.
2. Agrandir le réseau du Critique (ex: `32 -> 128 -> 128 -> 1`).
3. Forcer le Critique à apprendre par cœur cette vraie distance par Apprentissage Supervisé.
4. Fournir cette boussole "parfaite" au Planificateur.
