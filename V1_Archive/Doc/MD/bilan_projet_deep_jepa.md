# 📊 Bilan Technique Avancé : Architecture "Deep-JEPA" (V3)

Ce document détaille l'implémentation technique, les choix architecturaux et la formulation mathématique du World Model actuel, inspiré des architectures de type JEPA (Joint-Embedding Predictive Architecture) et Dreamer.

---

## 🏗️ 1. Architecture des Réseaux de Neurones

### A. Encodeur Perceptif (ResNet) : $\mathcal{E}_\theta$
L'encodeur a pour rôle de projeter l'observation spatiale $x_t \in \mathbb{R}^{4 \times 10 \times 10}$ vers un espace latent compact et isotrope $s_t \in \mathbb{R}^{32}$.
- **Topologie** : Remplacement du CNN linéaire par un mini-ResNet. L'architecture utilise des `ResidualBlock` (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> Add -> ReLU).
- **Avantage** : Les connexions résiduelles (Skip-Connections) préservent le gradient lors de la rétropropagation spatiale, permettant d'extraire des champs récepteurs beaucoup plus larges, indispensables pour comprendre les macro-structures (ex: murs étendus).

### B. Dynamique Latente Temporelle (World Model) : $\mathcal{D}_\phi$
- **Ancienne approche (Markovienne stricte)** : $s_{t+1} = \mathcal{D}(s_t, a_t)$. Aucune mémoire, sensibilité extrême à l'aliasing partiel.
- **Nouvelle approche (RNN - BPTT)** : Implémentation d'une dynamique récurrente basée sur un Gated Recurrent Unit (GRU).
  - L'état caché $h_t \in \mathbb{R}^{128}$ capture l'historique complet de la trajectoire.
  - Équation de mise à jour : $h_t = \text{GRUCell}(x=\text{Proj}(s_{t-1}, a_{t-1}), h_{t-1})$
  - Prédiction de l'état suivant : $\hat{s}_t = s_{t-1} + \text{MLP}(h_t)$
- **Avantage** : La mémoire $h_t$ résout l'amnésie de l'agent. Si l'agent heurte un mur, la collision est encodée dans $h_t$, modifiant radicalement les prédictions futures $\hat{s}_{t+k}$ pour éviter de répéter l'action.

### C. Encodeur Cible (Target EMA) : $\mathcal{E}_{\theta'}$
- Pour éviter l'effondrement trivial de la représentation (collapse), un encodeur cible est utilisé pour générer les labels $s_{t+1}$ lors de l'entraînement. 
- Les poids $\theta'$ sont une moyenne mobile exponentielle (EMA) des poids $\theta$ : $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$.

---

## 🧮 2. Fonctions de Perte (Losses) & Optimisation

L'entraînement repose sur l'optimisation conjointe de plusieurs objectifs via *Backpropagation Through Time* (BPTT).

### A. JEPA Loss (Prediction Loss sur Séquences)
Le Buffer de Replay extrait des séquences contiguës de longueur $T=8$. Le World Model est déroulé (unrolled) sur ces 8 pas.
$$ \mathcal{L}_{pred} = \frac{1}{T} \sum_{k=1}^{T} || \hat{s}_{t+k} - \mathcal{E}_{\theta'}(x_{t+k}) ||_2^2 $$

### B. SIGReg (Sketch Isotropic Gaussian Regularizer)
Pour forcer l'espace latent à exploiter l'ensemble de ses 32 dimensions (covariance unitaire) et éviter qu'il ne s'écrase sur un sous-espace de faible dimension, on applique la régularisation SIGReg sur l'état initial de la séquence :
$$ \mathcal{L}_{sigreg} = \text{SIGReg}(\mathcal{E}_\theta(x_t)) $$
Le Loss total du modèle du monde est donc : $\mathcal{L}_{WM} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{sigreg}$ (avec $\lambda = 1.0$).

### C. Critic Loss (Apprentissage par TD-Learning)
La fonction de valeur $V(s) \in \mathbb{R}$ est entraînée pour prédire le coût (distance) à long terme via l'équation de Bellman (TD-Error).
$$ \mathcal{L}_{critic} = || V(s_t) - (c_t + \gamma V(s_{t+1})) ||_2^2 $$

---

## 🧠 3. Planificateur de Trajectoire (Deep CEM)

L'acteur utilise la méthode de Cross-Entropy (CEM) opérant directement dans l'espace latent pour générer une politique d'action optimale.

- **Hyperparamètres du Planner** :
  - **Horizon ($H$)** : 15 pas dans le futur (contre 8 auparavant).
  - **Taille de l'échantillon ($N$)** : 2000 séquences candidates (contre 200). $4^{15}$ combinaisons nécessitent un échantillonnage massif.
  - **Itérations CEM ($M$)** : 10 itérations de raffinement.
  - **Taille de l'élite ($K$)** : 100 meilleures séquences conservées pour mettre à jour la distribution de probabilité d'action.

- **Dynamique Inférentielle** : À chaque étape d'environnement, le CEM clone l'état caché $h_t$ du GRU, déroule massivement 2000 trajectoires virtuelles dans l'espace latent sur $H=15$, et évalue le coût de chaque trajectoire via le Critique $V(s)$.

---

## 📈 4. État des Lieux de l'Entraînement

- **Checkpoints Actuels** : L'agent a dépassé l'épisode `16 450` sur 25 000 (GPU Cloud).
- **Curriculum Learning** : 
  - La Phase 1 d'exploration massive (random action decay $\epsilon : 1.0 \rightarrow 0.1$) s'est achevée à l'épisode 15 000. 
  - La Phase 2 (actuelle) exploite l'agent avec $\epsilon = 0.05$ et lance le lourd processus CEM à chaque pas pour peupler le buffer de transitions expertes.
- **Métriques** : Le module dynamique (GRU) affiche désormais un `delta_norm` moyen de $3.0$ pour ses inférences d'actions (contre $0.1$ sur les anciennes architectures sans mémoire), prouvant que la géométrie latente est mathématiquement corrélée à la transition spatiale.
