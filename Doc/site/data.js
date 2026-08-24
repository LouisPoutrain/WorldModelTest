const modulesData = {
  'ENV': {
    deep: `<ul>
      <li><strong>Moteur Physique :</strong> Gère la grille réelle, l'énergie (batterie) et génère l'observation partielle (POV).</li>
      <li><strong>In (PyTorch) :</strong> Action discrète a_t ∈ {0, 1, 2, 3} (Haut, Bas, Gauche, Droite).</li>
      <li><strong>Out (PyTorch) :</strong> Vision x_t Tensor <code>[Batch=1, Channels=1, H=5, W=5]</code>, Énergie e_t (float), Reward r_t (float), Done (bool).</li>
    </ul>`,

    icon: "",
    title: "Environnement",
    subtitle: "Fichier : env/gridworld.py",
    simple: "C'est le monde physique (le labyrinthe). L'agent s'y déplace, se cogne contre les murs, et consomme de l'énergie. L'environnement lui donne juste ce qu'il a devant lui (vision 5x5).",
    tech: "Grille 2D générée de manière procédurale. Utilise un algorithme BFS (Breadth-First Search) pour garantir mathématiquement qu'il y a toujours un chemin possible vers la cible et la station.",
    input: "Action choisie a_t",
    output: "Image 5x5 (torch.Tensor)"
  },
    'PERC': {
    deep: `<ul>
      <li><strong>Dualité de l'Encodeur :</strong> JEPA exige DEUX encodeurs pour éviter l'effondrement (collapse). L'Encodeur Principal ($\theta$) est entraîné par gradient. L'Encodeur Cible ($\bar{\theta}$) est mis à jour par EMA : $\bar{\theta} = \tau\bar{\theta} + (1-\tau)\theta$.</li>
      <li><strong>Dimensions Tenseur :</strong> In = <code>[B, 1, 5, 5]</code> Out = $s_t$ Tensor <code>[B, 128]</code>.</li>
      <li><strong>Prévention du Collapse :</strong> Utilise SIGReg/VICReg pour forcer l'espace latent à être structuré. $\mathcal{L}_{V} = \max(0, 1 - \text{Var}(s_t))$.</li>
    </ul>`,

    math: "L_{SIGReg} = \\frac{1}{d} \\sum_{i} (\\text{Var}(z_i) - 1)^2 + \\lambda \\sum_{i \\neq j} \\text{Cov}(z_i, z_j)^2",
    icon: "",
    title: "Perception",
    subtitle: "Fichier : modules/perception.py",
    simple: "L'œil de l'agent. Il regarde l'image brute (les pixels) et la compresse en une 'idée mathématique' (un vecteur de 32 nombres). C'est beaucoup plus facile de réfléchir avec des idées qu'avec des pixels.",
    tech: "Petit réseau de neurones convolutif (ResNet). Il n'est pas entraîné par erreur classique mais via une régularisation appelée SIGReg, qui force ses 32 neurones à être indépendants et informatifs.",
    input: "Vision 5x5",
    output: "s_t (Vecteur de taille 32)"
  },
  'CONF': {
    deep: `<ul>
      <li><strong>Logique Opérationnelle :</strong> Cortex préfrontal agissant via des seuils.</li>
      <li>Si <code>energy < seuil</code> : $s_{goal} \leftarrow \text{Enc}(x_{\text{recharge}})$, $w_{energy} = 1.0, w_{goal} = 0.5$.</li>
      <li>Sinon : $s_{goal} \leftarrow \text{Enc}(x_{\text{cible}})$, $w_{energy} = 0.1, w_{goal} = 1.0$.</li>
      <li><strong>Sorties :</strong> $s_{goal}$ <code>[1, 128]</code>, et tenseur de poids $w$.</li>
    </ul>`,

    icon: "",
    title: "Configurateur",
    subtitle: "Fichier : modules/configurator.py",
    simple: "C'est l'instinct de survie. Il surveille la jauge d'énergie. Si elle est haute, il dit à l'agent de chercher la cible. Si elle est basse, il annule la mission et le force à chercher une station de recharge.",
    tech: "Règle algorithmique basée sur des seuils. Change dynamiquement la cible virtuelle s_goal que l'Acteur essaiera d'atteindre pendant sa planification.",
    input: "Énergie actuelle",
    output: "Cible (s_goal)"
  },
    'WM': {
    deep: `<ul>
      <li><strong>L'Algorithme CEM (Model Predictive Control) :</strong>
        <ol>
          <li>Initialiser $P(a) = \text{Uniform}(0, 1)$ sur horizon $H$ (ex: $H=5$).</li>
          <li>Boucle $M$ itérations : Échantillonner $N$ séquences (Shape: <code>[100, 5, 4]</code>).</li>
          <li>Simuler via World Model et évaluer le coût cumulé.</li>
          <li>Garder les $K$ séquences d'élite et mettre à jour $P(a)$.</li>
        </ol>
      </li>
      <li><strong>Mode-2 de LeCun :</strong> Planification délibérative ("s'arrêter pour réfléchir").</li>
    </ul>`,

    math: "a^*_{t:t+H} = \\arg\\min_{a} \\sum_{\\tau=t}^{t+H} \\gamma^{\\tau-t} c(\\hat{s}_\\tau)",
    icon: "",
    title: "Acteur (Planificateur)",
    subtitle: "Fichier : modules/actor.py",
    simple: "C'est le cerveau stratégique. Avant de faire un vrai pas, il s'assoit et simule mentalement 500 chemins possibles. Il évalue chaque chemin, croise les meilleurs pour en créer de nouveaux, jusqu'à trouver le chemin parfait.",
    tech: "Implémente la méthode CEM (Cross-Entropy Method). Génère N trajectoires aléatoires, garde les K meilleures, ajuste sa distribution de probabilité d'actions et recommence. Model Predictive Control.",
    input: "s_t actuel",
    output: "Une action parfaite a_t"
  },
    'SIM': {
    deep: `<ul>
      <li><strong>Architecture Récurrente :</strong> Réseau (MLP/RNN) qui prédit les conséquences d'une action strictement dans l'espace latent : $\hat{s}_{t+1} = \text{Pred}_\phi(s_t, z_t, a_t)$.</li>
      <li><strong>Perte d'Apprentissage (JEPA Loss) :</strong> La prédiction est comparée au véritable état encodé par l'Encodeur Cible $s'_{t+1}$, <em>jamais</em> aux pixels.</li>
      <li>$\mathcal{L}_{WM} = \mathbb{E}[\Vert\hat{s}_{t+1} - s'_{t+1}\Vert_2^2]$</li>
    </ul>`,

    math: "L_{JEPA} = \\| \\hat{s}_{t+1} - s_{t+1} \\|_2^2",
    icon: "",
    title: "Modèle du Monde (JEPA)",
    subtitle: "Fichier : modules/world_model.py",
    simple: "C'est l'imagination de l'agent. Quand l'Acteur demande 'Et si j'allais à droite ?', le Modèle du Monde prédit la conséquence. Il prédit l'avenir directement sous forme de concept, sans dessiner les pixels.",
    tech: "Cœur de l'architecture JEPA. Un RNN (Recurrent Neural Network) ou MLP séquentiel qui calcule s_t+1 = f(s_t, action). Entraîné en mode auto-supervisé avec Backpropagation Through Time (BPTT).",
    input: "s_t, action imaginée",
    output: "s_next (Avenir prédit)"
  },
    'COST': {
    deep: `<ul>
      <li><strong>Coût Intrinsèque (Boussole) :</strong> $c_{intr} = \Vert \hat{s}_{t+1} - s_{goal} \Vert_2$.</li>
      <li><strong>Le Critique (Value Function $V_{\psi}$) :</strong> Réseau MLP prédisant la douleur/plaisir à long terme. $c_{total} = w_{goal} \cdot c_{intr} + V_{\psi}(\hat{s}_{t+1})$.</li>
      <li><strong>Apprentissage (TD-Learning) :</strong> $\mathcal{L}_{Critic} = (V_\psi(s_t) - (c_{intr\_reel} + \gamma V_\psi(s_{t+1})))^2$</li>
    </ul>`,

    math: "L_{critic} = \\| V(s_t) - (c_t + \\gamma V_{target}(s_{t+1})) \\|_2^2",
    icon: "",
    title: "Critique",
    subtitle: "Fichier : modules/cost.py",
    simple: "C'est l'intuition de l'agent forgée par son expérience. Il donne une mauvaise note aux chemins imaginés qui mènent dans un mur (car ça fait mal), et une bonne note à ceux qui rapprochent du but.",
    tech: "Combine la distance euclidienne (L2) vers l'objectif avec un réseau neuronal (Q-value / V-value). Entraîné via TD-Learning (Temporal Difference) avec une cible mise à jour par Exponential Moving Average (EMA).",
    input: "s_next imaginé",
    output: "Score / Coût scalaire"
  },
  'MEM': {
    deep: `<ul>
      <li><strong>Entraînement Asynchrone :</strong> Stocke les batchs extraits pour un entraînement hors ligne ou itératif.</li>
      <li><strong>Dynamique Globale (Backprop) :</strong>
        <br><code>opt_encoder.zero_grad()</code>
        <br><code>loss_total = loss_jepa + lambda * loss_sigreg</code>
        <br><code>loss_total.backward()</code>
      </li>
      <li>Déclenche la mise à jour EMA de l'Encodeur Cible.</li>
    </ul>`,

    icon: "",
    title: "Replay Buffer",
    subtitle: "Fichier : modules/memory.py",
    simple: "C'est le journal intime de l'agent. Pendant qu'il joue, il écrit tout ce qui lui arrive. Le soir (en arrière-plan), ses réseaux de neurones relisent ce journal pour s'entraîner et devenir plus intelligents.",
    tech: "Structure de données (buffer circulaire) stockant les 10 000 dernières transitions (état, action, récompense, état_suivant). Permet d'entraîner le World Model et le Critique en Off-Policy.",
    input: "Expériences en direct",
    output: "Batchs pour entraînement"
  }
};
