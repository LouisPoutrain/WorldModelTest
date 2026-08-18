const modulesData = {
    'ENV': {
        icon: "🌍",
        title: "Environnement",
        subtitle: "Fichier : env/gridworld.py",
        simple: "C'est le monde physique (le labyrinthe). L'agent s'y déplace, se cogne contre les murs, et consomme de l'énergie. L'environnement lui donne juste ce qu'il a devant lui (vision 5x5).",
        tech: "Grille 2D générée de manière procédurale. Utilise un algorithme BFS (Breadth-First Search) pour garantir mathématiquement qu'il y a toujours un chemin possible vers la cible et la station.",
        input: "Action choisie a_t",
        output: "Image 5x5 (torch.Tensor)"
    },
    'PERC': {
        icon: "👁️",
        title: "Perception",
        subtitle: "Fichier : modules/perception.py",
        simple: "L'œil de l'agent. Il regarde l'image brute (les pixels) et la compresse en une 'idée mathématique' (un vecteur de 32 nombres). C'est beaucoup plus facile de réfléchir avec des idées qu'avec des pixels.",
        tech: "Petit réseau de neurones convolutif (ResNet). Il n'est pas entraîné par erreur classique mais via une régularisation appelée SIGReg, qui force ses 32 neurones à être indépendants et informatifs.",
        input: "Vision 5x5",
        output: "s_t (Vecteur de taille 32)"
    },
    'CONF': {
        icon: "🧭",
        title: "Configurateur",
        subtitle: "Fichier : modules/configurator.py",
        simple: "C'est l'instinct de survie. Il surveille la jauge d'énergie. Si elle est haute, il dit à l'agent de chercher la cible. Si elle est basse, il annule la mission et le force à chercher une station de recharge.",
        tech: "Règle algorithmique basée sur des seuils. Change dynamiquement la cible virtuelle s_goal que l'Acteur essaiera d'atteindre pendant sa planification.",
        input: "Énergie actuelle",
        output: "Cible (s_goal)"
    },
    'WM': {
        icon: "🎯",
        title: "Acteur (Planificateur)",
        subtitle: "Fichier : modules/actor.py",
        simple: "C'est le cerveau stratégique. Avant de faire un vrai pas, il s'assoit et simule mentalement 500 chemins possibles. Il évalue chaque chemin, croise les meilleurs pour en créer de nouveaux, jusqu'à trouver le chemin parfait.",
        tech: "Implémente la méthode CEM (Cross-Entropy Method). Génère N trajectoires aléatoires, garde les K meilleures, ajuste sa distribution de probabilité d'actions et recommence. Model Predictive Control.",
        input: "s_t actuel",
        output: "Une action parfaite a_t"
    },
    'SIM': {
        icon: "🧠",
        title: "Modèle du Monde (JEPA)",
        subtitle: "Fichier : modules/world_model.py",
        simple: "C'est l'imagination de l'agent. Quand l'Acteur demande 'Et si j'allais à droite ?', le Modèle du Monde prédit la conséquence. Il prédit l'avenir directement sous forme de concept, sans dessiner les pixels.",
        tech: "Cœur de l'architecture JEPA. Un RNN (Recurrent Neural Network) ou MLP séquentiel qui calcule s_t+1 = f(s_t, action). Entraîné en mode auto-supervisé avec Backpropagation Through Time (BPTT).",
        input: "s_t, action imaginée",
        output: "s_next (Avenir prédit)"
    },
    'COST': {
        icon: "⚖️",
        title: "Critique",
        subtitle: "Fichier : modules/cost.py",
        simple: "C'est l'intuition de l'agent forgée par son expérience. Il donne une mauvaise note aux chemins imaginés qui mènent dans un mur (car ça fait mal), et une bonne note à ceux qui rapprochent du but.",
        tech: "Combine la distance euclidienne (L2) vers l'objectif avec un réseau neuronal (Q-value / V-value). Entraîné via TD-Learning (Temporal Difference) avec une cible mise à jour par Exponential Moving Average (EMA).",
        input: "s_next imaginé",
        output: "Score / Coût scalaire"
    },
    'MEM': {
        icon: "💾",
        title: "Replay Buffer",
        subtitle: "Fichier : modules/memory.py",
        simple: "C'est le journal intime de l'agent. Pendant qu'il joue, il écrit tout ce qui lui arrive. Le soir (en arrière-plan), ses réseaux de neurones relisent ce journal pour s'entraîner et devenir plus intelligents.",
        tech: "Structure de données (buffer circulaire) stockant les 10 000 dernières transitions (état, action, récompense, état_suivant). Permet d'entraîner le World Model et le Critique en Off-Policy.",
        input: "Expériences en direct",
        output: "Batchs pour entraînement"
    }
};
