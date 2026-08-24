
        const nodeData = {
            'ENV': {
                icon: '🌍',
                title: 'Environnement Physique',
                subtitle: 'GridWorld 10x10',
                simple: "C'est la grille physique réelle, qui cache des murs, des impasses, et la cible. L'agent ne reçoit qu'un petit carré de 5x5 pixels autour de lui (vision locale).",
                tech: "Environnement Gym-like qui met à jour la position de l'agent si la collision renvoie False. Le vrai layout global y est stocké secrètement.",
                deep: "<ul><li><strong>Rendu visuel :</strong> Tensor NumPy binarisé.</li><li><strong>Step :</strong> <code>obs, reward, done, info = env.step(action)</code></li></ul>",
                input: "Meilleure action a_t",
                output: "Observation partielle (x_t)"
            },
            'ASTAR': {
                icon: '🗺️',
                title: 'A* MacroPlanner',
                subtitle: 'L\'Omniscient',
                simple: "Il triche ! Il regarde la grille physique entière (vue de haut) et trace le chemin mathématiquement parfait jusqu'à la fin. Sans lui, notre agent se perdrait car il est myope.",
                tech: "Algorithme de pathfinding déterministe. Calcule le plus court chemin en évitant les murs grâce à une heuristique de Manhattan.",
                deep: "<ul><li><strong>Calcul :</strong> Fonction récursive de recherche de nœuds ouverts/fermés.</li><li><strong>Stockage :</strong> Liste de tuples (x, y) représentant chaque pas vers la cible.</li></ul>",
                input: "Grille globale",
                output: "Liste de coordonnées"
            },
            'WP': {
                icon: '📍',
                title: 'Waypoint',
                subtitle: 'Lookahead=1',
                simple: "Au lieu de montrer la fin du labyrinthe à l'agent, on lui montre juste la prochaine case du chemin tracé par A*. C'est le secret pour éviter qu'il ne se cogne dans les murs !",
                tech: "Prend la trajectoire d'A* et extrait l'indice `actuel + lookahead`. Avec lookahead=1, c'est la case immédiatement adjacente valide.",
                math: "W_t = Path_{A*}[t + 1]",
                input: "Trajectoire A*",
                output: "Coordonnées (x, y) cible"
            },
            'TOBS': {
                icon: '👁️',
                title: 'Observation Cible',
                subtitle: 'Image Synthétique',
                simple: "On dessine informatiquement ce que l'agent verrait s'il était déjà arrivé sur le waypoint.",
                tech: "Fonction `get_target_observation(env, waypoint)` qui force l'environnement à générer une vue 5x5 virtuelle depuis une position arbitraire, sans déplacer le véritable agent.",
                input: "Waypoint (x, y)",
                output: "Vision virtuelle (x_waypoint)"
            },
            'PERC': {
                icon: '🧠',
                title: 'Perception (CNN)',
                subtitle: 'L\'Encodeur',
                simple: "Traduit les images en idées mathématiques (vecteurs). Il compresse ce que l'agent voit actuellement, ET l'image synthétique du waypoint qu'il doit atteindre.",
                tech: "Réseau Convolutif entraîné via JEPA (sans labels). Convertit un tenseur [1, 5, 5] en un vecteur latent [16].",
                deep: "<ul><li><strong>Encodeur :</strong> <code>CNNEncoder(in_channels=1, hidden_dim=16)</code></li><li><strong>Traitement :</strong> Deux appels, l'un pour $x_t$, l'autre pour $x_{waypoint}$.</li></ul>",
                input: "Images (x_t, x_waypoint)",
                output: "Latents (s_t, s_waypoint)"
            },
            'ACT': {
                icon: '🎲',
                title: 'CEM Actor',
                subtitle: 'Planificateur T=1',
                simple: "Propose 1024 actions immédiates possibles au hasard. Comme on a réduit l'horizon à 1 (T=1), il sature littéralement les 4 directions de base.",
                tech: "Cross-Entropy Method bridée. Au lieu de séquences de longueur 5, il génère des tenseurs d'actions pour $t=1$.",
                math: "A \\sim \\mathcal{N}(\\mu, \\Sigma) \\text{ pour } T=1",
                input: "s_t actuel",
                output: "Séquences a_t"
            },
            'SIM': {
                icon: '⚙️',
                title: 'Simulation Latente',
                subtitle: 'OneHot Prep',
                simple: "Prépare les actions imaginées par l'Acteur dans un format (One-Hot) compréhensible par le cerveau du World Model.",
                tech: "Conversion de l'action discrète {0,1,2,3} en vecteur OneHot pour compatibilité matricielle avec $s_t$.",
                input: "Actions discrètes a_t",
                output: "Actions OneHot"
            },
            'WM': {
                icon: '🔮',
                title: 'World Model',
                subtitle: 'Réseau ConvGRU',
                simple: "Il imagine le futur ! Il prend la position actuelle, regarde l'action imaginée, et prédit à quoi ressemblera l'agent dans 1 pas.",
                tech: "Cellule GRU convolutive adaptée. Calcule l'état latent futur sans jamais repasser par l'espace des pixels (JEPA). Loss finale : 0.45 !",
                math: "\\hat{s}_{t+1} = \\text{ConvGRU}(s_t, a_{onehot})",
                input: "s_t, action OneHot",
                output: "s_{t+1} prédit"
            },
            'COST': {
                icon: '⚖️',
                title: 'Fonction de Coût',
                subtitle: 'Distance de Manhattan',
                simple: "Compare l'avenir imaginé avec le waypoint cible. Donne une mauvaise note (grand coût) si c'est loin, et une excellente note (coût zéro) si c'est la bonne case.",
                tech: "Calcule la norme L2 ou Manhattan entre le latent imaginé $\\hat{s}_{t+1}$ et le latent cible $s_{waypoint}$.",
                math: "Cost = \\Vert \\hat{s}_{t+1} - s_{waypoint} \\Vert_2",
                input: "s_{t+1} et s_{waypoint}",
                output: "Coût scalaire"
            },
            'EVAL': {
                icon: '🏆',
                title: 'Évaluation',
                subtitle: 'Sélection des Élites',
                simple: "Garde uniquement les 100 meilleures idées de l'Acteur. À la fin de la réflexion, l'idée gagnante est officiellement exécutée dans le vrai labyrinthe.",
                tech: "Tri de la matrice de coûts (argsort). Extrait le top-K (élites) pour mettre à jour la moyenne et variance du CEM.",
                deep: "<ul><li><strong>Output finale :</strong> <code>best_action = elites[0][0].item()</code></li></ul>",
                input: "Coûts de toutes les actions",
                output: "Meilleure action a_t"
            }
        };

        const nodes = document.querySelectorAll('.node');
        const defaultInfo = document.getElementById('panel-placeholder');
        const dynamicInfo = document.getElementById('panel-content');
        
        window.showNode = function(moduleId) {
            try {
                nodes.forEach(n => n.classList.remove('active'));
            
            const clickedNode = document.getElementById('node-' + moduleId);
            if (clickedNode) clickedNode.classList.add('active');

            document.querySelectorAll('.conn-group').forEach(group => {
                const src = group.getAttribute('data-src');
                const dst = group.getAttribute('data-dst');
                if (src === 'node-' + moduleId || dst === 'node-' + moduleId) {
                    group.classList.add('active');
                } else {
                    group.classList.remove('active');
                }
            });

            const data = nodeData[moduleId];
            if (data) {
                defaultInfo.style.display = 'none';
                dynamicInfo.style.display = 'block';
                
                document.getElementById('p-icon').textContent = data.icon;
                document.getElementById('p-title').textContent = data.title;
                document.getElementById('p-subtitle').textContent = data.subtitle;
                document.getElementById('p-simple').innerHTML = data.simple;
                document.getElementById('p-tech').innerHTML = data.tech;
                
                // Optional sections
                const deepSection = document.getElementById('p-deep-section');
                if (data.deep) {
                    deepSection.style.display = 'block';
                    document.getElementById('p-deep').innerHTML = data.deep;
                } else {
                    deepSection.style.display = 'none';
                }

                const mathSection = document.getElementById('p-math-section');
                if (data.math) {
                    mathSection.style.display = 'block';
                    document.getElementById('p-math').textContent = "$$ " + data.math + " $$";
                } else {
                    mathSection.style.display = 'none';
                }

                document.getElementById('p-in').textContent = data.input;
                document.getElementById('p-out').textContent = data.output;
                
                if (window.renderMathInElement) {
                    renderMathInElement(dynamicInfo, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false}
                        ]
                    });
                }
            }
            } catch(e) {
                document.getElementById('panel-placeholder').innerHTML = "<h3 style='color:red;'>JS Error</h3><p>" + e.message + "</p><pre>" + e.stack + "</pre>";
                document.getElementById('panel-placeholder').style.display = 'block';
                document.getElementById('panel-content').style.display = 'none';
            }
        };
    