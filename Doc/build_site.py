import os

html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep-JEPA — Architecture Interactive</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>

    <!-- ==================== HEADER ==================== -->
    <header class="hero">
        <div class="hero-glow"></div>
        <div class="hero-content">
            <span class="hero-badge">🧠 Intelligence Artificielle Autonome</span>
            <h1>Deep-JEPA</h1>
            <p class="hero-subtitle">
                Un agent capable de <strong>penser avant d'agir</strong>.
                <br>Inspiré de l'architecture cognitive proposée par <em>Yann LeCun</em> (Meta AI / Turing Award 2018).
            </p>
        </div>
    </header>

    <!-- ==================== INTRO SECTION ==================== -->
    <section class="intro-section">
        <div class="intro-card">
            <h2>💡 L'idée en une phrase</h2>
            <p>
                Au lieu d'apprendre par essai-erreur aveugle, notre IA possède une capacité <strong>d'imagination</strong>. Elle simule mentalement les conséquences de ses actes avant même de bouger.
            </p>
        </div>
        <div class="intro-card">
            <h2>🎯 La mission</h2>
            <p>
                Trouver une cible dans un <strong>labyrinthe totalement inconnu</strong>, tout en gérant son énergie. L'agent doit planifier son chemin et faire des détours pour se recharger si besoin.
            </p>
        </div>
        <div class="intro-card">
            <h2>🧬 L'Espace Latent</h2>
            <p>
                Au lieu d'imaginer des images complexes pixel par pixel, l'agent réfléchit dans un espace mathématique simplifié (<strong>l'espace latent</strong>). C'est beaucoup plus rapide et performant.
            </p>
        </div>
    </section>

    <!-- ==================== MAIN ARCHITECTURE ==================== -->
    <section class="arch-section">
        <h2 class="section-title">Le Cerveau de l'Agent</h2>
        <p class="section-subtitle">Cliquez sur un module du schéma pour comprendre son rôle (Panneau de droite)</p>

        <div class="arch-layout">
            <!-- LEFT: SVG Diagram -->
            <div class="diagram-panel">
                <svg viewBox="0 0 700 560" id="arch-svg" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <!-- Arrow markers -->
                        <marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
                        </marker>
                        <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8"/>
                        </marker>
                    </defs>

                    <!-- Background grids for visual flair -->
                    <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#1e293b" stroke-width="1" opacity="0.5"/>
                    </pattern>
                    <rect width="100%" height="100%" fill="url(#grid)" rx="12"/>

                    <!-- Zones (Subgraphs) -->
                    <!-- Monde Extérieur -->
                    <rect x="20" y="40" width="160" height="120" rx="12" class="zone-rect"/>
                    <text x="100" y="60" class="zone-title">MONDE EXTÉRIEUR</text>

                    <!-- Agent -->
                    <rect x="210" y="20" width="470" height="510" rx="16" class="zone-rect"/>
                    <text x="445" y="45" class="zone-title">AGENT INTELLIGENT (JEPA)</text>

                    <!-- Boucle CEM -->
                    <rect x="370" y="150" width="290" height="280" rx="12" class="zone-rect-cem"/>
                    <text x="515" y="170" class="zone-title-cem">🔄 BOUCLE D'IMAGINATION (CEM)</text>


                    <!-- CONNECTIONS -->
                    <path class="conn-line" d="M 160 95 L 240 95" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="200" y="85">Vision</text>

                    <path class="conn-line" d="M 100 130 L 100 285 L 240 285" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="160" y="275">Énergie</text>

                    <path class="conn-line" d="M 350 95 L 400 95 L 400 225 L 390 225" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="380" y="160">s_t</text>

                    <path class="conn-line" d="M 350 285 L 390 285" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="370" y="275">s_goal</text>

                    <path class="conn-line" d="M 500 195 L 500 160 L 530 160 L 530 195" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="515" y="172">a_sim</text>

                    <path class="conn-line" d="M 590 285 L 590 310 L 480 310 L 480 290" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="535" y="322">s_next</text>

                    <path class="conn-line" d="M 500 290 L 500 350 L 520 350" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="510" y="340">Évaluer</text>

                    <path class="conn-line" d="M 520 380 L 460 380 L 460 290" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="490" y="392">Coût</text>

                    <path class="conn-line conn-main" d="M 390 255 L 300 255 L 300 480 L 100 480 L 100 150" marker-end="url(#arrow-active)"/>
                    <text class="conn-text conn-main-text" x="200" y="495">Action choisie (a_t)</text>

                    <path class="conn-line conn-dashed" d="M 100 150 L 100 520 L 500 520" marker-end="url(#arrow-gray)"/>
                    <text class="conn-text" x="250" y="535">Souvenirs</text>


                    <!-- NODES -->
                    <!-- 1. ENV -->
                    <g class="node" id="node-ENV" onclick="selectModule('ENV')">
                        <rect x="30" y="70" width="130" height="60" rx="8" class="node-rect node-env"/>
                        <text x="95" y="95" class="node-text">🌍 Environnement</text>
                        <text x="95" y="115" class="node-subtext">Gridworld 10x10</text>
                    </g>

                    <!-- 2. PERC -->
                    <g class="node" id="node-PERC" onclick="selectModule('PERC')">
                        <rect x="240" y="65" width="110" height="60" rx="8" class="node-rect node-core"/>
                        <text x="295" y="90" class="node-text">👁️ Perception</text>
                        <text x="295" y="110" class="node-subtext">Réseau Visuel</text>
                    </g>

                    <!-- 3. CONF -->
                    <g class="node" id="node-CONF" onclick="selectModule('CONF')">
                        <rect x="240" y="255" width="110" height="60" rx="8" class="node-rect node-core"/>
                        <text x="295" y="280" class="node-text">🧭 Objectif</text>
                        <text x="295" y="300" class="node-subtext">Directeur</text>
                    </g>

                    <!-- 4. WM -->
                    <g class="node" id="node-WM" onclick="selectModule('WM')">
                        <rect x="390" y="195" width="110" height="95" rx="8" class="node-rect node-core"/>
                        <text x="445" y="225" class="node-text">🎯 Acteur</text>
                        <text x="445" y="245" class="node-subtext">Planificateur</text>
                        <text x="445" y="265" class="node-subtext">(Méthode CEM)</text>
                    </g>

                    <!-- 5. SIM -->
                    <g class="node" id="node-SIM" onclick="selectModule('SIM')">
                        <rect x="530" y="195" width="110" height="90" rx="8" class="node-rect node-sim"/>
                        <text x="585" y="225" class="node-text">🧠 Modèle du</text>
                        <text x="585" y="245" class="node-text">Monde</text>
                        <text x="585" y="265" class="node-subtext">Simulateur JEPA</text>
                    </g>

                    <!-- 6. COST -->
                    <g class="node" id="node-COST" onclick="selectModule('COST')">
                        <rect x="520" y="340" width="130" height="60" rx="8" class="node-rect node-sim"/>
                        <text x="585" y="365" class="node-text">⚖️ Critique</text>
                        <text x="585" y="385" class="node-subtext">Évaluateur</text>
                    </g>

                    <!-- 7. MEM -->
                    <g class="node" id="node-MEM" onclick="selectModule('MEM')">
                        <rect x="500" y="490" width="160" height="60" rx="8" class="node-rect node-mem"/>
                        <text x="580" y="515" class="node-text">💾 Replay Buffer</text>
                        <text x="580" y="535" class="node-subtext">Mémoire (10k épisodes)</text>
                    </g>
                </svg>
            </div>

            <!-- RIGHT: Detail Panel -->
            <div class="detail-panel" id="detail-panel">
                <!-- Initial empty state -->
                <div class="panel-placeholder" id="panel-placeholder">
                    <div class="icon-bounce">👈</div>
                    <h3>Explorez l'Architecture</h3>
                    <p>Cliquez sur l'un des blocs du diagramme interactif pour découvrir ce qui se passe sous le capot de notre IA.</p>
                </div>

                <!-- Content state (hidden initially) -->
                <div class="panel-content" id="panel-content" style="display: none;">
                    <div class="panel-header">
                        <div class="panel-icon-wrap">
                            <span class="panel-icon" id="p-icon"></span>
                        </div>
                        <div class="panel-header-text">
                            <h2 id="p-title">Titre</h2>
                            <span class="panel-subtitle" id="p-subtitle">Sous-titre</span>
                        </div>
                    </div>

                    <div class="panel-section">
                        <h3 class="section-heading">🎓 Pour faire simple...</h3>
                        <div class="concept-box">
                            <p id="p-simple"></p>
                        </div>
                    </div>

                    <div class="panel-section">
                        <h3 class="section-heading">⚙️ Sous le capot</h3>
                        <p id="p-tech"></p>
                    </div>

                    <div class="panel-section io-grid">
                        <div class="io-box in-box">
                            <span class="io-label">ENTRÉE</span>
                            <span class="io-val" id="p-in"></span>
                        </div>
                        <div class="io-box out-box">
                            <span class="io-label">SORTIE</span>
                            <span class="io-val" id="p-out"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== FLOW SECTION ==================== -->
    <section class="flow-section">
        <h2 class="section-title">L'Histoire d'une Décision</h2>
        <p class="section-subtitle">Voici comment l'agent raisonne avant de faire le moindre pas.</p>

        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-marker">1</div>
                <div class="timeline-content">
                    <h3>Je regarde</h3>
                    <p>L'agent observe sa vision locale (murs, cible). La <strong>Perception</strong> traduit cette image en un code mathématique (l'espace latent).</p>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">2</div>
                <div class="timeline-content">
                    <h3>Qu'est-ce qui est urgent ?</h3>
                    <p>Le <strong>Configurateur</strong> regarde la batterie. Si elle est pleine, la cible est le point vert. Si elle est vide, la priorité absolue devient la station de recharge la plus proche.</p>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">3</div>
                <div class="timeline-content">
                    <h3>J'imagine 500 avenirs</h3>
                    <p>Sans bouger, l'<strong>Acteur</strong> génère 500 chemins aléatoires (ex: "droite, haut, droite..."). Le <strong>Modèle du Monde</strong> simule l'avenir et prédit la situation de l'agent à la fin de chaque chemin.</p>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">4</div>
                <div class="timeline-content">
                    <h3>Je juge mes idées</h3>
                    <p>Le <strong>Critique</strong> donne une note à chaque avenir imaginé. Il pénalise les scénarios où l'agent se cogne dans un mur, et récompense ceux qui s'approchent de la cible.</p>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">5</div>
                <div class="timeline-content">
                    <h3>J'affine mon plan</h3>
                    <p>L'Acteur garde les 50 meilleurs chemins, les mélange, et imagine 500 <em>nouvelles</em> variations autour de ces bonnes idées. Il répète ça 10 fois pour trouver le chemin parfait.</p>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">6</div>
                <div class="timeline-content">
                    <h3>J'agis</h3>
                    <p>L'agent exécute <strong>uniquement le tout premier pas</strong> de son plan parfait. Puis, il rouvre les yeux et recommence tout le processus depuis le début !</p>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== FOOTER ==================== -->
    <footer>
        <p>Projet d'Intelligence Artificielle Autonome (Deep-JEPA) - 2026</p>
    </footer>

    <script src="data.js"></script>
    <script src="app.js"></script>
</body>
</html>"""

css_content = """/* Global Styles & Variables */
:root {
    --bg-color: #0f172a; /* Slate 900 */
    --card-bg: #1e293b; /* Slate 800 */
    --card-border: #334155; /* Slate 700 */
    
    --primary: #38bdf8; /* Light Blue */
    --secondary: #a855f7; /* Purple */
    --accent: #f59e0b; /* Amber */
    --success: #10b981; /* Emerald */
    
    --text-main: #f8fafc; /* Slate 50 */
    --text-muted: #94a3b8; /* Slate 400 */
    
    --font-sans: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    background-color: var(--bg-color);
    color: var(--text-main);
    font-family: var(--font-sans);
    line-height: 1.6;
    overflow-x: hidden;
}

/* ================= HERO ================= */
.hero {
    position: relative;
    text-align: center;
    padding: 80px 20px 60px;
    border-bottom: 1px solid var(--card-border);
    overflow: hidden;
}

.hero-glow {
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
    z-index: 0;
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 800px;
    margin: 0 auto;
}

.hero-badge {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(56, 189, 248, 0.1);
    color: var(--primary);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 20px;
}

h1 {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 20px;
    background: linear-gradient(to right, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-muted);
    margin-bottom: 0;
}

.hero-subtitle strong { color: var(--text-main); }

/* ================= INTRO ================= */
.intro-section {
    display: flex;
    gap: 20px;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
    flex-wrap: wrap;
}

.intro-card {
    flex: 1;
    min-width: 300px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 30px;
    transition: transform 0.3s ease;
}

.intro-card:hover { transform: translateY(-5px); }
.intro-card h2 { font-size: 1.25rem; color: var(--text-main); margin-bottom: 15px; }
.intro-card p { color: var(--text-muted); font-size: 0.95rem; }

/* ================= ARCHITECTURE SECTION ================= */
.arch-section {
    max-width: 1300px;
    margin: 0 auto;
    padding: 60px 20px;
}

.section-title {
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.section-subtitle {
    text-align: center;
    color: var(--primary);
    margin-bottom: 40px;
    font-weight: 500;
}

.arch-layout {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 30px;
    align-items: start;
}

@media (max-width: 900px) {
    .arch-layout { grid-template-columns: 1fr; }
}

/* LEFT: SVG Panel */
.diagram-panel {
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

#arch-svg {
    width: 100%;
    height: auto;
}

/* SVG Styles */
.zone-rect { fill: transparent; stroke: #334155; stroke-width: 1.5; stroke-dasharray: 6; }
.zone-title { fill: #64748b; font-size: 11px; font-weight: 700; text-anchor: middle; font-family: var(--font-sans); }
.zone-rect-cem { fill: rgba(168, 85, 247, 0.05); stroke: #7c3aed; stroke-width: 1; stroke-dasharray: 4; }
.zone-title-cem { fill: #a855f7; font-size: 11px; font-weight: 700; text-anchor: middle; font-family: var(--font-sans); }

.conn-line { fill: none; stroke: #475569; stroke-width: 2; }
.conn-text { fill: #94a3b8; font-size: 10px; font-weight: 500; font-family: var(--font-mono); }
.conn-main { stroke: var(--primary); stroke-width: 2.5; }
.conn-main-text { fill: var(--primary); font-weight: 700; }
.conn-dashed { stroke-dasharray: 4; }

.node { cursor: pointer; transition: transform 0.2s; transform-origin: center; }
.node:hover { transform: scale(1.05); filter: brightness(1.2); }
.node.active .node-rect { stroke-width: 3; stroke: #fff; filter: drop-shadow(0 0 8px rgba(255,255,255,0.4)); }

.node-rect { stroke-width: 1.5; }
.node-env { fill: #0c4a6e; stroke: #0ea5e9; }
.node-core { fill: #78350f; stroke: #f59e0b; }
.node-sim { fill: #4c1d95; stroke: #8b5cf6; }
.node-mem { fill: #1e293b; stroke: #64748b; }

.node-text { fill: #fff; font-size: 13px; font-weight: 600; font-family: var(--font-sans); text-anchor: middle; pointer-events: none;}
.node-subtext { fill: rgba(255,255,255,0.7); font-size: 10px; font-family: var(--font-sans); text-anchor: middle; pointer-events: none;}

/* RIGHT: Detail Panel */
.detail-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    min-height: 500px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Placeholder */
.panel-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 500px;
    padding: 40px;
    text-align: center;
    color: var(--text-muted);
}
.icon-bounce {
    font-size: 4rem;
    margin-bottom: 20px;
    animation: bounceX 2s infinite;
}
@keyframes bounceX {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(-15px); }
}

/* Real Content */
.panel-content {
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 25px 30px;
    border-bottom: 1px solid var(--card-border);
    background: rgba(0,0,0,0.2);
}
.panel-icon-wrap {
    width: 50px;
    height: 50px;
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}
.panel-header-text h2 { font-size: 1.4rem; margin: 0; color: #fff; }
.panel-subtitle { color: var(--primary); font-size: 0.9rem; font-family: var(--font-mono); }

.panel-section { padding: 25px 30px; border-bottom: 1px solid var(--card-border); }
.panel-section:last-child { border-bottom: none; }

.section-heading { font-size: 1rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; }

.concept-box {
    background: rgba(56, 189, 248, 0.1);
    border-left: 4px solid var(--primary);
    padding: 15px 20px;
    border-radius: 0 8px 8px 0;
    color: #e0f2fe;
    font-size: 1.05rem;
}

#p-tech { font-size: 0.95rem; color: #cbd5e1; }

.io-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    background: rgba(0,0,0,0.1);
}
.io-box {
    background: #0f172a;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid var(--card-border);
}
.io-label { display: block; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 8px; font-weight: 700;}
.io-val { font-family: var(--font-mono); font-size: 0.85rem; color: var(--secondary); }

/* ================= TIMELINE SECTION ================= */
.flow-section {
    max-width: 900px;
    margin: 0 auto;
    padding: 60px 20px;
}

.timeline {
    position: relative;
    margin-top: 40px;
}
.timeline::before {
    content: '';
    position: absolute;
    top: 0; left: 24px;
    height: 100%; width: 2px;
    background: var(--card-border);
}

.timeline-item {
    display: flex;
    gap: 30px;
    margin-bottom: 40px;
    position: relative;
}
.timeline-marker {
    width: 50px; height: 50px;
    min-width: 50px;
    background: var(--card-bg);
    border: 2px solid var(--primary);
    border-radius: 50%;
    display: flex;
    align-items: center; justify-content: center;
    font-weight: 700; color: var(--primary);
    font-size: 1.2rem;
    z-index: 1;
}
.timeline-content {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 25px;
    border-radius: 12px;
    flex: 1;
}
.timeline-content h3 { color: #fff; margin-bottom: 10px; }
.timeline-content p { color: var(--text-muted); font-size: 0.95rem; }

/* ================= FOOTER ================= */
footer {
    text-align: center;
    padding: 40px 20px;
    border-top: 1px solid var(--card-border);
    margin-top: 60px;
    color: var(--text-muted);
}
"""

js_data = """const modulesData = {
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
        simple: "L'œil de l'agent. Il regarde l'image brute (les pixels) et la compresse en une \"idée mathématique\" (un vecteur de 32 nombres). C'est beaucoup plus facile de réfléchir avec des idées qu'avec des pixels.",
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
        simple: "C'est l'imagination de l'agent. Quand l'Acteur demande \"Et si j'allais à droite ?\", le Modèle du Monde prédit la conséquence. Il prédit l'avenir directement sous forme de concept, sans dessiner les pixels.",
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
        tech: "Structure de données (buffer circulaire) stockant les 10 000 dernières transitions (état, action, récompense, état_suivant). Permet d'entraîner le World Model et le Critique en \"Off-Policy\".",
        input: "Expériences en direct",
        output: "Batchs pour entraînement"
    }
};
"""

js_app = """// Wait for DOM
document.addEventListener('DOMContentLoaded', () => {
    // Expose selectModule globally
    window.selectModule = function(moduleId) {
        const data = modulesData[moduleId];
        if (!data) return;

        // Visual update on SVG
        document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
        document.getElementById('node-' + moduleId).classList.add('active');

        // Show panel
        document.getElementById('panel-placeholder').style.display = 'none';
        const content = document.getElementById('panel-content');
        content.style.display = 'block';

        // Animate content
        content.style.animation = 'none';
        content.offsetHeight; /* trigger reflow */
        content.style.animation = null; 

        // Inject Data
        document.getElementById('p-icon').textContent = data.icon;
        document.getElementById('p-title').textContent = data.title;
        document.getElementById('p-subtitle').textContent = data.subtitle;
        
        document.getElementById('p-simple').textContent = data.simple;
        document.getElementById('p-tech').textContent = data.tech;
        
        document.getElementById('p-in').textContent = data.input;
        document.getElementById('p-out').textContent = data.output;
    }
});
"""

with open('Doc/Archi.html', 'w') as f:
    f.write(html_content)

with open('Doc/style.css', 'w') as f:
    f.write(css_content)

with open('Doc/data.js', 'w') as f:
    f.write(js_data)

with open('Doc/app.js', 'w') as f:
    f.write(js_app)

print("Files created successfully.")
