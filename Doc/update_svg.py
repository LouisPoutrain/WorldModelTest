import re

with open('Doc/Archi.html', 'r') as f:
    html = f.read()

new_svg = """<svg viewBox="0 0 900 650" id="arch-svg" xmlns="http://www.w3.org/2000/svg">
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
                    <rect x="20" y="50" width="180" height="130" rx="12" class="zone-rect"/>
                    <text x="110" y="70" class="zone-title">MONDE EXTÉRIEUR</text>

                    <!-- Agent -->
                    <rect x="250" y="30" width="620" height="600" rx="16" class="zone-rect"/>
                    <text x="560" y="55" class="zone-title">AGENT INTELLIGENT (JEPA)</text>

                    <!-- Boucle CEM -->
                    <rect x="460" y="180" width="390" height="330" rx="12" class="zone-rect-cem"/>
                    <text x="655" y="205" class="zone-title-cem">🔄 BOUCLE D'IMAGINATION (CEM)</text>


                    <!-- CONNECTIONS -->
                    <g class="conn-group" data-src="ENV" data-dst="PERC">
                        <path class="conn-line" d="M 180 125 L 270 125" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="225" y="115" text-anchor="middle">Vision</text>
                    </g>

                    <g class="conn-group" data-src="ENV" data-dst="CONF">
                        <path class="conn-line" d="M 110 160 L 110 335 L 270 335" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="195" y="325" text-anchor="middle">Énergie</text>
                    </g>

                    <g class="conn-group" data-src="PERC" data-dst="WM">
                        <path class="conn-line" d="M 410 125 L 445 125 L 445 260 L 470 260" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="455" y="190" text-anchor="start">s_t</text>
                    </g>

                    <g class="conn-group" data-src="CONF" data-dst="WM">
                        <path class="conn-line" d="M 410 335 L 445 335 L 445 300 L 470 300" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="455" y="330" text-anchor="start">s_goal</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="SIM">
                        <path class="conn-line" d="M 620 260 L 690 260" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="655" y="250" text-anchor="middle">a_sim</text>
                    </g>

                    <g class="conn-group" data-src="SIM" data-dst="WM">
                        <path class="conn-line" d="M 700 300 L 630 300" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="665" y="320" text-anchor="middle">s_next</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="COST">
                        <path class="conn-line" d="M 550 330 L 550 445 L 670 445" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="610" y="435" text-anchor="middle">Évaluer</text>
                    </g>

                    <g class="conn-group" data-src="COST" data-dst="WM">
                        <path class="conn-line" d="M 755 410 L 755 370 L 600 370 L 600 340" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="675" y="360" text-anchor="middle">Coût</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="ENV">
                        <path class="conn-line conn-main" d="M 480 280 L 360 280 L 360 500 L 90 500 L 90 170" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text conn-main-text" x="225" y="490" text-anchor="middle">Action choisie (a_t)</text>
                    </g>

                    <g class="conn-group" data-src="ENV" data-dst="MEM">
                        <path class="conn-line conn-dashed" d="M 70 160 L 70 585 L 640 585" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="350" y="575" text-anchor="middle">Souvenirs</text>
                    </g>

                    <!-- NODES -->
                    <!-- 1. ENV -->
                    <g class="node" id="node-ENV" onclick="selectModule('ENV')">
                        <rect x="40" y="90" width="140" height="70" rx="10" class="node-rect node-env"/>
                        <text x="110" y="125" class="node-text">🌍 Environnement</text>
                        <text x="110" y="145" class="node-subtext">Gridworld 10x10</text>
                    </g>

                    <!-- 2. PERC -->
                    <g class="node" id="node-PERC" onclick="selectModule('PERC')">
                        <rect x="280" y="90" width="130" height="70" rx="10" class="node-rect node-core"/>
                        <text x="345" y="125" class="node-text">👁️ Perception</text>
                        <text x="345" y="145" class="node-subtext">Réseau Visuel</text>
                    </g>

                    <!-- 3. CONF -->
                    <g class="node" id="node-CONF" onclick="selectModule('CONF')">
                        <rect x="280" y="300" width="130" height="70" rx="10" class="node-rect node-core"/>
                        <text x="345" y="335" class="node-text">🧭 Objectif</text>
                        <text x="345" y="355" class="node-subtext">Directeur</text>
                    </g>

                    <!-- 4. WM -->
                    <g class="node" id="node-WM" onclick="selectModule('WM')">
                        <rect x="480" y="230" width="140" height="100" rx="10" class="node-rect node-core"/>
                        <text x="550" y="265" class="node-text">🎯 Acteur</text>
                        <text x="550" y="285" class="node-subtext">Planificateur</text>
                        <text x="550" y="305" class="node-subtext">(Méthode CEM)</text>
                    </g>

                    <!-- 5. SIM -->
                    <g class="node" id="node-SIM" onclick="selectModule('SIM')">
                        <rect x="700" y="230" width="130" height="100" rx="10" class="node-rect node-sim"/>
                        <text x="765" y="265" class="node-text">🧠 Modèle du</text>
                        <text x="765" y="285" class="node-text">Monde</text>
                        <text x="765" y="305" class="node-subtext">Simulateur JEPA</text>
                    </g>

                    <!-- 6. COST -->
                    <g class="node" id="node-COST" onclick="selectModule('COST')">
                        <rect x="680" y="410" width="150" height="70" rx="10" class="node-rect node-sim"/>
                        <text x="755" y="440" class="node-text">⚖️ Critique</text>
                        <text x="755" y="460" class="node-subtext">Évaluateur</text>
                    </g>

                    <!-- 7. MEM -->
                    <g class="node" id="node-MEM" onclick="selectModule('MEM')">
                        <rect x="650" y="550" width="180" height="70" rx="10" class="node-rect node-mem"/>
                        <text x="740" y="585" class="node-text">💾 Replay Buffer</text>
                        <text x="740" y="605" class="node-subtext">Mémoire (10k épisodes)</text>
                    </g>
                </svg>"""

pattern = re.compile(r'<svg viewBox="0 0 700 560" id="arch-svg".*?</svg>', re.DOTALL)
if pattern.search(html):
    html = pattern.sub(new_svg, html)
else:
    # Just to be safe if the viewBox was changed
    pattern = re.compile(r'<svg viewBox=".*?id="arch-svg".*?</svg>', re.DOTALL)
    html = pattern.sub(new_svg, html)

with open('Doc/Archi.html', 'w') as f:
    f.write(html)

print("SVG coordinates updated successfully!")
