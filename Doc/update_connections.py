import re

with open('Doc/Archi.html', 'r') as f:
    html = f.read()

new_connections = """                    <!-- CONNECTIONS -->
                    <g class="conn-group" data-src="ENV" data-dst="PERC">
                        <path class="conn-line" d="M 160 95 L 240 95" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="200" y="85">Vision</text>
                    </g>

                    <g class="conn-group" data-src="ENV" data-dst="CONF">
                        <path class="conn-line" d="M 100 130 L 100 285 L 240 285" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="160" y="275">Énergie</text>
                    </g>

                    <g class="conn-group" data-src="PERC" data-dst="WM">
                        <path class="conn-line" d="M 350 95 L 400 95 L 400 225 L 390 225" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="380" y="160">s_t</text>
                    </g>

                    <g class="conn-group" data-src="CONF" data-dst="WM">
                        <path class="conn-line" d="M 350 285 L 390 285" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="370" y="275">s_goal</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="SIM">
                        <path class="conn-line" d="M 500 195 L 500 160 L 530 160 L 530 195" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="515" y="172">a_sim</text>
                    </g>

                    <g class="conn-group" data-src="SIM" data-dst="WM">
                        <path class="conn-line" d="M 590 285 L 590 310 L 480 310 L 480 290" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="535" y="322">s_next</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="COST">
                        <path class="conn-line" d="M 500 290 L 500 350 L 520 350" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="510" y="340">Évaluer</text>
                    </g>

                    <g class="conn-group" data-src="COST" data-dst="WM">
                        <path class="conn-line" d="M 520 380 L 460 380 L 460 290" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="490" y="392">Coût</text>
                    </g>

                    <g class="conn-group" data-src="WM" data-dst="ENV">
                        <path class="conn-line conn-main" d="M 390 255 L 300 255 L 300 480 L 100 480 L 100 150" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text conn-main-text" x="200" y="495">Action choisie (a_t)</text>
                    </g>

                    <g class="conn-group" data-src="ENV" data-dst="MEM">
                        <path class="conn-line conn-dashed" d="M 100 150 L 100 520 L 500 520" marker-end="url(#arrow-gray)"/>
                        <text class="conn-text" x="250" y="535">Souvenirs</text>
                    </g>"""

# Using regex to replace the connections block
pattern = re.compile(r'<!-- CONNECTIONS -->.*?<!-- NODES -->', re.DOTALL)
new_html = pattern.sub(new_connections + '\n\n                    <!-- NODES -->', html)

with open('Doc/Archi.html', 'w') as f:
    f.write(new_html)

print("Archi.html updated!")

# Now update app.js
with open('Doc/app.js', 'r') as f:
    app_js = f.read()

new_app_js = app_js.replace("document.getElementById('node-' + moduleId).classList.add('active');", """
        document.getElementById('node-' + moduleId).classList.add('active');

        // Highlight connections
        document.querySelectorAll('.conn-group').forEach(g => {
            if (g.getAttribute('data-src') === moduleId || g.getAttribute('data-dst') === moduleId) {
                g.classList.add('active-conn');
                // Change marker to active
                g.querySelector('path').setAttribute('marker-end', 'url(#arrow-active)');
            } else {
                g.classList.remove('active-conn');
                // Reset marker
                g.querySelector('path').setAttribute('marker-end', 'url(#arrow-gray)');
            }
        });
""")

with open('Doc/app.js', 'w') as f:
    f.write(new_app_js)

print("app.js updated!")
