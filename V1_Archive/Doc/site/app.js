// Wait for DOM
document.addEventListener('DOMContentLoaded', () => {

    // Render math globally (for glossary and other static parts)
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
                {left: "\\(", right: "\\)", display: false},
                {left: "\\[", right: "\\]", display: true}
            ],
            throwOnError: false
        });
    }

    // Expose selectModule globally
    window.selectModule = function(moduleId) {
        const data = modulesData[moduleId];
        if (!data) return;

        // Visual update on SVG
        document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
        
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

        
        // Handle Deep Dive
        const deepSection = document.getElementById('p-deep-section');
        if (data.deep) {
            deepSection.style.display = 'block';
            document.getElementById('p-deep').innerHTML = data.deep;
            // Render Math in Deep Dive
            if (window.katex && window.renderMathInElement) {
                renderMathInElement(document.getElementById('p-deep'), {
                    delimiters: [
                        {left: "$$", right: "$$", display: true},
                        {left: "$", right: "$", display: false}
                    ]
                });
            }
        } else {
            deepSection.style.display = 'none';
        }

        // Handle Math
        const mathSection = document.getElementById('p-math-section');
        if (data.math) {
            mathSection.style.display = 'block';
            if (window.katex) {
                katex.render(data.math, document.getElementById('p-math'), { displayMode: true });
            } else {
                document.getElementById('p-math').textContent = data.math; // fallback
            }
        } else {
            mathSection.style.display = 'none';
        }

    }
});
