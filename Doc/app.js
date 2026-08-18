// Wait for DOM
document.addEventListener('DOMContentLoaded', () => {
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
    }
});
