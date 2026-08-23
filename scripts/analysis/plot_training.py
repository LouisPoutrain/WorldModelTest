"""
Script de visualisation des courbes d'entraînement.
Lit le fichier logs/training_log.csv et génère des graphiques.
"""
import csv
import os
import sys

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv(path):
    """Charge le CSV et retourne un dict de listes."""
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for header in reader.fieldnames:
            data[header] = []
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(val)
    return data

def plot_training(log_path="logs/training_log.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Fichier {log_path} introuvable.")
        print("   Lancez d'abord un entraînement avec main.py.")
        return
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif (compatible Colab/serveur)
    except ImportError:
        print("❌ matplotlib non installé. Installez-le avec: pip install matplotlib")
        return
    
    data = load_csv(log_path)
    
    if len(data.get('episode', [])) == 0:
        print("⚠️  Le fichier CSV est vide. Pas encore de données.")
        return
    
    episodes = data['episode']
    
    # Créer le dossier de sortie
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('📊 Deep-JEPA — Tableau de Bord d\'Entraînement', fontsize=16, fontweight='bold')
    
    # 1. Taux de succès
    ax = axes[0, 0]
    ax.plot(episodes, data['success_rate'], color='#2ecc71', linewidth=2, label='Succès')
    ax.axhline(y=0.8, color='#e74c3c', linestyle='--', alpha=0.5, label='Objectif 80%')
    ax.fill_between(episodes, data['success_rate'], alpha=0.15, color='#2ecc71')
    ax.set_title('🎯 Taux de Succès', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Taux')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Nombre de pas moyens
    ax = axes[0, 1]
    ax.plot(episodes, data['avg_steps'], color='#3498db', linewidth=2)
    ax.fill_between(episodes, data['avg_steps'], alpha=0.15, color='#3498db')
    ax.set_title('🦶 Pas Moyens par Épisode', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Pas')
    ax.grid(True, alpha=0.3)
    
    # 3. Pertes (L_pred + L_sigreg)
    ax = axes[1, 0]
    ax.plot(episodes, data['l_pred'], color='#e67e22', linewidth=2, label='L_pred (BPTT)')
    ax.plot(episodes, data['l_sigreg'], color='#9b59b6', linewidth=2, label='L_sigreg')
    ax.set_title('📉 Pertes du World Model', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. L_critic
    ax = axes[1, 1]
    ax.plot(episodes, data['l_critic'], color='#e74c3c', linewidth=2)
    ax.fill_between(episodes, data['l_critic'], alpha=0.15, color='#e74c3c')
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.5, label='Seuil vert (< 1.0)')
    ax.axhline(y=5.0, color='#f39c12', linestyle='--', alpha=0.5, label='Seuil jaune (< 5.0)')
    ax.set_title('🎓 Perte du Critique (TD-Learning)', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. V-values (Critique)
    ax = axes[2, 0]
    ax.plot(episodes, data['v_target'], color='#2ecc71', linewidth=2, label='V(target)')
    ax.plot(episodes, data['v_agent'], color='#e74c3c', linewidth=2, label='V(agent)')
    ax.fill_between(episodes, data['v_target'], data['v_agent'], alpha=0.1, color='#3498db')
    ax.set_title('🧠 Valeurs du Critique', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('V(s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Santé de l'espace latent
    ax = axes[2, 1]
    ax.plot(episodes, data['latent_std'], color='#1abc9c', linewidth=2, label='Std latent')
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.5, label='Idéal (1.0)')
    ax.axhspan(0.5, 2.0, alpha=0.08, color='#2ecc71', label='Zone saine')
    ax.set_title('🔬 Santé de l\'Espace Latent', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Écart-type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés dans {out_path}")
    plt.close()

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/training_log.csv"
    plot_training(log_file)
