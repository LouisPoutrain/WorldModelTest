import csv
import os
import sys

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv(path):
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

def plot_training(log_path="logs/training_jepa_v2.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Fichier {log_path} introuvable.")
        return
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("❌ matplotlib non installé. Installez-le avec: pip install matplotlib")
        return
    
    data = load_csv(log_path)
    
    # Pour V2 : Timestamp, Episode, L_pred, L_sigreg
    episodes = data['Episode']
    
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('📊 JEPA V2 — Évolution des Pertes', fontsize=16, fontweight='bold')
    
    # 1. L_pred
    ax = axes[0]
    ax.plot(episodes, data['L_pred'], color='#e67e22', linewidth=2)
    ax.fill_between(episodes, data['L_pred'], alpha=0.15, color='#e67e22')
    ax.set_title('📉 Perte de Prédiction (MSE)', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Loss (L_pred)')
    ax.grid(True, alpha=0.3)
    
    # 2. L_sigreg
    ax = axes[1]
    ax.plot(episodes, data['L_sigreg'], color='#9b59b6', linewidth=2)
    ax.fill_between(episodes, data['L_sigreg'], alpha=0.15, color='#9b59b6')
    ax.set_title('🔬 Régularisation Latente (SIGReg)', fontweight='bold')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Loss (L_sigreg)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "training_curves_v2.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés dans {out_path}")
    plt.close()

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/training_jepa_v2.csv"
    plot_training(log_file)
