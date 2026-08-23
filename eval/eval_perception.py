import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ajouter le parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception

def main():
    print("🧪 Évaluation 1 : Perception (Linear Probing)")
    device = torch.device("cpu")
    
    # 1. Charger le modèle
    perception = Perception(in_channels=4, grid_size=10, embed_dim=64, latent_dim=32).to(device)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "perception_jepa.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        print("✅ Poids de la Perception chargés.")
    else:
        print("❌ Aucun checkpoint trouvé. Lancez d'abord l'entraînement.")
        return
        
    perception.eval()
    
    # 2. Générer le Dataset
    print("Génération du dataset de probing (5000 grilles)...")
    env = GridWorldEnv(size=10, max_energy=100)
    
    X = []
    Y = []
    
    with torch.no_grad():
        for _ in tqdm(range(5000), desc="Génération Observations"):

            obs = env.reset()
            # On veut prédire la position (x, y) de l'agent
            y_agent, x_agent = env.agent_pos
            
            obs_tensor = obs.unsqueeze(0).to(device)
            s_t = perception(obs_tensor).cpu().numpy().flatten()
            
            X.append(s_t)
            Y.append([x_agent, y_agent])
            
    X = np.array(X)
    Y = np.array(Y)
    
    # Split Train/Test (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # 3. Linear Probing
    print("Entraînement de la couche linéaire (Sondage)...")
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    
    # 4. Évaluation
    Y_pred = reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    print(f"📊 RÉSULTATS DU LINEAR PROBING :")
    print(f"   - R² Score (Proche de 1.0 = Parfait) : {r2:.4f}")
    print(f"   - Erreur MSE (Distance moyenne)      : {mse:.4f}")
    
    if r2 > 0.90:
        print("   ➔ DIAGNOSTIC : EXCELLENT ! L'encodeur a structuré l'espace latent parfaitement.")
    elif r2 > 0.70:
        print("   ➔ DIAGNOSTIC : CORRECT. L'encodeur comprend la géométrie globale mais manque de précision.")
    else:
        print("   ➔ DIAGNOSTIC : ÉCHEC. L'espace latent est chaotique (Latent Collapse probable).")
        
    # 5. Visualisation
    os.makedirs("media", exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    # Plot X coordinate
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test[:, 0], Y_pred[:, 0], alpha=0.5, color="#38bdf8")
    plt.plot([0, 9], [0, 9], "r--")
    plt.title("Prédiction Coordonnée X")
    plt.xlabel("X Réel")
    plt.ylabel("X Prédit (à partir de s_t)")
    
    # Plot Y coordinate
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test[:, 1], Y_pred[:, 1], alpha=0.5, color="#10b981")
    plt.plot([0, 9], [0, 9], "r--")
    plt.title("Prédiction Coordonnée Y")
    plt.xlabel("Y Réel")
    plt.ylabel("Y Prédit (à partir de s_t)")
    
    plt.tight_layout()
    plt.savefig("media/perception_probing.png", dpi=300)
    print("📈 Graphique sauvegardé dans media/perception_probing.png")

if __name__ == "__main__":
    main()
