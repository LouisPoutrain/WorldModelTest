import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel

def main():
    print("🧪 Évaluation 2 : Dynamiques (Rollout Drift)")
    device = torch.device("cpu")
    
    # 1. Charger les modèles
    perception = Perception(in_channels=4, latent_dim=32).to(device)
    world_model = WorldModel(latent_dim=32, action_dim=4, hidden_dim=128).to(device)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "agent_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        print("✅ Poids de la Perception et du World Model chargés.")
    else:
        print("❌ Aucun checkpoint trouvé.")
        return
        
    perception.eval()
    world_model.eval()
    
    env = GridWorldEnv(size=10, max_energy=100)
    
    horizon_test = 10
    num_episodes = 500
    
    # Stocker les erreurs pour chaque pas t dans le futur
    mse_per_step = {t: [] for t in range(1, horizon_test + 1)}
    
    print(f"Début des rollouts (100 trajectoires de {horizon_test} pas)...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc="Rollouts (Dynamique)"):

            obs = env.reset()
            obs_tensor = obs.unsqueeze(0).to(device)
            s_t_real = perception(obs_tensor)
            
            # L'état imaginé démarre sur le vrai état
            s_t_imagined = s_t_real.clone()
            
            # État caché du RNN du WM
            h_t = None 
            
            for t in range(1, horizon_test + 1):
                # On choisit une action réelle aléatoire (ou diriger vers la cible, mais l'aléatoire teste bien la physique)
                action = np.random.randint(0, 4)
                a_tensor = torch.zeros(1, 4).to(device)
                a_tensor[0, action] = 1.0
                
                # Le vrai environnement avance
                obs_next, reward, done = env.step(action)
                obs_next_tensor = obs_next.unsqueeze(0).to(device)
                
                # Le vrai encodage du futur (Target)
                s_next_real = perception(obs_next_tensor)
                
                # Le World Model imagine le futur à partir de son PROPRE futur imaginé (Rollout)
                if h_t is None:
                    h_t = world_model.init_hidden(1, device=device)
                
                # a_tensor est [1, 4] et s_t_imagined est [1, 32]
                if len(s_t_imagined.shape) == 1:
                    s_t_imagined = s_t_imagined.unsqueeze(0)
                
                s_next_imagined, h_t = world_model.forward_step(s_t_imagined, a_tensor, h_t)
                
                # Calcul de l'erreur MSE pour le pas t
                mse = F.mse_loss(s_next_imagined, s_next_real).item()
                mse_per_step[t].append(mse)
                
                # On continue d'imaginer à partir de l'hallucination ! C'est ça le Rollout Drift.
                s_t_imagined = s_next_imagined
                
                if done:
                    break
                    
    # Calcul des moyennes
    avg_mse = [np.mean(mse_per_step[t]) for t in range(1, horizon_test + 1)]
    
    print(f"📊 RÉSULTATS DU ROLLOUT DRIFT :")
    print(f"   - Erreur à t+1  : {avg_mse[0]:.5f}")
    print(f"   - Erreur à t+5  : {avg_mse[4]:.5f}")
    print(f"   - Erreur à t+10 : {avg_mse[9]:.5f}")
    
    if avg_mse[9] < avg_mse[0] * 3:
        print("   ➔ DIAGNOSTIC : EXCELLENT ! Le World Model reste très stable sur 10 pas de prédiction.")
    elif avg_mse[9] < avg_mse[0] * 10:
        print("   ➔ DIAGNOSTIC : CORRECT. L'erreur s'accumule mais reste exploitable par le CEM.")
    else:
        print("   ➔ DIAGNOSTIC : DÉRIVE SÉVÈRE (Drift). L'agent planifie dans le brouillard après 5 pas.")
        
    # Visualisation
    os.makedirs("media", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, horizon_test + 1), avg_mse, marker='o', color="#f43f5e", linewidth=2)
    plt.title("Dérive du World Model (Rollout Drift)")
    plt.xlabel("Pas de prédiction dans le futur (t)")
    plt.ylabel("Erreur Moyenne (MSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("media/dynamics_drift.png", dpi=300)
    print("📈 Graphique sauvegardé dans media/dynamics_drift.png")

if __name__ == "__main__":
    main()
