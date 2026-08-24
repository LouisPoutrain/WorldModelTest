import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import csv
import time
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.cost import SpatialCritic

def main():
    print("🚀 Démarrage de l'entraînement Supervisé du Critic V3 (Geodesic BFS)")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    latent_dim = 16
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    cost = SpatialCritic(latent_dim=latent_dim, hidden_dim=64).to(device)
    
    checkpoint_path = "checkpoints/agent_h_jepa.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        print("✅ Modèle Perception V3 chargé.")
    else:
        print("❌ Aucun checkpoint V3 trouvé.")
        return
        
    perception.eval()
    for param in perception.parameters():
        param.requires_grad = False
        
    cost.train()
    optimizer = optim.Adam(cost.parameters(), lr=1e-3)
    
    dataset_path = "archive/v2/dataset/grids_v2_bfs.pt"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset {dataset_path} introuvable.")
        return
        
    grids_dataset = torch.load(dataset_path)
    print(f"✅ Dataset BFS chargé : {len(grids_dataset)} grilles.")
    
    env = GridWorldEnv(size=10, max_energy=100)
    
    epochs = 10
    batch_size = 256
    steps_per_epoch = 200
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_critic_v3.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Epoch', 'Loss'])
        
    pbar_epoch = tqdm(range(epochs), desc="Epochs")
    
    for epoch in pbar_epoch:
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            batch_obs_sim = []
            batch_obs_goal = []
            batch_target = []
            
            for _ in range(batch_size):
                grid = random.choice(grids_dataset)
                env.obstacles = grid['obstacles']
                env.target_pos = grid['target_pos']
                env.station_pos = grid['station_pos']
                distances = grid['distances']
                
                while True:
                    y, x = random.randint(0, 9), random.randint(0, 9)
                    if distances[y, x] < 99.0:
                        break
                        
                # 1. Obs Sim
                env.agent_pos = [y, x]
                obs_sim = env.get_local_observation()
                batch_obs_sim.append(obs_sim)
                
                # 2. Obs Goal (Agent is at target_pos)
                env.agent_pos = grid['target_pos']
                obs_goal = env.get_local_observation()
                batch_obs_goal.append(obs_goal)
                
                batch_target.append(distances[y, x].item())
                
            x_sim_batch = torch.stack(batch_obs_sim).to(device)
            x_goal_batch = torch.stack(batch_obs_goal).to(device)
            y_batch = torch.tensor(batch_target, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # On utilise l'espace latent (pas projeté) pour la planification
                s_sim = perception(x_sim_batch)
                s_goal = perception(x_goal_batch)
                
            v_preds = cost(s_sim, s_goal)
            loss = nn.MSELoss()(v_preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / steps_per_epoch
        pbar_epoch.set_postfix(loss=f"{avg_loss:.4f}")
        
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, f"{avg_loss:.4f}"])
            
    save_path = "checkpoints/agent_critic_v3.pth"
    torch.save({'cost': cost.state_dict()}, save_path)
    print(f"🎉 Entraînement terminé ! Checkpoint sauvegardé dans {save_path}")

if __name__ == "__main__":
    main()
