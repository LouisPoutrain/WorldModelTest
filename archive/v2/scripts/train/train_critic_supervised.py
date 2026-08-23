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
from modules.cost import Cost

def main():
    print("🚀 Démarrage de l'entraînement Supervisé du Critique (Geodesic BFS)")
    device = torch.device("cpu")
    
    # 1. Models
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    cost = Cost(latent_dim=latent_dim).to(device)
    
    # 2. Load V2 Perception
    checkpoint_path = "checkpoints/agent_checkpoint_v2.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        print("✅ Modèle Perception V2 chargé.")
    else:
        print("❌ Aucun checkpoint V2 trouvé.")
        return
        
    perception.eval()
    for param in perception.parameters():
        param.requires_grad = False
        
    cost.train()
    optimizer = optim.Adam(cost.parameters(), lr=1e-3)
    
    # 3. Load BFS Dataset
    dataset_path = "dataset/grids_v2_bfs.pt"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset {dataset_path} introuvable.")
        return
        
    grids_dataset = torch.load(dataset_path)
    print(f"✅ Dataset BFS chargé : {len(grids_dataset)} grilles.")
    
    env = GridWorldEnv(size=10, max_energy=100)
    
    # 4. Training loop
    epochs = 20
    batch_size = 256
    steps_per_epoch = 500
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_critic_supervised.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Epoch', 'Loss'])
        
    pbar_epoch = tqdm(range(epochs), desc="Epochs")
    
    for epoch in pbar_epoch:
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            # Batch construction
            batch_obs = []
            batch_target = []
            
            for _ in range(batch_size):
                grid = random.choice(grids_dataset)
                env.obstacles = grid['obstacles']
                env.target_pos = grid['target_pos']
                env.station_pos = grid['station_pos']
                distances = grid['distances']
                
                # Sample a random valid position
                while True:
                    y, x = random.randint(0, 9), random.randint(0, 9)
                    if distances[y, x] < 99.0: # Valid cell
                        break
                        
                env.agent_pos = [y, x]
                obs = env.get_local_observation()
                batch_obs.append(obs)
                batch_target.append(distances[y, x].item())
                
            x_batch = torch.stack(batch_obs).to(device)
            y_batch = torch.tensor(batch_target, dtype=torch.float32, device=device).unsqueeze(1)
            
            with torch.no_grad():
                s_batch = perception(x_batch)
                
            v_preds = cost(s_batch)
            loss = nn.MSELoss()(v_preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / steps_per_epoch
        pbar_epoch.set_postfix(loss=f"{avg_loss:.4f}")
        
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, f"{avg_loss:.4f}"])
            
    # 5. Save
    save_path = "checkpoints/agent_critic_v2_supervised.pth"
    torch.save({'cost': cost.state_dict()}, save_path)
    print(f"🎉 Entraînement terminé ! Checkpoint sauvegardé dans {save_path}")

if __name__ == "__main__":
    main()
