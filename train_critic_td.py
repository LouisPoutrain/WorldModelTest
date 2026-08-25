import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import random
import csv
import time
from tqdm import tqdm
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import SpatialCritic, intrinsic_cost, update_ema

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, s_t, a_t, s_next, c_t, s_goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s_t, a_t, s_next, c_t, s_goal)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s_t = torch.stack([x[0] for x in batch])
        a_t = torch.stack([x[1] for x in batch])
        s_next = torch.stack([x[2] for x in batch])
        c_t = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        s_goal = torch.stack([x[4] for x in batch])
        return s_t, a_t, s_next, c_t, s_goal
        
    def __len__(self):
        return len(self.buffer)

def main():
    print("🚀 Entraînement du Critique (Phase 3 - TD-Learning / Dyna-Q Imagination pure)")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    env = GridWorldEnv(size=20, max_energy=100)
    dataset_path = "dataset/grids_dungeon.pt"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset introuvable ({dataset_path}).")
        return
    grids_dataset = torch.load(dataset_path)
    print(f"✅ Dataset hors-ligne chargé : {len(grids_dataset)} grilles.")
    
    latent_dim = 16
    action_dim = 4
    
    # H-JEPA utilise in_channels=4 et hidden_dim=32
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32, spatial_size=20).to(device)
    
    checkpoint_path_v2 = "checkpoints/agent_h_jepa.pth"
    if os.path.exists(checkpoint_path_v2):
        cp = torch.load(checkpoint_path_v2, map_location=device)
        perception.load_state_dict(cp['perception'])
        world_model.load_state_dict(cp['world_model'])
        print(f"✅ Modèles JEPA chargés (Perception & World Model).")
    else:
        print(f"❌ Checkpoint {checkpoint_path_v2} introuvable.")
        return
        
    perception.eval()
    world_model.eval()
    for p in perception.parameters(): p.requires_grad = False
    for p in world_model.parameters(): p.requires_grad = False
    
    # Initialisation des critiques (Online et Target)
    online_critic = SpatialCritic(latent_dim=latent_dim, spatial_size=20).to(device)
    target_critic = SpatialCritic(latent_dim=latent_dim, spatial_size=20).to(device)
    target_critic.load_state_dict(online_critic.state_dict())
    for p in target_critic.parameters(): p.requires_grad = False
    
    optimizer = optim.Adam(online_critic.parameters(), lr=3e-4)
    buffer = ReplayBuffer(100000)
    batch_size = 256
    
    num_episodes = 5000
    gamma = 0.95
    tau = 0.01
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_critic_td.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Episode', 'TD_Loss', 'Avg_Intrinsic_Cost'])
        
    avg_td_loss = 0.0
    avg_intrinsic_cost = 0.0
    train_steps = 0
    
    for episode in tqdm(range(num_episodes)):
        # 1. Sélectionner une grille aléatoire
        grid_data = random.choice(grids_dataset)
        
        env.obstacles = grid_data[0].nonzero().tolist()
        
        target_idx = grid_data[1].nonzero()
        env.target_pos = target_idx[0].tolist() if len(target_idx) > 0 else [0, 0]
        
        agent_idx = grid_data[3].nonzero()
        env.agent_pos = agent_idx[0].tolist() if len(agent_idx) > 0 else [0, 0]
        
        # Obtenir l'observation réelle initiale
        # env.get_local_observation() -> [4, 10, 10]. unsqueeze(0) -> [1, 4, 10, 10]
        obs = env.get_local_observation().unsqueeze(0).to(device)
        
        # Obtenir l'observation de la cible (idéale) pour s_goal
        old_pos = env.agent_pos.copy()
        env.agent_pos = env.target_pos.copy()
        target_obs = env.get_local_observation().unsqueeze(0).to(device)
        env.agent_pos = old_pos # Restore
        
        with torch.no_grad():
            s_t = perception(obs).squeeze(0) # [16, 10, 10]
            s_goal = perception(target_obs).squeeze(0) # [16, 10, 10]
            
        h_t = world_model.init_hidden(1, device)
        
        # Imagination pure (Dyna-Q) sur H pas
        H = 50
        episode_ic = 0.0
        
        for step in range(H):
            # Politique exploratoire aléatoire
            a_t = random.randint(0, 3)
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            
            # Prédire le futur
            with torch.no_grad():
                s_next, h_next = world_model.forward_step(s_t.unsqueeze(0), a_t_onehot, h_t)
                s_next = s_next.squeeze(0)
            
            # Calcul du Coût Intrinsèque en dur
            c_t = intrinsic_cost(s_t.unsqueeze(0), s_next.unsqueeze(0), s_goal.unsqueeze(0)).item()
            
            # Stockage de la transition imaginée
            buffer.push(s_t, a_t_onehot, s_next, c_t, s_goal)
            
            episode_ic += c_t
            
            # Mise à jour pour le prochain pas imaginaire
            s_t = s_next
            h_t = h_next
            
        # 2. Entraînement TD-Learning (Fitted Value Iteration)
        if len(buffer) > batch_size:
            online_critic.train()
            
            # On fait 10 updates par épisode
            for _ in range(10):
                # On a juste besoin des états pour Value Iteration !
                s_batch, _, _, _, s_goal_batch = buffer.sample(batch_size)
                
                # --- BELLMAN OPTIMALITY (Value Iteration) ---
                # V*(s) = min_a [ c(s,a) + gamma * V_target(s') ]
                
                with torch.no_grad():
                    # On va évaluer les 4 actions pour chaque état du batch
                    best_target_vals = torch.full((batch_size,), float('inf'), device=device)
                    
                    # On crée un hidden state neutre pour l'imagination 1-step
                    h_batch = world_model.init_hidden(batch_size, device)
                    
                    for a in range(4):
                        a_onehot = F.one_hot(torch.full((batch_size,), a, device=device), num_classes=4).float()
                        
                        s_next, _ = world_model.forward_step(s_batch, a_onehot, h_batch)
                        
                        # Coût intrinsèque pour cette action
                        c_a = intrinsic_cost(s_batch, s_next, s_goal_batch)
                        
                        # Valeur future
                        v_next = target_critic(s_next, s_goal_batch)
                        
                        # Valeur totale pour cette action
                        q_val = c_a + gamma * v_next
                        
                        # Prendre le min
                        best_target_vals = torch.min(best_target_vals, q_val)
                        
                    td_target = best_target_vals
                    
                # Prédiction = V_online(s_t)
                v_pred = online_critic(s_batch, s_goal_batch)
                
                loss = F.mse_loss(v_pred, td_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # EMA Update
                update_ema(target_critic, online_critic, tau)
                
                avg_td_loss += loss.item()
                train_steps += 1
                
        avg_intrinsic_cost += (episode_ic / H)
        
        if (episode + 1) % 50 == 0:
            n = max(train_steps, 1)
            # tqdm will overlap print, but it's fine for simple logging
            tqdm.write(f"Ep {episode+1:4d}/{num_episodes} | TD_Loss: {avg_td_loss/n:8.4f} | Avg IC: {avg_intrinsic_cost/50:8.4f}")
            
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    episode + 1,
                    f"{avg_td_loss/n:.4f}",
                    f"{avg_intrinsic_cost/50:.4f}"
                ])
                
            avg_td_loss = 0.0
            avg_intrinsic_cost = 0.0
            train_steps = 0
            
        if (episode + 1) % 500 == 0 or (episode + 1) == num_episodes:
            checkpoint_out = "checkpoints/agent_critic_td.pth"
            torch.save({'critic': online_critic.state_dict()}, checkpoint_out)
            
if __name__ == "__main__":
    main()
