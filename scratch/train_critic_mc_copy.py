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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import Cost

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, target_g):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, target_g)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([x[0] for x in batch])
        targets = torch.tensor([x[1] for x in batch], dtype=torch.float32).unsqueeze(1)
        return states, targets
        
    def __len__(self):
        return len(self.buffer)

def main():
    print("🚀 Entraînement du Critique (Phase 2 - Monte Carlo) avec Buffer & H_t corrigé")
    device = torch.device("cpu")
    
    env = GridWorldEnv(size=10, max_energy=100)
    dataset_path = "dataset/grids_v2.pt"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset introuvable ({dataset_path}).")
        return
    grids_dataset = torch.load(dataset_path)
    print(f"✅ Dataset hors-ligne chargé : {len(grids_dataset)} grilles.")
    
    latent_dim = 32
    action_dim = 4
    
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=128).to(device)
    
    checkpoint_path_v2 = "checkpoints/agent_checkpoint_v2.pth"
    if os.path.exists(checkpoint_path_v2):
        cp = torch.load(checkpoint_path_v2, map_location=device)
        perception.load_state_dict(cp['perception'])
        world_model.load_state_dict(cp['world_model'])
        print(f"✅ Modèles V2 chargés.")
    else:
        print(f"❌ Checkpoint V2 introuvable.")
        return
        
    perception.eval()
    world_model.eval()
    for p in perception.parameters(): p.requires_grad = False
    for p in world_model.parameters(): p.requires_grad = False
    
    cost = Cost(latent_dim=latent_dim).to(device)
    optimizer_critic = optim.Adam(cost.parameters(), lr=1e-3)
    
    buffer = ReplayBuffer(50000)
    batch_size = 256
    
    num_episodes = 5000
    gamma = 0.99
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_critic_v2.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Episode', 'L_critic', 'Success_Rate'])
        
    avg_loss_critic = 0.0
    recent_successes = 0
    train_steps = 0
    
    for episode in range(num_episodes):
        grid_data = random.choice(grids_dataset)
        env.obstacles = grid_data['obstacles']
        env.agent_pos = grid_data['agent_pos'].copy()
        env.target_pos = grid_data['target_pos']
        env.station_pos = grid_data['station_pos']
        env.energy = env.max_energy
        env.done = False
        
        obs = env.get_local_observation()
        
        episode_states = []
        episode_rewards = []
        
        last_action = random.randint(0, 3)
        epsilon = max(0.1, 1.0 - (episode / 2000.0))
        
        # Initialisation du vrai H_t pour l'épisode !
        h_t_env = world_model.init_hidden(1, device)
        
        for step in range(100):
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(obs_tensor)
            
            episode_states.append(s_t.squeeze(0)) # Sauvegarder sans la dim batch
            
            if random.random() < epsilon:
                if random.random() < 0.7:
                    a_t = last_action
                else:
                    a_t = random.randint(0, 3)
            else:
                with torch.no_grad():
                    best_a = 0
                    min_v = float('inf')
                    for a in range(4):
                        a_onehot = F.one_hot(torch.tensor([a]), num_classes=4).float().to(device)
                        s_next, _ = world_model.forward_step(s_t, a_onehot, h_t_env)
                        v_next = cost(s_next).item()
                        if v_next < min_v:
                            min_v = v_next
                            best_a = a
                    a_t = best_a
                    
            last_action = a_t
            obs, reward, done = env.step(a_t)
            
            # Mise à jour du vrai h_t_env avec l'action choisie
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            with torch.no_grad():
                _, h_t_env = world_model.forward_step(s_t, a_t_onehot, h_t_env)
            
            if reward == 100.0:
                step_cost = 0.0
            elif reward == -5.0:
                step_cost = 5.0
            else:
                step_cost = 1.0
                
            episode_rewards.append(step_cost)
            
            if done:
                if env.agent_pos == env.target_pos:
                    recent_successes += 1
                break
                
        # Calcul des retours et ajout au buffer
        G = 0
        for i in reversed(range(len(episode_rewards))):
            G = episode_rewards[i] + gamma * G
            buffer.push(episode_states[i], G)
            
        # Entraînement sur batch
        if len(buffer) > batch_size:
            cost.train()
            optimizer_critic.zero_grad()
            
            s_batch, g_batch = buffer.sample(batch_size)
            s_batch = s_batch.to(device)
            g_batch = g_batch.to(device)
            
            v_preds = cost(s_batch)
            loss_critic = F.mse_loss(v_preds, g_batch)
            
            loss_critic.backward()
            optimizer_critic.step()
            
            avg_loss_critic += loss_critic.item()
            train_steps += 1
        
        if (episode + 1) % 50 == 0:
            n = max(train_steps, 1)
            sr = recent_successes / 50.0
            print(f"Ep {episode+1:4d}/{num_episodes} | L_critic: {avg_loss_critic/n:8.4f} | Succès (Eps={epsilon:.2f}): {sr*100:3.0f}%")
            
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    episode + 1,
                    f"{avg_loss_critic/n:.4f}",
                    f"{sr:.4f}"
                ])
                
            avg_loss_critic = 0.0
            recent_successes = 0
            train_steps = 0
            
        if (episode + 1) % 500 == 0 or (episode + 1) == num_episodes:
            checkpoint_out = "checkpoints/agent_critic_v2.pth"
            torch.save({'cost': cost.state_dict()}, checkpoint_out)
            
if __name__ == "__main__":
    main()
