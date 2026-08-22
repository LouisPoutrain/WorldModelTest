import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import copy
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor
from modules.memory import ShortTermMemory
from visualize import create_synthetic_target_obs

class MixedGridWorldEnv(GridWorldEnv):
    def reset(self):
        super().reset()
        
        # 50% de chance d'avoir un environnement OOD
        rand = random.random()
        if rand < 0.25:
            # U-Trap
            self.obstacles = []
            self.agent_pos = [5, 3]
            self.target_pos = [5, 6]
            self.station_pos = [0, 0]
            for i in range(3, 8):
                self.obstacles.append([i, 5])
            for j in range(5, 8):
                self.obstacles.append([3, j])
                self.obstacles.append([7, j])
        elif rand < 0.50:
            # ZigZag
            self.obstacles = []
            self.agent_pos = [0, 0]
            self.target_pos = [9, 9]
            self.station_pos = [5, 5]
            for j in range(0, 8):
                self.obstacles.append([2, j])
            for j in range(2, 10):
                self.obstacles.append([5, j])
            for j in range(0, 8):
                self.obstacles.append([8, j])
        else:
            # Random ID grid (déjà fait par super().reset())
            # On s'assure juste que c'est propre
            pass
            
        self.done = False
        return self.get_local_observation()

def main():
    print("🚀 Démarrage de l'entraînement isolé du Critique (N-Step TD)")
    device = torch.device("cpu")
    
    # 1. Initialiser
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=128).to(device)
    cost = Cost(latent_dim=latent_dim).to(device)
    
    configurator = Configurator(latent_dim=latent_dim)
    # Actor utilisant les paramètres de base stables
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50)
    
    # 2. Charger les poids
    checkpoint_path = "checkpoints/agent_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        # cost.load_state_dict(checkpoint['cost']) # COMMENTÉ : Le critique doit réapprendre de zéro !
        print("✅ Modèles de base chargés.")
    else:
        print("❌ Aucun checkpoint trouvé.")
        return
        
    # 3. Geler la Perception et le World Model
    for param in perception.parameters():
        param.requires_grad = False
    for param in world_model.parameters():
        param.requires_grad = False
        
    perception.eval()
    world_model.eval()
    cost.train()
    
    # Target Cost pour stabiliser le TD-Learning
    target_cost = copy.deepcopy(cost).to(device)
    target_cost.eval()
    for param in target_cost.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(cost.parameters(), lr=1e-3)
    memory = ShortTermMemory(capacity=50000)
    env = MixedGridWorldEnv(size=10, max_energy=100)
    
    # Hyperparamètres
    num_episodes = 5000
    batch_size = 128
    gamma = 0.90
    epsilon = 0.5  # 50% aléatoire pour forcer l'exploration des pièges
    seq_len = 3    # N-Step TD avec N=3
    tau = 0.005    # Soft update lent (on part de zéro)
    
    total_steps = 0
    losses = []
    
    pbar = tqdm(range(num_episodes), desc="Entraînement Critique")
    for ep in pbar:
        obs = env.reset()
        x_t = obs
        h_t = world_model.init_hidden(1, device=device)
        
        # Set goals pour le configurator
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0).to(device)
        with torch.no_grad():
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
            
        for step in range(100):
            x_t_tensor = x_t.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(x_t_tensor)
                
            s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
            
            # Epsilon-Greedy CEM
            if random.random() < epsilon:
                a_t = random.randint(0, 3)
            else:
                a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, w_goal)
                
            x_next, reward, done = env.step(a_t)
            
            # Mise à jour du hidden state pour le prochain plan
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            with torch.no_grad():
                _, h_t = world_model.forward_step(s_t, a_t_onehot, h_t)
                
            # Clipper les rewards pour stabiliser le TD-Learning
            clipped_reward = max(min(float(reward), 10.0), -5.0)
            
            # Stockage
            memory.push(x_t_tensor, a_t, x_next.unsqueeze(0), clipped_reward, done)
            x_t = x_next
            total_steps += 1
            
            # Entraînement
            if len(memory) > batch_size * seq_len:
                # N-Step TD Learning (N=3)
                try:
                    x_0, a_seq, x_next_seq, r_seq, d_seq = memory.sample_sequences(batch_size, seq_len=seq_len)
                except ValueError:
                    continue # Pas assez de séquences valides
                    
                x_0 = x_0.to(device)
                x_next_seq = x_next_seq.to(device)
                r_seq = r_seq.to(device)
                d_seq = d_seq.to(device)
                
                with torch.no_grad():
                    s_0 = perception(x_0)
                    
                    # Calcul de la cible N-step
                    # INVERSION DES REWARDS : Le planificateur minimise le Critic, 
                    # donc le Critic doit apprendre le COÛT (négatif de la récompense)
                    target_value = -r_seq[:, 0]
                    discount = gamma
                    
                    for i in range(1, seq_len):
                        not_done_prev = (1.0 - d_seq[:, i-1])
                        target_value += discount * (-r_seq[:, i]) * not_done_prev
                        discount *= gamma
                        
                    not_done_last = (1.0 - d_seq[:, seq_len-1])
                    # État à t+3
                    s_next_n = perception(x_next_seq[:, seq_len-1])
                    v_next = target_cost(s_next_n).squeeze(1)
                    
                    target_value += discount * v_next * not_done_last
                    
                # Forward Critique
                v_0 = cost(s_0).squeeze(1)
                
                loss = F.mse_loss(v_0, target_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                # Soft update Target Cost
                for target_param, param in zip(target_cost.parameters(), cost.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    
            if done:
                break
                
        # Logging temps réel
        if len(losses) > 0:
            pbar.set_postfix(loss=f"{np.mean(losses[-100:]):.4f}")
            
    # Sauvegarde
    save_path = "checkpoints/agent_critic_nstep.pth"
    torch.save({
        'perception': perception.state_dict(),
        'world_model': world_model.state_dict(),
        'cost': cost.state_dict()
    }, save_path)
    
    # Génération du graphique de la Loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    # Lisser la courbe
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
        
    if len(losses) > 0:
        plt.plot(smooth(losses), color="#8b5cf6", linewidth=2)
        plt.title("Évolution de la Loss du Critique (N-Step TD)")
        plt.xlabel("Mises à jour (Steps)")
        plt.ylabel("Erreur MSE")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        
        os.makedirs("media", exist_ok=True)
        plt.savefig("media/critic_loss.png", dpi=300)
        print("📈 Graphique de la Loss sauvegardé dans media/critic_loss.png")

    print(f"🎉 Entraînement terminé ! Checkpoint sauvegardé dans {save_path}")

if __name__ == "__main__":
    main()
