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
            pass
            
        self.done = False
        return self.get_local_observation()

def main():
    print("🚀 Entraînement du Critique par Monte Carlo Returns (pas de bootstrap)")
    device = torch.device("cpu")
    
    # 1. Initialiser
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=128).to(device)
    cost = Cost(latent_dim=latent_dim).to(device)
    
    configurator = Configurator(latent_dim=latent_dim)
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50)
    
    # 2. Charger les poids (Perception + WorldModel seulement)
    checkpoint_path = "checkpoints/agent_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        print("✅ Perception + WorldModel chargés (Critique part de zéro).")
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
    
    optimizer = torch.optim.Adam(cost.parameters(), lr=3e-4)
    env = MixedGridWorldEnv(size=10, max_energy=100)
    
    # Hyperparamètres
    num_episodes = 10000
    gamma = 0.90
    epsilon = 0.5  # 50% aléatoire pour explorer les pièges
    
    losses = []
    successes = 0
    
    # Replay buffer pour stocker des (s_t, G_t) pré-calculés
    mc_buffer = []
    mc_buffer_capacity = 200000
    batch_size = 256
    updates_per_episode = 4
    
    pbar = tqdm(range(num_episodes), desc="Entraînement MC")
    for ep in pbar:
        obs = env.reset()
        h_t = world_model.init_hidden(1, device=device)
        
        # Set goals
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0).to(device)
        with torch.no_grad():
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
        
        # Collecter une trajectoire complète
        episode_states = []  # Les états latents
        episode_costs = []   # Les coûts instantanés (-reward)
        
        for step in range(100):
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(obs_tensor)
                episode_states.append(s_t.cpu())
            
            s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
            
            # Epsilon-Greedy CEM
            if random.random() < epsilon:
                a_t = random.randint(0, 3)
            else:
                a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, w_goal)
            
            obs, reward, done = env.step(a_t)
            
            # Coût instantané = négatif de la récompense (le CEM minimise)
            # Récompenses shapées pour créer des montagnes d'énergie
            if reward == -5.0:
                instant_cost = 20.0
            elif reward == 100.0:
                instant_cost = -50.0
            elif reward == 10.0:
                instant_cost = -5.0
            else:
                instant_cost = -float(reward)
            episode_costs.append(instant_cost)
            
            # Update hidden state
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            with torch.no_grad():
                _, h_t = world_model.forward_step(s_t, a_t_onehot, h_t)
            
            if done:
                if env.agent_pos == env.target_pos:
                    successes += 1
                break
        
        # ===== MONTE CARLO : Calculer les retours G_t en remontant le temps =====
        T = len(episode_costs)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = episode_costs[t] + gamma * G
            returns[t] = G
        
        # Stocker dans le buffer MC
        for t in range(T):
            mc_buffer.append((episode_states[t], returns[t]))
        
        # Garder le buffer à taille raisonnable
        if len(mc_buffer) > mc_buffer_capacity:
            mc_buffer = mc_buffer[-mc_buffer_capacity:]
        
        # ===== Entraîner le Critique sur le buffer MC =====
        if len(mc_buffer) >= batch_size:
            ep_losses = []
            for _ in range(updates_per_episode):
                batch = random.sample(mc_buffer, batch_size)
                s_batch = torch.cat([b[0] for b in batch], dim=0).to(device)  # (B, 32)
                g_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32).to(device)  # (B,)
                
                v_pred = cost(s_batch).squeeze(1)  # (B,)
                loss = F.mse_loss(v_pred, g_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cost.parameters(), 1.0)
                optimizer.step()
                
                ep_losses.append(loss.item())
            
            losses.append(np.mean(ep_losses))
        
        # Logging
        if len(losses) > 0:
            recent_loss = np.mean(losses[-100:])
            sr = successes / (ep + 1) * 100
            pbar.set_postfix(loss=f"{recent_loss:.2f}", sr=f"{sr:.0f}%", buf=f"{len(mc_buffer)//1000}k")
    
    # Sauvegarde
    save_path = "checkpoints/agent_critic_mc.pth"
    torch.save({
        'perception': perception.state_dict(),
        'world_model': world_model.state_dict(),
        'cost': cost.state_dict()
    }, save_path)
    
    # Graphique
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    def smooth(scalars, weight=0.95):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            s = last * weight + (1 - weight) * point
            smoothed.append(s)
            last = s
        return smoothed
    
    if len(losses) > 0:
        plt.subplot(1, 1, 1)
        plt.plot(smooth(losses), color="#8b5cf6", linewidth=2)
        plt.title("Loss du Critique (Monte Carlo Returns)")
        plt.xlabel("Épisodes")
        plt.ylabel("MSE")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.yscale('log')
    
    os.makedirs("media", exist_ok=True)
    plt.savefig("media/critic_mc_loss.png", dpi=300, bbox_inches='tight')
    print(f"\n📈 Graphique sauvegardé dans media/critic_mc_loss.png")
    print(f"🎉 Entraînement terminé ! Checkpoint : {save_path}")
    print(f"📊 Taux de succès final (exploration ε=0.5) : {successes/num_episodes*100:.1f}%")

if __name__ == "__main__":
    main()
