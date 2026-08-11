import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor
from modules.memory import ShortTermMemory

def create_synthetic_target_obs(env, target_type='target'):
    # Creates a 4x10x10 observation where the agent is placed exactly at the goal
    obs = torch.zeros((4, env.size, env.size), dtype=torch.float32)
    
    # Static elements
    for o in env.obstacles:
        obs[0, o[0], o[1]] = 1.0
    obs[1, env.target_pos[0], env.target_pos[1]] = 1.0
    obs[2, env.station_pos[0], env.station_pos[1]] = 1.0
    
    # Agent position
    if target_type == 'target':
        obs[3, env.target_pos[0], env.target_pos[1]] = 1.0
    elif target_type == 'station':
        obs[3, env.station_pos[0], env.station_pos[1]] = 1.0
    return obs

def main():
    print("Initialisation des modules...")
    env = GridWorldEnv(size=10, max_energy=100)
    
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim)
    
    # Target Encoder (Phase 1 - EMA)
    target_encoder = copy.deepcopy(perception)
    for param in target_encoder.parameters():
        param.requires_grad = False
        
    configurator = Configurator(latent_dim=latent_dim)
    # WorldModel now takes z_dim=4 by default
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, z_dim=4)
    cost = Cost(latent_dim=latent_dim)
    actor = Actor(action_dim=4, num_sequences=50, horizon=5, gamma=0.9, cem_iterations=3, elite_size=10)
    memory = ShortTermMemory(capacity=10000)
    
    # Optimizers (Le Critique doit aussi entraîner le CNN pour structurer l'espace latent)
    optimizer_wm = optim.Adam(list(world_model.parameters()) + list(perception.parameters()), lr=1e-4)
    optimizer_critic = optim.Adam(list(cost.critic.parameters()) + list(perception.parameters()), lr=1e-4)
    
    # --- Système de Sauvegarde / Chargement ---
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "agent_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Chargement des poids depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        perception.load_state_dict(checkpoint['perception'])
        target_encoder.load_state_dict(checkpoint['target_encoder'])
        world_model.load_state_dict(checkpoint['world_model'])
        cost.load_state_dict(checkpoint['cost'])
        optimizer_wm.load_state_dict(checkpoint['optimizer_wm'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        print("Modèles restaurés avec succès, l'agent reprend ses connaissances !")
    else:
        print("Aucun modèle existant trouvé. L'agent part de zéro.")
        
    num_episodes = 600
    batch_size = 32
    
    print("Début de l'entraînement...")
    for episode in range(num_episodes):
        # Mettre à jour les représentations latentes des cibles avec le CNN ACTUEL
        with torch.no_grad():
            target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
            station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
            
        x_t = env.reset()
        episode_reward = 0
        
        for step in range(200): # max steps
            # 1. Perception
            x_t_tensor = x_t.unsqueeze(0)
            with torch.no_grad():
                s_t = perception(x_t_tensor)
                
            # 2. Configuration
            s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
            
            import random
            
            # Phase 1 : Pré-entraînement (Exploration pure) pour structurer l'espace latent
            if episode < 500:
                a_t = random.randint(0, 3)
            else:
                # Phase 2 : Planification (Actor Mode-2) avec la boussole spatiale
                a_t, _, _ = actor.plan(
                    s_t, world_model, cost, s_goal, 
                    w_energy, w_collision, w_goal, env.energy
                )
                
                # --- Exploration Epsilon-Greedy (10%) ---
                if random.random() < 0.10:
                    a_t = random.randint(0, 3)
            
            # 4. Exécution
            x_next, reward, done = env.step(a_t)
            
            # 5. Mémoire (Stockage de l'expérience et du reward)
            x_next_tensor = x_next.unsqueeze(0)
            memory.push(x_t_tensor, a_t, x_next_tensor, reward, done)
            x_t = x_next
            
            # 6. Apprentissage (en arrière-plan)
            if len(memory) > batch_size:
                x_batch, a_batch, x_next_batch, reward_batch, done_batch = memory.sample(batch_size)
                a_batch_onehot = F.one_hot(a_batch, num_classes=4).float()
                
                # Repass through main encoder to get fresh s_batch with gradients
                s_batch = perception(x_batch)
                
                # Get target representations from target encoder (no gradients)
                with torch.no_grad():
                    s_next_batch_target = target_encoder(x_next_batch)
                
                # -- Train World Model & Perception --
                optimizer_wm.zero_grad()
                optimizer_critic.zero_grad()
                
                # z is sampled internally by the world_model forward pass
                s_next_pred = world_model(s_batch, a_batch_onehot)
                
                # MSE loss between predicted next state and target encoder's representation
                loss_pred = F.mse_loss(s_next_pred, s_next_batch_target)
                
                # VICReg / SIGReg Variance Loss on the batch to prevent representation collapse
                std = torch.sqrt(s_batch.var(dim=0) + 1e-04)
                loss_var = torch.mean(F.relu(1.0 - std))
                
                loss_wm = loss_pred + 1.0 * loss_var
                
                loss_wm.backward(retain_graph=True)
                
                # -- Train Critic (TD-Learning) --
                step_cost_batch = -reward_batch
                
                with torch.no_grad():
                    V_next_batch = cost(s_next_batch_target)
                
                # Bellman equation: Target Cost = step_cost + gamma * V(s_next) * (1 - done)
                target_value_batch = step_cost_batch + 0.9 * V_next_batch * (1.0 - done_batch)
                
                # On détache s_batch pour que le Critic n'interfère pas avec l'apprentissage du CNN (JEPA pur)
                V_t_batch = cost(s_batch.detach())
                
                loss_critic = F.mse_loss(V_t_batch, target_value_batch)
                loss_critic.backward()
                
                optimizer_wm.step()
                optimizer_critic.step()
                
                # -- EMA Update for Target Encoder --
                tau = 0.01
                for p_tgt, p_main in zip(target_encoder.parameters(), perception.parameters()):
                    p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                
            if done:
                break
                
        print(f"Episode {episode + 1}/{num_episodes} - Steps: {step+1} - Energy left: {env.energy} - Reached Target: {env.agent_pos == env.target_pos}")
        
        # Sauvegarde périodique (tous les 10 épisodes ou à la fin)
        if (episode + 1) % 10 == 0 or (episode + 1) == num_episodes:
            torch.save({
                'perception': perception.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'world_model': world_model.state_dict(),
                'cost': cost.state_dict(),
                'optimizer_wm': optimizer_wm.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict()
            }, checkpoint_path)
            print(f"💾 Poids sauvegardés à l'épisode {episode + 1}.")

if __name__ == "__main__":
    main()
