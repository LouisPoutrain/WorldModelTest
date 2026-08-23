"""
Visualize the agent's behavior on a procedurally generated grid.
Saves a video to the project directory.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_synthetic_target_obs(env, target_type='target'):
    obs = torch.zeros((4, env.size, env.size), dtype=torch.float32)
    for o in env.obstacles:
        obs[0, o[0], o[1]] = 1.0
    obs[1, env.target_pos[0], env.target_pos[1]] = 1.0
    obs[2, env.station_pos[0], env.station_pos[1]] = 1.0
    if target_type == 'target':
        obs[3, env.target_pos[0], env.target_pos[1]] = 1.0
    elif target_type == 'station':
        obs[3, env.station_pos[0], env.station_pos[1]] = 1.0
    return obs

def main():
    env = GridWorldEnv(size=10, max_energy=100)

    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=128)
    cost = Cost(latent_dim=latent_dim)
    actor = Actor(action_dim=4, num_sequences=2000, horizon=15, cem_iterations=10, elite_size=100)
    configurator = Configurator(latent_dim=latent_dim)

    checkpoint_path = os.path.join("checkpoints", "agent_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Chargement des poids depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        cost.load_state_dict(checkpoint['cost'])
        print("Modèles restaurés avec succès !")
    else:
        print("Aucun modèle trouvé ! Veuillez d'abord entraîner l'agent avec main.py")
        return
        
    perception.eval()
    world_model.eval()
    cost.eval()

    # Initialize Goals and Hidden State
    x_t = env.reset()
    h_t = world_model.init_hidden(1, device=x_t.device)
    with torch.no_grad():
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
        s_target = perception(target_obs)
        s_station = perception(station_obs)
        configurator.set_goals(s_target, s_station)
    
    print(f"Grille générée:")
    print(f"  Agent: {env.agent_pos}, Cible: {env.target_pos}, Station: {env.station_pos}")
    print(f"  Obstacles: {len(env.obstacles)}")

    # Run episode
    frames = []
    max_steps = 100
    
    for step in range(max_steps):
        grid = env.render()
        frames.append(grid.copy())
        
        x_t_tensor = x_t.unsqueeze(0)
        with torch.no_grad():
            s_t = perception(x_t_tensor)
        
        s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
        
        # Planification pure (distance au goal, pas de critique)
        a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, w_goal)
        
        action_names = ["Haut", "Bas", "Gauche", "Droite"]
        x_next, reward, done = env.step(a_t)
        
        a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float()
        with torch.no_grad():
            _, h_next = world_model.forward_step(s_t, a_t_onehot, h_t)
            
        print(f"  Step {step+1}: {action_names[a_t]} -> {env.agent_pos} (reward={reward:.1f}, energy={env.energy})")
        
        x_t = x_next
        h_t = h_next
        
        if done:
            if env.agent_pos == env.target_pos:
                print(f"  🎯 Cible atteinte en {step+1} pas !")
            else:
                print(f"  💀 Énergie épuisée après {step+1} pas.")
            frames.append(env.render().copy())
            break
    
    # Save video
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.colors as mcolors
        
        cmap = mcolors.ListedColormap(['white', 'black', 'green', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        def animate(i):
            ax.clear()
            ax.imshow(frames[i], cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(f'Step {i+1}/{len(frames)}')
            ax.grid(True, linewidth=0.5, alpha=0.3)
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=300, blit=False)
        
        output_path = "media/agent_run.gif"
        anim.save(output_path, writer='pillow', fps=3)
        plt.close()
        print(f"  📹 Vidéo sauvegardée : {output_path}")
    except Exception as e:
        print(f"  Erreur lors de la sauvegarde vidéo: {e}")

if __name__ == "__main__":
    main()
