import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import SpatialCritic
from modules.actor import HierarchicalActor
from modules.macro_planner import MacroPlanner

def visualize_interactive():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Chargement des modèles sur {device}...")
    
    env = GridWorldEnv(size=20, obstacle_density=0.08)
    
    perception = Perception(in_channels=4, latent_dim=16).to(device)
    world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=20).to(device)
    critic = SpatialCritic(latent_dim=16, hidden_dim=64, spatial_size=20).to(device)
    
    # Chemins relatifs depuis le dossier eval/
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    
    try:
        perception.load_state_dict(torch.load(f'{base_dir}/agent_h_jepa.pth', map_location=device)['perception'])
        world_model.load_state_dict(torch.load(f'{base_dir}/agent_h_jepa.pth', map_location=device)['world_model'])
        critic.load_state_dict(torch.load(f'{base_dir}/agent_critic_td.pth', map_location=device)['critic'])
    except FileNotFoundError as e:
        print(f"❌ Erreur: Impossible de trouver les checkpoints dans {base_dir}")
        return
        
    perception.eval()
    world_model.eval()
    critic.eval()
    
    astar = MacroPlanner()
    hierarchical_actor = HierarchicalActor(perception, critic, astar)
    
    cmap = mcolors.ListedColormap(['#1e293b', '#64748b', '#ef4444', '#eab308', '#38bdf8'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    while True:
        print("\nRecherche d'un nouvel épisode multi-pièces...")
        frames = []
        obs = env.reset()
        
        # Forcer des pièces différentes pour que ce soit intéressant
        start_room = env.get_current_room(env.agent_pos)
        target_room = env.get_current_room(env.target_pos)
        
        if start_room == target_room or start_room == -1 or target_room == -1:
            continue
            
        done = False
        steps = 0
        frames.append(env.render().copy())
        
        while not done and steps < 100:
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(obs_tensor)
                goal_obs = torch.zeros_like(obs_tensor)
                goal_obs[0, 1, env.target_pos[0], env.target_pos[1]] = 1.0
                s_goal = perception(goal_obs)
                
                path = hierarchical_actor.plan(env, s_t, s_goal)
                
                if not path:
                    import random
                    action = random.randint(0, 3)
                else:
                    next_pos = path[0]
                    if next_pos[0] < env.agent_pos[0]: action = 0
                    elif next_pos[0] > env.agent_pos[0]: action = 1
                    elif next_pos[1] < env.agent_pos[1]: action = 2
                    else: action = 3
                    
            obs, reward, done = env.step(action)
            frames.append(env.render().copy())
            steps += 1
            
            if done and reward == 100.0:
                print(f"✅ Épisode réussi en {steps} pas !")
                break
        else:
            print(f"❌ Échec après 100 pas (l'agent s'est perdu).")
            
        print("Affichage de l'animation... (Fermez la fenêtre pour générer le suivant, ou faites Ctrl+C dans le terminal pour quitter)")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(frames[0], cmap=cmap, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        
        title = "Succès" if (done and reward == 100.0) else "Échec"
        ax.set_title(f"H-JEPA 20x20 - {title} ({steps} pas)", color='white')
        fig.patch.set_facecolor('#0f172a')
        
        def update(frame):
            im.set_array(frame)
            return [im]
            
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=150, blit=True)
        plt.show()

if __name__ == "__main__":
    try:
        visualize_interactive()
    except KeyboardInterrupt:
        print("\nArrêt du script de visualisation.")
