import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import imageio_ffmpeg

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor
from visualize import create_synthetic_target_obs

class CustomGridWorldEnv(GridWorldEnv):
    def reset(self):
        self.agent_pos = [0, 0]
        self.energy = self.max_energy
        
        # Nouvelle position pour la cible et la station
        self.target_pos = [9, 9]
        self.station_pos = [1, 8]
        
        # Nouvelle configuration de labyrinthe (Un grand mur vertical avec un passage en bas)
        self.obstacles = []
        for i in range(0, 8):
            self.obstacles.append([i, 5])
            
        # Ajouter quelques obstacles aléatoires
        self.obstacles.extend([[8, 2], [7, 2], [2, 2], [2, 3], [2, 8], [3, 8], [4, 8]])
            
        # Ensure target and station are not in obstacles
        self.obstacles = [obs for obs in self.obstacles if obs != self.target_pos and obs != self.station_pos and obs != self.agent_pos]
        
        self.done = False
        return self.get_local_observation()

def main():
    print("Initialisation des modules pour le test de généralisation (Zéro-Shot)...")
    env = CustomGridWorldEnv(size=10, max_energy=100)
    
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim)
    configurator = Configurator(latent_dim=latent_dim)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=128)
    cost = Cost(latent_dim=latent_dim)
    # L'acteur a besoin d'une énorme puissance de calcul pour un horizon de 15
    actor = Actor(action_dim=4, num_sequences=2000, horizon=15, cem_iterations=10, elite_size=100)
    
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

    # Initialize Goals
    with torch.no_grad():
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
        s_target = perception(target_obs)
        s_station = perception(station_obs)
        configurator.set_goals(s_target, s_station)

    frames = []
    energy_history = []
    
    x_t = env.reset()
    h_t = world_model.init_hidden(1, device=x_t.device)
    
    frames.append(env.render())
    energy_history.append(env.energy)
    
    print("Démarrage de l'agent dans le nouvel environnement...")
    for step in range(100):
        x_t_tensor = x_t.unsqueeze(0)
        with torch.no_grad():
            s_t = perception(x_t_tensor)
            
        s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
        
        # Planification pure (distance au goal, pas de critique)
        a_t, _, _ = actor.plan(s_t, h_t, world_model, s_goal)
        
        x_next, reward, done = env.step(a_t)
        
        a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float()
        with torch.no_grad():
            _, h_next = world_model.forward_step(s_t, a_t_onehot, h_t)
            
        x_t = x_next
        h_t = h_next
        frames.append(env.render())
        energy_history.append(env.energy)
        
        if done:
            break
            
    print(f"Épisode terminé en {step+1} étapes. Énergie restante: {env.energy}. Cible atteinte: {env.agent_pos == env.target_pos}")
    
    # --- Création de la vidéo ---
    print("Génération de la vidéo...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    cmap = ListedColormap(['white', 'black', 'green', 'blue', 'red'])
    mat = ax.matshow(frames[0], cmap=cmap, vmin=0, vmax=4)
    
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx):
        mat.set_data(frames[frame_idx])
        ax.set_title(f"Step {frame_idx} | Energy: {energy_history[frame_idx]}")
        return mat,

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)
    
    video_path = "generalization_run.gif"
    anim.save(video_path, writer='pillow', fps=5)
    print(f"Vidéo de généralisation sauvegardée dans : {video_path}")

if __name__ == "__main__":
    main()
