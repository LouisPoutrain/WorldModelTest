import torch
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor
from visualize import create_synthetic_target_obs

def generate_u_trap(env):
    """Créer un piège en U autour de la cible."""
    env.obstacles = []
    # Agent au centre
    env.agent_pos = [5, 3]
    # Cible juste à côté, mais entourée de murs
    env.target_pos = [5, 6]
    env.station_pos = [0, 0] # Loin
    
    # Mur en U ouvert vers la droite
    for i in range(3, 8):
        env.obstacles.append([i, 5]) # Mur gauche du U
    for j in range(5, 8):
        env.obstacles.append([3, j]) # Mur haut du U
        env.obstacles.append([7, j]) # Mur bas du U

def generate_zigzag(env):
    """Labyrinthe en Zig-Zag."""
    env.obstacles = []
    env.agent_pos = [0, 0]
    env.target_pos = [9, 9]
    env.station_pos = [5, 5]
    
    for j in range(0, 8):
        env.obstacles.append([2, j])
    for j in range(2, 10):
        env.obstacles.append([5, j])
    for j in range(0, 8):
        env.obstacles.append([8, j])

def evaluate_suite(name, env_setup_func, perception, configurator, actor, world_model, cost, device, num_episodes=50):
    env = GridWorldEnv(size=10, max_energy=100)
    successes = 0
    total_steps = 0
    
    for ep in tqdm(range(num_episodes), desc=f"Test {name}"):

        env.reset()
        env_setup_func(env) # Forcer le setup OOD ou ID
        obs = env.get_local_observation()
        
        # Définir les buts pour le configurator
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0).to(device)
        with torch.no_grad():
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
            
        # Init hidden state
        h_t = world_model.init_hidden(1, device=device)
        
        for step in range(100): # Max 100 steps
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(obs_tensor)
                
                # Configuration (using the Configurator signature)
                s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(env.energy)
                
                # Planification
                a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, w_goal)
                
            obs, reward, done = env.step(a_t)
            
            import torch.nn.functional as F
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            with torch.no_grad():
                _, h_t = world_model.forward_step(s_t, a_t_onehot, h_t)
            
            if done:
                if env.agent_pos == env.target_pos:
                    successes += 1
                total_steps += (step + 1)
                break
        
        if not done:
            total_steps += 100
            
    success_rate = (successes / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    
    return success_rate, avg_steps

def setup_id(env):
    env._generate_random_grid()

def main():
    print("🧪 Évaluation 3 : Comportement (In-Distribution vs OOD)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    perception = Perception(in_channels=4, latent_dim=32).to(device)
    configurator = Configurator(latent_dim=32)
    world_model = WorldModel(latent_dim=32, action_dim=4, hidden_dim=128).to(device)
    cost = Cost(latent_dim=32).to(device)
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "agent_critic_nstep.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        cost.load_state_dict(checkpoint['cost'])
        print("✅ Modèles chargés.")
    else:
        print("❌ Aucun checkpoint.")
        return
        
    perception.eval()
    world_model.eval()
    cost.eval()
    
    print("🚀 Lancement des évaluations...")
    
    print("1. Test In-Distribution (Grilles aléatoires standard)...")
    sr_id, step_id = evaluate_suite("ID", setup_id, perception, configurator, actor, world_model, cost, device, 100)
    
    print("2. Test OOD : Le Piège en U...")
    sr_u, step_u = evaluate_suite("U-Trap", generate_u_trap, perception, configurator, actor, world_model, cost, device, 20)
    
    print("3. Test OOD : Le Labyrinthe (Zig-Zag)...")
    sr_zig, step_zig = evaluate_suite("ZigZag", generate_zigzag, perception, configurator, actor, world_model, cost, device, 20)
    
    print("📊 BILAN DE GÉNÉRALISATION :")
    print(f"| Test                 | Taux de Succès | Pas Moyens |")
    print(f"|----------------------|----------------|------------|")
    print(f"| In-Distribution (ID) | {sr_id:6.1f}%       | {step_id:6.1f}     |")
    print(f"| OOD: Piège en U      | {sr_u:6.1f}%       | {step_u:6.1f}     |")
    print(f"| OOD: Labyrinthe      | {sr_zig:6.1f}%       | {step_zig:6.1f}     |")
    
    print("➔ DIAGNOSTIC :")
    if sr_u > 50 and sr_zig > 50:
        print("L'agent est capable de navigation spatiale complexe et d'évitement à long terme.")
    elif sr_u <= 50 and sr_id > 70:
        print("L'agent souffre de myopie (il fonce vers la cible au lieu de contourner les murs). Le Critique manque d'entraînement.")

if __name__ == "__main__":
    main()
