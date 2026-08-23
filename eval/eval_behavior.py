import torch
import numpy as np
import os
import sys
import csv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.actor import Actor
from modules.cost import Cost
from scripts.analysis.visualize import create_synthetic_target_obs

def generate_u_trap(env):
    env.obstacles = []
    env.agent_pos = [5, 3]
    env.target_pos = [5, 6]
    env.station_pos = [0, 0]
    for i in range(3, 8): env.obstacles.append([i, 5])
    for j in range(5, 8):
        env.obstacles.append([3, j])
        env.obstacles.append([7, j])

def generate_zigzag(env):
    env.obstacles = []
    env.agent_pos = [0, 0]
    env.target_pos = [9, 9]
    env.station_pos = [5, 5]
    for j in range(0, 8): env.obstacles.append([2, j])
    for j in range(2, 10): env.obstacles.append([5, j])
    for j in range(0, 8): env.obstacles.append([8, j])

def setup_id(env):
    env._generate_random_grid()

def evaluate_suite(name, env_setup_func, perception, configurator, actor, world_model, cost, device, csv_writer, num_episodes=50):
    env = GridWorldEnv(size=10, max_energy=100)
    successes = 0
    total_steps = 0
    
    for ep in tqdm(range(num_episodes), desc=f"Test {name}"):
        env.reset()
        env_setup_func(env)
        obs = env.get_local_observation()
        
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0).to(device)
        with torch.no_grad():
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
            
        h_t = world_model.init_hidden(1, device=device)
        
        for step in range(100):
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                s_t = perception(obs_tensor)
                s_goal, _, _, w_goal = configurator.get_configuration(env.energy)
                
                # Calcul de la distance latente réelle actuelle
                dist_latente_reelle = torch.sum((s_t - s_goal)**2).item()
                
                a_t, best_cost, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, 0.0) # w_goal=0.0 pour 100% Critic
                
            old_pos = env.agent_pos.copy()
            obs, reward, done = env.step(a_t)
            
            # Si le reward est -5.0, c'est un mur (collision)
            wall_hit = (reward == -5.0)
            
            # Enregistrement dans le log
            csv_writer.writerow([
                name,
                ep,
                step,
                f"({old_pos[0]},{old_pos[1]})",
                ["HAUT", "BAS", "GAUCHE", "DROITE"][a_t],
                wall_hit,
                f"{dist_latente_reelle:.4f}",
                f"{best_cost:.4f}"
            ])
            
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

def main():
    print("🧪 Évaluation V2 : Collecte de Télémétrie")
    device = torch.device("cpu")
    
    perception = Perception(in_channels=4, latent_dim=32).to(device)
    configurator = Configurator(latent_dim=32)
    world_model = WorldModel(latent_dim=32, action_dim=4, hidden_dim=128).to(device)
    cost = Cost(latent_dim=32).to(device)
    
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50, w_critic=1.0)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = os.path.join(base_dir, "checkpoints", "agent_checkpoint_v2.pth")
    print(f"📦 Checkpoint : {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        
        critic_path = os.path.join(base_dir, "checkpoints", "agent_critic_v2_td.pth")
        if os.path.exists(critic_path):
            cp_critic = torch.load(critic_path, map_location=device)
            cost.load_state_dict(cp_critic['cost'])
            print("✅ Critique V2 chargé.")
        else:
            print("❌ Critique V2 introuvable.")
            return
            
        print("✅ Modèles V2 chargés.")
    else:
        print("❌ Aucun checkpoint.")
        return
        
    perception.eval()
    world_model.eval()
    cost.eval()
    
    # Préparation du CSV
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "eval_metrics_v2.csv")
    
    f_csv = open(csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow([
        "Test_Name", "Episode_ID", "Step", "Agent_Pos", "Action", 
        "Wall_Hit", "Latent_Dist_to_Goal", "CEM_Predicted_Cost"
    ])
    
    print("🚀 Lancement des évaluations...")
    
    sr_id, step_id = evaluate_suite("ID", setup_id, perception, configurator, actor, world_model, cost, device, writer, 100)
    sr_u, step_u = evaluate_suite("U-Trap", generate_u_trap, perception, configurator, actor, world_model, cost, device, writer, 20)
    sr_zig, step_zig = evaluate_suite("ZigZag", generate_zigzag, perception, configurator, actor, world_model, cost, device, writer, 20)
    
    f_csv.close()
    
    print("📊 BILAN DE GÉNÉRALISATION V2 :")
    print(f"| Test                 | Taux de Succès | Pas Moyens |")
    print(f"|----------------------|----------------|------------|")
    print(f"| In-Distribution (ID) | {sr_id:6.1f}%       | {step_id:6.1f}     |")
    print(f"| OOD: Piège en U      | {sr_u:6.1f}%       | {step_u:6.1f}     |")
    print(f"| OOD: Labyrinthe      | {sr_zig:6.1f}%       | {step_zig:6.1f}     |")
    print(f"✅ Télémétrie sauvegardée dans : {csv_path}")

if __name__ == "__main__":
    main()
