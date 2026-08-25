import torch
import sys
import os
import random
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import SpatialCritic
from modules.actor import Actor, HierarchicalActor
from modules.macro_planner import MacroPlanner

def evaluate(agent_type="h-jepa", num_episodes=10, max_steps=100):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Évaluation du Donjon 20x20 avec {agent_type.upper()} sur {device} ---")
    
    env = GridWorldEnv(size=20, obstacle_density=0.08)
    
    # Models
    perception = Perception(in_channels=4, latent_dim=16).to(device)
    world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=20).to(device)
    critic = SpatialCritic(latent_dim=16, hidden_dim=64, spatial_size=20).to(device)
    
    try:
        perception.load_state_dict(torch.load('checkpoints/agent_h_jepa.pth', map_location=device)['perception'])
        world_model.load_state_dict(torch.load('checkpoints/agent_h_jepa.pth', map_location=device)['world_model'])
        critic.load_state_dict(torch.load('checkpoints/agent_critic_td.pth', map_location=device)['critic'])
    except Exception as e:
        print(f"Avertissement: Modèles non chargés (normal si l'entraînement 20x20 n'a pas encore été fait). {e}")
        
    perception.eval()
    world_model.eval()
    critic.eval()
    
    astar = MacroPlanner() # Modifié pour accepter get_path(start, goal, env)
    cem_actor = Actor(action_dim=4, num_sequences=50, horizon=5, cem_iterations=2, elite_size=10)
    hierarchical_actor = HierarchicalActor(perception, critic, astar)
    
    successes = 0
    total_steps = 0
    
    for ep in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if agent_type == "astar_bridé":
                # A* essaie de planifier jusqu'au bout, mais est bridé par la pièce courante
                path = astar.get_path(env.agent_pos, env.target_pos, env)
                if not path:
                    # Impossible de trouver un chemin
                    action = random.randint(0, 3)
                else:
                    # Prendre la prochaine action
                    next_pos = path[0]
                    if next_pos[0] < env.agent_pos[0]: action = 0
                    elif next_pos[0] > env.agent_pos[0]: action = 1
                    elif next_pos[1] < env.agent_pos[1]: action = 2
                    else: action = 3
            
            elif agent_type == "cem":
                obs_tensor = obs.unsqueeze(0).to(device)
                with torch.no_grad():
                    s_t = perception(obs_tensor)
                    # S_goal est fictif, l'acteur va extraire la coordonnée via le decode_agent
                    # Pour simplifier, on prend l'obs avec le goal
                    goal_obs = torch.zeros_like(obs_tensor)
                    goal_obs[0, 3, env.target_pos[0], env.target_pos[1]] = 1.0 # Le goal devient la cible
                    s_goal = perception(goal_obs)
                    action, _, _ = cem_actor.plan(s_t, None, world_model, critic, s_goal, 1.0)
                    
            elif agent_type == "h-jepa":
                # L'acteur hiérarchique utilise le critique pour choisir la porte, puis A* pour y aller
                with torch.no_grad():
                    obs_tensor = obs.unsqueeze(0).to(device)
                    s_t = perception(obs_tensor)
                    
                    goal_obs = torch.zeros_like(obs_tensor)
                    goal_obs[0, 3, env.target_pos[0], env.target_pos[1]] = 1.0
                    s_goal = perception(goal_obs)
                    
                    path = hierarchical_actor.plan(env, s_t, s_goal)
                    
                    if not path:
                        action = random.randint(0, 3)
                    else:
                        next_pos = path[0]
                        if next_pos[0] < env.agent_pos[0]: action = 0
                        elif next_pos[0] > env.agent_pos[0]: action = 1
                        elif next_pos[1] < env.agent_pos[1]: action = 2
                        else: action = 3
            
            obs, reward, done = env.step(action)
            steps += 1
            
            if reward == 100.0:
                successes += 1
                break
                
        total_steps += steps
        
    print(f"Taux de Succès ({agent_type}) : {successes / num_episodes * 100:.1f}%")
    print(f"Pas Moyens : {total_steps / num_episodes:.1f}\n")

if __name__ == "__main__":
    evaluate("astar_bridé", num_episodes=50)
    evaluate("cem", num_episodes=50)
    evaluate("h-jepa", num_episodes=50)
