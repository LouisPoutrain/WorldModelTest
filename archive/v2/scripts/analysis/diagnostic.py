"""
Diagnostic V3 - Compatible avec la refonte JEPA (sans z, sans critique dans le planning).
"""
import torch
import torch.nn.functional as F
import os
import random

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.configurator import Configurator
from modules.actor import Actor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_obs_at_position(env, row, col):
    obs = torch.zeros((4, env.size, env.size), dtype=torch.float32)
    for o in env.obstacles:
        obs[0, o[0], o[1]] = 1.0
    obs[1, env.target_pos[0], env.target_pos[1]] = 1.0
    obs[2, env.station_pos[0], env.station_pos[1]] = 1.0
    obs[3, row, col] = 1.0
    return obs

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
    env.reset()
    
    latent_dim = 32
    perception = Perception(in_channels=4, latent_dim=latent_dim)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=128)
    cost = Cost(latent_dim=latent_dim)
    configurator = Configurator(latent_dim=latent_dim)
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50)
    
    checkpoint_path = os.path.join("checkpoints", "agent_checkpoint.pth")
    device = torch.device("cpu")
    if not os.path.exists(checkpoint_path):
        print("❌ Aucun checkpoint trouvé!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    perception.load_state_dict(checkpoint['perception'])
    world_model.load_state_dict(checkpoint['world_model'])
    cost.load_state_dict(checkpoint['cost'])
    
    perception.eval()
    world_model.eval()
    cost.eval()
    
    print("=" * 60)
    print("DIAGNOSTIC V3 DE L'AGENT JEPA (Refonte le-wm)")
    print("=" * 60)
    
    # Afficher la grille
    print(f"\n  Grille générée:")
    print(f"    Agent:   {env.agent_pos}")
    print(f"    Cible:   {env.target_pos}")
    print(f"    Station: {env.station_pos}")
    print(f"    Obstacles: {len(env.obstacles)}")
    import numpy as np
    grid = env.render()
    print("    " + "-" * 21)
    for r in range(env.size):
        row_str = "    |"
        for c in range(env.size):
            v = grid[r, c]
            if v == 0: row_str += ". "
            elif v == 1: row_str += "█ "
            elif v == 2: row_str += "G "
            elif v == 3: row_str += "S "
            elif v == 4: row_str += "A "
        row_str += "|"
        print(row_str)
    print("    " + "-" * 21)
    
    # TEST 1: Latent space
    print("\n--- TEST 1: Structure de l'espace latent ---")
    positions = {
        "Agent": tuple(env.agent_pos),
        "Cible": tuple(env.target_pos),
        "Station": tuple(env.station_pos),
    }
    obstacle_set = set(tuple(o) for o in env.obstacles)
    free_cells = [(r,c) for r in range(env.size) for c in range(env.size) 
                  if (r,c) not in obstacle_set and (r,c) not in [tuple(env.agent_pos), tuple(env.target_pos), tuple(env.station_pos)]]
    random.shuffle(free_cells)
    for i, cell in enumerate(free_cells[:3]):
        positions[f"Libre_{cell}"] = cell
    
    latent_vectors = {}
    with torch.no_grad():
        for label, pos in positions.items():
            obs = create_obs_at_position(env, pos[0], pos[1]).unsqueeze(0)
            s = perception(obs)
            latent_vectors[label] = s
            print(f"  {label} {pos}: norm={s.norm().item():.3f}, std={s.std().item():.4f}")
    
    print("\n  Distances latentes:")
    keys = list(latent_vectors.keys())
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j > i:
                d = torch.sum((latent_vectors[k1] - latent_vectors[k2])**2).item()
                p1, p2 = positions[k1], positions[k2]
                geo = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
                print(f"    {k1} <-> {k2}: latent={d:.4f}, manhattan={geo}")
    
    # TEST 2: Collapse check
    print("\n--- TEST 2: Collapse Check ---")
    all_vecs = torch.cat(list(latent_vectors.values()), dim=0)
    var_per_dim = all_vecs.var(dim=0)
    active = (var_per_dim > 0.01).sum().item()
    print(f"  Dimensions actives (var > 0.01): {active}/{latent_dim}")
    if active < latent_dim // 2:
        print("  ⚠️  EFFONDREMENT DÉTECTÉ!")
    else:
        print("  ✅ Espace latent sain")
    
    # TEST 3: World Model
    print("\n--- TEST 3: World Model (RNN, FiLM) ---")
    with torch.no_grad():
        s_agent = latent_vectors["Agent"]
        h_agent = world_model.init_hidden(1, device=s_agent.device)
        action_names = ["Haut", "Bas", "Gauche", "Droite"]
        for a in range(4):
            a_onehot = F.one_hot(torch.tensor([a]), num_classes=4).float()
            s_pred, _ = world_model.forward_step(s_agent, a_onehot, h_agent)
            delta = (s_pred - s_agent).norm().item()
            print(f"    Action {action_names[a]}: delta_norm={delta:.4f}")
    
    # TEST 4: Critique (monitoring)
    print("\n--- TEST 4: Critique (monitoring uniquement) ---")
    with torch.no_grad():
        for label, s in latent_vectors.items():
            v = cost(s).item()
            print(f"  {label}: V(s) = {v:.4f}")
    
    # TEST 5: Planificateur CEM (distance pure)
    print("\n--- TEST 5: Planificateur CEM (distance pure au goal) ---")
    with torch.no_grad():
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
        s_target = perception(target_obs)
        s_station = perception(station_obs)
        configurator.set_goals(s_target, s_station)
    
    s_goal, _, _, w_goal = configurator.get_configuration(100)
    dist_to_goal = torch.sum((latent_vectors["Agent"] - s_goal)**2).item()
    print(f"  Distance Agent -> Goal (latent): {dist_to_goal:.4f}")
    
    s_agent = latent_vectors["Agent"]
    h_agent = world_model.init_hidden(1, device=s_agent.device)
    a, seq, cost_val = actor.plan(s_agent, h_agent, world_model, cost, s_goal, w_goal)
    seq_str = [action_names[s.item()] for s in seq[:8]]
    print(f"  Plan: {seq_str} (cost={cost_val:.4f})")
    
    # TEST 6: Simulation de 20 pas
    print("\n--- TEST 6: Simulation de 20 pas ---")
    x_t = env.reset()
    with torch.no_grad():
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
        s_target = perception(target_obs)
        s_station = perception(station_obs)
        configurator.set_goals(s_target, s_station)
    
    trajectory = [tuple(env.agent_pos)]
    x_t = env.reset()
    h_t = world_model.init_hidden(1, device=x_t.device)
    for step in range(20):
        x_t_tensor = x_t.unsqueeze(0)
        with torch.no_grad():
            s_t = perception(x_t_tensor)
        s_goal, _, _, w_goal = configurator.get_configuration(env.energy)
        a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal, w_goal)
        x_next, reward, done = env.step(a_t)
        
        a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float()
        with torch.no_grad():
            _, h_next = world_model.forward_step(s_t, a_t_onehot, h_t)
            
        x_t = x_next
        h_t = h_next
        trajectory.append(tuple(env.agent_pos))
        print(f"  Step {step+1}: {action_names[a_t]} -> {env.agent_pos} (r={reward:.1f})")
        if done:
            print(f"  🏁 Terminé! Cible: {env.agent_pos == env.target_pos}")
            break
    
    unique = len(set(trajectory))
    print(f"\n  Positions uniques: {unique}/{len(trajectory)}")
    if unique < len(trajectory) * 0.5:
        print("  ⚠️  L'agent tourne en rond!")

if __name__ == "__main__":
    main()
