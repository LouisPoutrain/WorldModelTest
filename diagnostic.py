"""
Diagnostic script to understand why the agent fails.
Tests: latent space structure, world model predictions, cost function, and planner.
"""
import torch
import torch.nn.functional as F
import os
import numpy as np

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.configurator import Configurator
from modules.actor import Actor

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

def create_obs_at_position(env, row, col):
    """Create an observation with the agent at a specific (row, col) position."""
    obs = torch.zeros((4, env.size, env.size), dtype=torch.float32)
    for o in env.obstacles:
        obs[0, o[0], o[1]] = 1.0
    obs[1, env.target_pos[0], env.target_pos[1]] = 1.0
    obs[2, env.station_pos[0], env.station_pos[1]] = 1.0
    obs[3, row, col] = 1.0
    return obs

def main():
    env = GridWorldEnv(size=10, max_energy=100)
    latent_dim = 32
    
    perception = Perception(in_channels=4, latent_dim=latent_dim)
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, z_dim=4)
    cost = Cost(latent_dim=latent_dim)
    configurator = Configurator(latent_dim=latent_dim)
    actor = Actor(action_dim=4, num_sequences=50, horizon=5, gamma=0.9, cem_iterations=3, elite_size=10)
    
    checkpoint_path = os.path.join("checkpoints", "agent_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print("❌ Aucun checkpoint trouvé!")
        return
    
    checkpoint = torch.load(checkpoint_path)
    perception.load_state_dict(checkpoint['perception'])
    world_model.load_state_dict(checkpoint['world_model'])
    cost.load_state_dict(checkpoint['cost'])
    
    perception.eval()
    world_model.eval()
    cost.eval()
    
    print("=" * 60)
    print("DIAGNOSTIC DE L'AGENT JEPA")
    print("=" * 60)
    
    # ============================================================
    # TEST 1: Latent space structure - is distance meaningful?
    # ============================================================
    print("\n--- TEST 1: Structure de l'espace latent ---")
    
    # Encode several positions and check distances
    positions = [(0,0), (0,9), (9,0), (9,9), (5,5), (0,5), (5,0)]
    labels = ["(0,0) Start", "(0,9) Top-Right", "(9,0) Bot-Left", 
              "(9,9) Target", "(5,5) Station", "(0,5) Mid-Top", "(5,0) Mid-Left"]
    
    latent_vectors = {}
    with torch.no_grad():
        for pos, label in zip(positions, labels):
            # Skip positions that are obstacles
            if list(pos) in env.obstacles:
                print(f"  {label}: OBSTACLE (skip)")
                continue
            obs = create_obs_at_position(env, pos[0], pos[1]).unsqueeze(0)
            s = perception(obs)
            latent_vectors[label] = s
            print(f"  {label}: norm={s.norm().item():.3f}, mean={s.mean().item():.4f}, std={s.std().item():.4f}")
    
    # Distance matrix
    print("\n  Distances entre positions clés:")
    keys = list(latent_vectors.keys())
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j > i:
                d = torch.sum((latent_vectors[k1] - latent_vectors[k2])**2).item()
                print(f"    {k1} <-> {k2}: {d:.4f}")
    
    # ============================================================
    # TEST 2: Are all latent vectors collapsed to the same point?
    # ============================================================
    print("\n--- TEST 2: Collapse Check ---")
    all_vecs = torch.cat(list(latent_vectors.values()), dim=0)
    var_per_dim = all_vecs.var(dim=0)
    print(f"  Variance moyenne par dimension: {var_per_dim.mean().item():.6f}")
    print(f"  Variance min: {var_per_dim.min().item():.6f}")
    print(f"  Variance max: {var_per_dim.max().item():.6f}")
    print(f"  Dimensions avec variance > 0.01: {(var_per_dim > 0.01).sum().item()}/{latent_dim}")
    
    if var_per_dim.mean().item() < 0.001:
        print("  ⚠️  EFFONDREMENT DÉTECTÉ! L'espace latent est collapsé.")
    else:
        print("  ✅ L'espace latent a de la variance.")
    
    # ============================================================
    # TEST 3: World Model - does it predict different next states 
    #          for different actions?
    # ============================================================
    print("\n--- TEST 3: World Model - Prédictions par action ---")
    with torch.no_grad():
        s_start = latent_vectors["(0,0) Start"]
        action_names = ["Haut", "Bas", "Gauche", "Droite"]
        
        print(f"  État de départ (0,0): norm={s_start.norm().item():.3f}")
        for a in range(4):
            a_onehot = F.one_hot(torch.tensor([a]), num_classes=4).float()
            # Use z=0 to remove stochasticity
            z_zero = torch.zeros(1, 4)
            s_pred = world_model(s_start, a_onehot, z=z_zero)
            delta = s_pred - s_start
            delta_norm = delta.norm().item()
            print(f"    Action {a} ({action_names[a]}): delta_norm={delta_norm:.4f}")
    
    # ============================================================
    # TEST 4: Cost/Critic values at different positions
    # ============================================================
    print("\n--- TEST 4: Valeurs du Critique à différentes positions ---")
    with torch.no_grad():
        for label, s in latent_vectors.items():
            v = cost(s).item()
            print(f"  {label}: V(s) = {v:.4f}")
    
    # ============================================================
    # TEST 5: What does the planner actually choose?
    # ============================================================
    print("\n--- TEST 5: Décisions du Planificateur (CEM) ---")
    with torch.no_grad():
        target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0)
        station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0)
        s_target = perception(target_obs)
        s_station = perception(station_obs)
        configurator.set_goals(s_target, s_station)
    
    s_goal, w_energy, w_collision, w_goal = configurator.get_configuration(100)
    
    print(f"  s_goal (target) norm: {s_goal.norm().item():.3f}")
    print(f"  w_goal: {w_goal}")
    
    # Test planner from (0,0)
    s_start = latent_vectors["(0,0) Start"]
    dist_start_to_goal = torch.sum((s_start - s_goal)**2).item()
    print(f"  Distance (0,0) -> goal: {dist_start_to_goal:.4f}")
    
    # Test planner from different positions
    test_positions = [(0,0), (5,5), (9,0)]
    for pos in test_positions:
        if list(pos) in env.obstacles:
            continue
        obs = create_obs_at_position(env, pos[0], pos[1]).unsqueeze(0)
        with torch.no_grad():
            s = perception(obs)
        
        a, seq, cost_val = actor.plan(s, world_model, cost, s_goal, w_energy, w_collision, w_goal, 100)
        action_names = ["Haut", "Bas", "Gauche", "Droite"]
        print(f"  Position {pos}: action={action_names[a]}, cost={cost_val:.4f}, sequence={[action_names[s.item()] for s in seq]}")
    
    # ============================================================
    # TEST 6: Does the reward signal actually exist?
    # ============================================================
    print("\n--- TEST 6: Signal de Récompense ---")
    print(f"  Reward quand l'agent atteint la cible: {0.0} (!!)")
    print(f"  ⚠️ env.step() retourne TOUJOURS reward=0.0 (ligne 71 de gridworld.py)")
    print(f"  Le Critique ne peut PAS apprendre sans signal de récompense!")

if __name__ == "__main__":
    main()
