import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
import time
import csv

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.cost import SpatialCritic
from modules.actor import Actor

from tqdm import tqdm


def generate_u_trap(env):
    env.obstacles = []
    env.agent_pos = [5, 3]
    env.target_pos = [5, 6]
    env.station_pos = [0, 0]
    for i in range(3, 8): env.obstacles.append([i, 5])
    env.obstacles.append([3, 4])
    env.obstacles.append([7, 4])

def generate_zigzag(env):
    env.obstacles = []
    env.agent_pos = [1, 1]
    env.target_pos = [8, 8]
    env.station_pos = [0, 0]
    for i in range(0, 7): env.obstacles.append([i, 4])
    for i in range(3, 10): env.obstacles.append([i, 7])

def run_test_suite(tests, perception, world_model, cost_net, actor, device):
    print('📊 BILAN DE GÉNÉRALISATION V4 (H-JEPA + TD Critic pur, sans A*) :')
    print('| Test                 | Taux de Succès | Pas Moyens |')
    print('|----------------------|----------------|------------|')
    
    for name, num_episodes, env_type in tests:
        success_rate, avg_steps = evaluate(perception, world_model, cost_net, actor, device, num_episodes, env_type=env_type)
        print(f'| {name:<20} | {success_rate:>6.1f}%       | {avg_steps:>6.1f}     |')

def evaluate(perception, world_model, cost_net, actor, device, num_episodes=10, env_type=None):
    successes = 0
    total_steps = 0
    
    env = GridWorldEnv(size=10, max_energy=200, obstacle_density=0.15)
    
    for ep in tqdm(range(num_episodes), desc=f'Evaluation'):
        env.reset()
        if env_type == 'u_trap': generate_u_trap(env)
        elif env_type == 'zigzag': generate_zigzag(env)
            
        done = False
        step = 0
        h_t = None
        
        # Obtenir x_goal (l'observation à la position de la cible)
        old_pos = env.agent_pos.copy()
        env.agent_pos = env.target_pos.copy()
        x_goal = env.get_local_observation().unsqueeze(0).to(device)
        env.agent_pos = old_pos
        
        while not done and step < 100:
            x_t = env.get_local_observation().unsqueeze(0).to(device)
            
            with torch.no_grad():
                s_t = perception(x_t)
                s_goal = perception(x_goal)
            
            # MICRO-PLANNER SEUL : CEM pur avec le TD-Critic, plus de MacroPlanner !
            a_t, _, best_h_t = actor.plan(s_t, h_t, world_model, cost_net, s_goal, w_goal=1.0)
            
            obs, reward, done = env.step(a_t)
            
            a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
            with torch.no_grad():
                _, h_t = world_model.forward_step(s_t, a_t_onehot, h_t)
            
            if done:
                if env.agent_pos == env.target_pos:
                    successes += 1
                total_steps += (step + 1)
                break
            step += 1
            
        if not done:
            total_steps += 100
            
    success_rate = (successes / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    return success_rate, avg_steps

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    perception = Perception(in_channels=4, latent_dim=16).to(device)
    
    world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32).to(device)
    
    checkpoint_path = 'checkpoints/agent_h_jepa.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        print('✅ Modèles V3 (H-JEPA) chargés.')
    else:
        print('❌ Modèles introuvables. Entraînez avec train_h_jepa.py d abord.')
        return

    # Nouveau TD Critic
    cost_net = SpatialCritic(latent_dim=16, hidden_dim=64).to(device)
    critic_path = 'checkpoints/agent_critic_td.pth'
    if os.path.exists(critic_path):
        cost_net.load_state_dict(torch.load(critic_path, map_location=device)['cost'])
        cost_net.eval()
        print('✅ TD Critic chargé avec succès.')
    else:
        print('⚠️ Aucun TD Critic trouvé. (Avez-vous fini train_critic_td.py ?)')
        return

    # Actor (CEM)
    # On donne un horizon un peu plus grand à CEM vu qu'il n'y a plus A* pour le guider pas-à-pas
    actor = Actor(action_dim=4, num_sequences=300, horizon=8, cem_iterations=5, elite_size=30, w_critic=0.0)
    
    tests = [
        ('In-Distribution (ID)', 50, None),
        ('OOD: Piège en U', 10, 'u_trap'),
        ('OOD: Labyrinthe', 10, 'zigzag')
    ]
    
    run_test_suite(tests, perception, world_model, cost_net, actor, device)

if __name__ == '__main__':
    main()
