import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.actor import Actor
from modules.macro_planner import MacroPlanner

device = torch.device('cpu')
perception = Perception(in_channels=4, latent_dim=16).to(device)
world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10).to(device)
actor = Actor(action_dim=4, num_sequences=100, horizon=5, cem_iterations=3, elite_size=20, w_critic=0.0)
macro_planner = MacroPlanner(waypoint_lookahead=5)

checkpoint = torch.load('checkpoints/agent_h_jepa.pth', map_location=device)
perception.load_state_dict(checkpoint['perception'])
world_model.load_state_dict(checkpoint['world_model'])
perception.eval()
world_model.eval()

env = GridWorldEnv(size=10, max_energy=200, obstacle_density=0.15)
env.reset()
h_t = None
print(f'Start: Agent at {env.agent_pos}, Target at {env.target_pos}')

for step in range(10):
    x_t = env.get_local_observation().unsqueeze(0).to(device)
    x_waypoint = macro_planner.get_waypoint_obs(x_t).to(device)
    
    agent_pos = tuple(torch.nonzero(x_t[0, 3])[0].tolist())
    waypoint_pos = tuple(torch.nonzero(x_waypoint[0, 3])[0].tolist())
    print(f'
Step {step}: Agent {agent_pos} -> Waypoint {waypoint_pos}')
    
    with torch.no_grad():
        s_t = perception(x_t)
        s_waypoint = perception(x_waypoint)
    
    a_t, cost, _ = actor.plan(s_t, h_t, world_model, None, s_waypoint, w_goal=1.0)
    
    action_names = ['Up', 'Down', 'Left', 'Right']
    print(f'Chosen Action: {action_names[a_t]} (Cost: {cost:.4f})')
    
    obs, reward, done = env.step(a_t)
    a_t_onehot = F.one_hot(torch.tensor([a_t]), num_classes=4).float().to(device)
    with torch.no_grad():
        _, h_t = world_model.forward_step(s_t, a_t_onehot, h_t)
        
    if done:
        print('DONE!')
        break
