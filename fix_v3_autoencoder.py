import torch
import torch.nn.functional as F
import sys
sys.path.append('/Users/poutrainlouis/code/WorldModelTest')

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from collections import deque
import random

device = torch.device('cpu')
perception = Perception(in_channels=4, latent_dim=16).to(device)
world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10).to(device)

optimizer_ae = torch.optim.Adam(list(perception.parameters()) + list(world_model.agent_decoder.parameters()), lr=1e-3)

env = GridWorldEnv(size=10, max_energy=100)

print("1. Pre-training Perception + AgentDecoder (AutoEncoder)...")
for ep in range(500):
    env.reset()
    x_t = env.get_local_observation().unsqueeze(0).to(device)
    
    s_t = perception(x_t)
    agent_pred = world_model.decode_agent(s_t)
    agent_mask = x_t[:, 3:4]
    
    weight_mask = torch.ones_like(agent_mask) + 100.0 * agent_mask
    loss = ((agent_pred - agent_mask)**2 * weight_mask).mean()
    
    optimizer_ae.zero_grad()
    loss.backward()
    optimizer_ae.step()

# Freeze Perception
for param in perception.parameters():
    param.requires_grad = False

print("2. Training World Model (Dynamics)...")
optimizer_wm = torch.optim.Adam(world_model.parameters(), lr=1e-3)

buffer = []
for ep in range(100):
    env.reset()
    obs = env.get_local_observation()
    for step in range(50):
        action = random.randint(0, 3)
        next_obs, _, done = env.step(action)
        buffer.append((obs, action, next_obs))
        obs = next_obs
        if done: break

for ep in range(1500):
    batch = random.sample(buffer, 64)
    x_t_batch = torch.stack([t[0] for t in batch]).to(device)
    a_batch = torch.tensor([t[1] for t in batch]).to(device)
    x_next_batch = torch.stack([t[2] for t in batch]).to(device)
    
    a_onehot = F.one_hot(a_batch, num_classes=4).float().unsqueeze(1) # [64, 1, 4]
    
    with torch.no_grad():
        s_t = perception(x_t_batch)
        s_next_target = perception(x_next_batch)
        
    s_preds, _ = world_model.forward_seq(s_t, a_onehot) # [64, 1, 16, 10, 10]
    
    mse = (s_preds[:, 0] - s_next_target)**2
    agent_mask_next = x_next_batch[:, 3:4]
    weight_mask_next = torch.ones_like(mse) + 100.0 * agent_mask_next
    loss_latent = (mse * weight_mask_next).mean()
    
    agent_preds = world_model.decode_agent(s_preds[:, 0])
    weight_mask_decoder = torch.ones_like(agent_mask_next) + 100.0 * agent_mask_next
    loss_recon = ((agent_preds - agent_mask_next)**2 * weight_mask_decoder).mean()
    
    loss = loss_latent + 10.0 * loss_recon
    
    optimizer_wm.zero_grad()
    loss.backward()
    optimizer_wm.step()
    
    if ep % 500 == 0:
        print(f'WM Epoch {ep}, Loss: {loss.item():.4f}')

torch.save({'perception': perception.state_dict(), 'world_model': world_model.state_dict()}, 'checkpoints/agent_h_jepa.pth')
print("Models saved successfully!")
