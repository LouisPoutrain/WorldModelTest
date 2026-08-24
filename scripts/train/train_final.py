import os
import time
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

import sys
sys.path.append('/Users/poutrainlouis/code/WorldModelTest')

from modules.perception import Perception
from modules.world_model import WorldModel
from modules.memory import ShortTermMemory

def get_batch(memory, batch_size, horizon):
    pass

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print('🚀 Démarrage Entraînement H-JEPA (V3) - FINAL T=1 + DECODER LOSS')

    perception = Perception(in_channels=4, latent_dim=16).to(device)
    world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10).to(device)
    
    if os.path.exists('checkpoints/agent_h_jepa.pth'):
        checkpoint = torch.load('checkpoints/agent_h_jepa.pth', map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        world_model.load_state_dict(checkpoint['world_model'])
        print('✅ AutoEncoder chargé. Perception gelée pour entraînement de la dynamique.')

    # FREEZE PERCEPTION
    for param in perception.parameters():
        param.requires_grad = False
    
    # FREEZE DECODER
    for param in world_model.agent_decoder.parameters():
        param.requires_grad = False

    target_encoder = Perception(in_channels=4, latent_dim=16).to(device)
    target_encoder.load_state_dict(perception.state_dict())
    
    optimizer_wm = optim.Adam(world_model.parameters(), lr=1e-3) # Increased LR
    
    memory = ShortTermMemory(capacity=10000)
    
    from env.gridworld import GridWorldEnv
    env = GridWorldEnv(size=10, max_energy=200, obstacle_density=0.15)
    
    num_episodes = 1000 # Reduced to 1000 since T=1 converges faster
    batch_size = 64 # Increased batch size
    seq_length = 5
    
    avg_loss_pred = 0
    log_count = 0
    
    for episode in range(num_episodes):
        env.reset()
        obs = env.get_local_observation()
        
        for step in range(100):
            action = random.randint(0, 3) if random.random() < 0.2 else 1
            next_obs, reward, done = env.step(action)
            memory.push(obs.unsqueeze(0), action, next_obs.unsqueeze(0), reward, done)
            obs = next_obs
            if done: break
            
        for update in range(20): # Increased updates per episode
            if len(memory.buffer) < batch_size * seq_length: continue
            
            try:
                x_0_batch, a_seq_batch, x_next_seq_batch, r_batch, d_batch = memory.sample_sequences(batch_size, seq_length)
                
                x_0_batch = x_0_batch.to(device)
                a_seq_batch = a_seq_batch.to(device)
                x_next_seq_batch = x_next_seq_batch.to(device)
                
                a_seq_onehot = F.one_hot(a_seq_batch.long(), num_classes=4).float()
                
                s_0 = perception(x_0_batch)
                
                with torch.no_grad():
                    B, T_s, C, H, W = x_next_seq_batch.shape
                    s_next_target_flat = target_encoder(x_next_seq_batch.view(B*T_s, C, H, W))
                    s_next_target = s_next_target_flat.view(B, T_s, 16, 10, 10)
                
                # TRAIN ONLY ON T=1
                s_preds, _ = world_model.forward_seq(s_0, a_seq_onehot[:, :1])
                
                mse = (s_preds - s_next_target[:, :1]) ** 2
                agent_mask = x_next_seq_batch[:, :1, 3:4]
                weight_mask = torch.ones_like(mse) + 100.0 * agent_mask
                loss_latent = (mse * weight_mask).mean()
                
                # DECODER LOSS
                agent_preds = world_model.decode_agent(s_preds[:, 0]) # [B, 1, 10, 10]
                weight_mask_decoder = torch.ones_like(agent_mask[:, 0]) + 100.0 * agent_mask[:, 0]
                loss_recon = ((agent_preds - agent_mask[:, 0].float())**2 * weight_mask_decoder).mean()
                
                loss_pred = loss_latent + 10.0 * loss_recon
                
                optimizer_wm.zero_grad()
                loss_pred.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
                optimizer_wm.step()
                
                tau = 0.01
                for p_tgt, p_main in zip(target_encoder.parameters(), perception.parameters()):
                    p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                    
                avg_loss_pred += loss_pred.item()
                log_count += 1
                
            except ValueError:
                pass
                
        if (episode + 1) % 50 == 0:
            if log_count > 0:
                print(f'Ep {episode+1}/{num_episodes} | L_pred: {avg_loss_pred/log_count:.4f}')
                avg_loss_pred = 0
                log_count = 0
                
    torch.save({
        'perception': perception.state_dict(),
        'world_model': world_model.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'optimizer_wm': optimizer_wm.state_dict()
    }, 'checkpoints/agent_h_jepa.pth')
    print(f'💾 Checkpoint final sauvegardé')

if __name__ == '__main__':
    main()
