import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import csv
import random

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.sigreg import SIGReg
from modules.memory import ShortTermMemory

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    env = GridWorldEnv(size=10, max_energy=50, obstacle_density=0.15)
    
    perception = Perception(in_channels=4, latent_dim=16).to(device)
    target_encoder = Perception(in_channels=4, latent_dim=16).to(device)
    target_encoder.load_state_dict(perception.state_dict())
    
    world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10).to(device)
    sigreg = SIGReg().to(device)
    
    optimizer_wm = optim.Adam([
        {'params': perception.parameters(), 'lr': 3e-4},
        {'params': world_model.parameters(), 'lr': 3e-4}
    ])
    
    memory = ShortTermMemory(capacity=10000)
    
    print('🚀 Démarrage Entraînement H-JEPA (V3) - Online Collection')
    
    batch_size = 32
    seq_length = 5
    num_episodes = 2000
    
    log_path = 'logs/training_h_jepa.csv'
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Episode', 'Loss_Pred', 'Loss_SIGReg'])
        
    avg_loss_pred = 0.0
    avg_loss_sigreg = 0.0
    log_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        # Collecte d'une trajectoire
        for step in range(100):
            a_t = random.randint(0, 3) # Random walk for diverse state space
            obs_next, reward, done = env.step(a_t)
            memory.push(obs.unsqueeze(0), a_t, obs_next.unsqueeze(0), reward, done)
            obs = obs_next
            
            if done:
                break
                
        # Entraînement
        if len(memory) >= batch_size * seq_length:
            try:
                x_0_batch, a_seq_batch, x_next_seq_batch, r_batch, d_batch = memory.sample_sequences(batch_size, seq_length)
                
                x_0_batch = x_0_batch.to(device)
                a_seq_batch = a_seq_batch.to(device)
                x_next_seq_batch = x_next_seq_batch.to(device)
                
                a_seq_onehot = F.one_hot(a_seq_batch, num_classes=4).float()
                
                s_0 = perception(x_0_batch, project=False)
                s_0_proj = perception(x_0_batch, project=True)
                
                with torch.no_grad():
                    B, T_s, C, H, W = x_next_seq_batch.shape
                    s_next_target_flat = target_encoder(x_next_seq_batch.view(B*T_s, C, H, W), project=True)
                    s_next_target = s_next_target_flat.view(B, T_s, 64, 10, 10)
                    
                s_preds, h_seq = world_model.forward_seq(s_0, a_seq_batch, project=True)
                
                mse = (s_preds - s_next_target) ** 2
                agent_mask = x_next_seq_batch[:, :, 3:4] # [B, T, 1, 10, 10]
                # The agent mask is 1 at the agent's position, 0 elsewhere
                weight_mask = torch.ones_like(mse) + 100.0 * agent_mask
                loss_pred = (mse * weight_mask).mean()
                loss_sigreg = sigreg(s_0_proj)
                
                loss = loss_pred + 1.0 * loss_sigreg
                
                optimizer_wm.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
                optimizer_wm.step()
                
                tau = 0.01
                for p_tgt, p_main in zip(target_encoder.parameters(), perception.parameters()):
                    p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                    
                avg_loss_pred += loss_pred.item()
                avg_loss_sigreg += loss_sigreg.item()
                log_count += 1
                
            except ValueError:
                pass
                
        if (episode + 1) % 50 == 0:
            if log_count > 0:
                print(f'Ep {episode+1}/{num_episodes} | L_pred: {avg_loss_pred/log_count:.4f} | L_sigreg: {avg_loss_sigreg/log_count:.4f}')
                with open(log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), episode+1, avg_loss_pred/log_count, avg_loss_sigreg/log_count])
                avg_loss_pred = 0.0
                avg_loss_sigreg = 0.0
                log_count = 0
                
        if (episode + 1) % 500 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'perception': perception.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'world_model': world_model.state_dict()
            }, 'checkpoints/agent_h_jepa.pth')
            print(f'💾 Checkpoint intermédiaire sauvegardé (Ep {episode+1})')
            
    print('Sauvegarde finale terminée.')

if __name__ == "__main__":
    main()
