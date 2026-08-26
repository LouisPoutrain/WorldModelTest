import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import time
import csv
import sys

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.memory import ShortTermMemory

from utils.losses import compute_sigreg_loss

def main():
    print("🚀 Démarrage de l'entraînement JEPA V2 (Physique & OOD)")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de l'appareil : {device}")
    
    # 1. Initialisation de l'Environnement
    env = GridWorldEnv(size=20, max_energy=100)
    
    # Charger le dataset offline
    dataset_path = "dataset/grids_dungeon.pt"
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur: Dataset introuvable ({dataset_path}). Générez-le avec generate_dataset.py.")
        return
    grids_dataset = torch.load(dataset_path)
    print(f"✅ Dataset hors-ligne chargé : {len(grids_dataset)} grilles complexes.")
    
    latent_dim = 32
    action_dim = 4
    
    # 2. Initialisation des Modules JEPA purs
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    target_encoder = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    target_encoder.load_state_dict(perception.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad = False
        
    world_model = WorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=128, spatial_size=20).to(device)
    
    # Plus de Critique !
    
    # Optimiseurs (seulement Encodeur et World Model)
    optimizer_wm = optim.Adam([
        {'params': perception.parameters(), 'lr': 1e-4},
        {'params': world_model.parameters(), 'lr': 3e-4}
    ])
    
    # 3. Buffer de Replay
    replay_buffer = ShortTermMemory(capacity=100000)
    
    # Paramètres d'entraînement
    num_episodes = 1500
    batch_size = 64
    
    # Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_jepa_v2.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Episode', 'L_pred', 'L_sigreg'])
        
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/agent_h_jepa.pth"
    
    avg_loss_pred = 0.0
    avg_loss_sigreg = 0.0
    log_count = 0
    
    # Boucle d'Épisodes
    for episode in range(num_episodes):
        # Tirer une grille du dataset (tensor 4x20x20)
        grid_data = random.choice(grids_dataset)
        
        # Reconstruire l'état de l'environnement à partir du tenseur
        env.obstacles = grid_data[0].nonzero().tolist()
        
        target_idx = grid_data[1].nonzero()
        env.target_pos = target_idx[0].tolist() if len(target_idx) > 0 else [0, 0]
        
        station_idx = grid_data[2].nonzero()
        env.station_pos = station_idx[0].tolist() if len(station_idx) > 0 else [0, 0]
        
        agent_idx = grid_data[3].nonzero()
        env.agent_pos = agent_idx[0].tolist() if len(agent_idx) > 0 else [0, 0]
        
        env.energy = env.max_energy
        env.done = False
        
        obs = env.get_local_observation()
        
        episode_transitions = []
        last_action = random.randint(0, 3)
        
        # Exploration de l'arène (marche aléatoire avec inertie)
        for step in range(100):
            # 70% de chances de continuer tout droit (Inertie)
            if random.random() < 0.7:
                a_t = last_action
            else:
                a_t = random.randint(0, 3)
            
            last_action = a_t
            
            obs_next, reward, done = env.step(a_t)
            
            episode_transitions.append((obs.clone(), a_t, reward, obs_next.clone(), done))
            obs = obs_next
            
            if done:
                break
                
        # Ajouter au buffer
        for trans in episode_transitions:
            replay_buffer.push(trans[0].unsqueeze(0), trans[1], trans[3].unsqueeze(0), trans[2], trans[4])
                
        # Apprentissage JEPA
        if len(replay_buffer) > batch_size * 2:
            perception.train()
            world_model.train()
            
            for _ in range(50): # 50 updates per episode (1 pour 2 pas)
                try:
                    T_seq = 5
                    s_0_batch, s_a, s_next, s_r, s_done = replay_buffer.sample_sequences(batch_size, seq_len=T_seq)
                    
                    s_0_batch = s_0_batch.to(device)
                    s_a = s_a.to(device)
                    s_next = s_next.to(device)
                    
                    B, T, C, H, W = s_next.shape
                    
                    # Encodage de l'état initial
                    z_0 = perception(s_0_batch) # (B, latent_dim)
                    z_t = z_0.unsqueeze(1) # on ne garde que t=0 pour initialiser, BPTT fera le reste
                    
                    s_next_flat = s_next.view(B * T, C, H, W)
                    
                    # Encodage par la Perception Cible (Réseau Cible - EMA)
                    with torch.no_grad():
                        z_next_target_flat = target_encoder(s_next_flat)
                        z_next_target = z_next_target_flat.view(B, T, latent_dim, 20, 20)
                        
                    # Prédiction par le World Model (BPTT)
                    h_t = world_model.init_hidden(B, device)
                    z_preds = []
                    curr_z = z_0
                    for t in range(T):
                        a_t_onehot = F.one_hot(s_a[:, t], num_classes=4).float()
                        z_pred, h_t = world_model.forward_step(curr_z, a_t_onehot, h_t)
                        z_preds.append(z_pred)
                        curr_z = z_pred # auto-regressif
                    z_preds = torch.stack(z_preds, dim=1) # (B, T, latent_dim)
                    
                    optimizer_wm.zero_grad()
                    
                    # Perte JEPA (Prédiction Latente)
                    loss_pred = F.mse_loss(z_preds, z_next_target)
                    
                    # Perte SIGReg (Anti-Collapse sur l'espace latent initial du batch)
                    loss_sigreg = compute_sigreg_loss(z_0)
                    
                    loss_wm = loss_pred + 1.0 * loss_sigreg
                    
                    loss_wm.backward()
                    torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
                    optimizer_wm.step()
                    
                    avg_loss_pred += loss_pred.item()
                    avg_loss_sigreg += loss_sigreg.item()
                    log_count += 1
                    
                    # Mise à jour EMA du Target Encoder
                    tau = 0.01
                    for p_tgt, p_main in zip(target_encoder.parameters(), perception.parameters()):
                        p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                        
                except Exception as e:
                    import traceback
                    print(f"Erreur d'apprentissage : {e}")
                    traceback.print_exc()
                    break

                break

        
        # Tracking
        if (episode + 1) % 50 == 0:
            n = max(log_count, 1)
            
            print(f"Ep {episode+1}/{num_episodes} | L_pred: {avg_loss_pred/n:.6f} | L_sigreg: {avg_loss_sigreg/n:.6f}")
            
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    episode + 1,
                    f"{avg_loss_pred/n:.6f}",
                    f"{avg_loss_sigreg/n:.6f}"
                ])
                
            avg_loss_pred = 0.0
            avg_loss_sigreg = 0.0
            log_count = 0
            
        if (episode + 1) % 100 == 0 or (episode + 1) == num_episodes:
            torch.save({
                'episode': episode + 1,
                'perception': perception.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'world_model': world_model.state_dict(),
                'optimizer_wm': optimizer_wm.state_dict()
            }, checkpoint_path)
            print(f"💾 Poids V2 sauvegardés (Épisode {episode+1})")

if __name__ == "__main__":
    main()
