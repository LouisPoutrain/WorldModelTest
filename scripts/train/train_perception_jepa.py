import torch
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Ajouter le parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception

def train_jepa():
    print("🚀 Démarrage de l'entraînement I-JEPA (Self-Supervised)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Init environment & model
    env = GridWorldEnv(size=10, max_energy=100, procedural=True)
    perception = Perception(in_channels=4, grid_size=10, embed_dim=64, latent_dim=32).to(device)
    
    # Optimizer (only on context_encoder, predictor and head)
    # The target_encoder is updated via EMA and does not require gradients
    params_to_update = [
        {'params': perception.patch_embed.parameters()},
        {'params': [perception.pos_embed, perception.mask_token]},
        {'params': perception.context_encoder.parameters()},
        {'params': perception.predictor.parameters()},
        {'params': perception.head.parameters()}
    ]
    optimizer = optim.AdamW(params_to_update, lr=1e-3, weight_decay=1e-4)
    
    epochs = 100
    batch_size = 64
    steps_per_epoch = 100
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # Init CSV Log
    import csv
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_perception_jepa.csv")
    
    # Reprise d'entraînement (Resuming)
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = "checkpoints/perception_jepa_latest.pth"
    best_model_path = "checkpoints/perception_jepa_best.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Reprise de l'entraînement à partir de {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"   -> Reprise à l'époque {start_epoch+1}")
    else:
        # Création du fichier CSV seulement si on commence de zéro
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Timestamp', 'Epoch', 'Loss', 'Rec_Loss', 'Std_Loss', 'Cov_Loss'])
            
    # Momentum for EMA
    momentum_start = 0.996
    momentum_end = 1.0
    total_steps = epochs * steps_per_epoch
    
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_rec = 0
        epoch_std = 0
        epoch_cov = 0
        
        perception.train()
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # Generate batch
            batch_obs = []
            for _ in range(batch_size):
                obs = env.reset()
                batch_obs.append(obs)
            
            x = torch.stack(batch_obs).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, rec_loss, std_loss, cov_loss = perception.forward_loss(x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # EMA Update
            current_momentum = momentum_start + (momentum_end - momentum_start) * (step / total_steps)
            perception.update_target_encoder(momentum=current_momentum)
            
            epoch_loss += loss.item()
            epoch_rec += rec_loss.item()
            epoch_std += std_loss.item()
            epoch_cov += cov_loss.item()
            step += 1
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/steps_per_epoch:.4f} | Rec: {epoch_rec/steps_per_epoch:.4f} | Std: {epoch_std/steps_per_epoch:.4f} | Cov: {epoch_cov/steps_per_epoch:.4f}")
        
        with open(log_path, 'a', newline='') as f:
            import time
            csv.writer(f).writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'), 
                epoch+1, 
                epoch_loss/steps_per_epoch, 
                epoch_rec/steps_per_epoch, 
                epoch_std/steps_per_epoch, 
                epoch_cov/steps_per_epoch
            ])
            
        # Save latest checkpoint at every epoch
        torch.save({
            'epoch': epoch + 1,
            'perception': perception.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)
            
        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'perception': perception.state_dict()}, best_model_path)
            # On garde aussi une copie compatible avec l'évaluation actuelle
            torch.save({'perception': perception.state_dict()}, "checkpoints/perception_jepa.pth")
            
    print(f"✅ Entraînement I-JEPA terminé. Meilleur modèle dans {best_model_path}")

if __name__ == "__main__":
    train_jepa()
