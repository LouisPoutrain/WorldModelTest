import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Ajouter le parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.perception import Perception
from modules.cost import Cost

def train_supervised_critic():
    print("🎯 Démarrage de l'entraînement Supervisé du Critique (Geodesic Compass)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Chemins
    dataset_path = "dataset/geodesic_data.pt"
    perception_ckpt = "checkpoints/perception_jepa.pth"
    critic_save_path = "checkpoints/critic_supervised.pth"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : Le dataset {dataset_path} est introuvable. Lancez generate_geodesic_dataset.py d'abord.")
        return
        
    if not os.path.exists(perception_ckpt):
        print(f"❌ Erreur : Le modèle de perception {perception_ckpt} est introuvable. Lancez train_perception_jepa.py d'abord.")
        return
        
    # Charger les modèles
    perception = Perception(in_channels=4, grid_size=10, embed_dim=64, latent_dim=32).to(device)
    perception.load_state_dict(torch.load(perception_ckpt, map_location=device)['perception'])
    perception.eval()
    
    cost_module = Cost(latent_dim=32).to(device)
    
    # Charger le dataset
    print(f"📦 Chargement du dataset depuis {dataset_path}...")
    dataset = torch.load(dataset_path, map_location="cpu")
    observations = dataset['observations'] # (N, 4, 10, 10)
    distances = dataset['distances'] # (N, 1)
    
    N = len(observations)
    print(f"   Nombre d'exemples : {N}")
    
    # Pré-calculer l'espace latent pour gagner du temps
    print("🧠 Pré-calcul de l'espace latent s_t...")
    batch_size = 512
    s_t_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size)):
            obs_batch = observations[i:i+batch_size].to(device)
            # Utilise le Target Encoder stable grâce à la méthode forward par défaut
            s_t_batch = perception(obs_batch)
            s_t_list.append(s_t_batch.cpu())
            
    latents = torch.cat(s_t_list, dim=0)
    
    # Split Train / Val (90/10)
    split = int(0.9 * N)
    
    train_dataset = TensorDataset(latents[:split], distances[:split])
    val_dataset = TensorDataset(latents[split:], distances[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Entraînement
    optimizer = optim.AdamW(cost_module.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss() # Erreur Absolue Moyenne (MAE) pour apprendre la vraie distance
    
    # Init CSV Log
    import csv
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_critic_supervised.csv")
    epochs = 50
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = "checkpoints/critic_supervised_latest.pth"
    best_model_path = "checkpoints/critic_supervised_best.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Reprise de l'entraînement à partir de {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cost_module.critic.load_state_dict(checkpoint['critic'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   -> Reprise à l'époque {start_epoch+1}")
    else:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Timestamp', 'Epoch', 'Train_MAE', 'Val_MAE'])
    
    for epoch in range(start_epoch, epochs):
        cost_module.train()
        train_loss = 0
        
        for s_t, target_dist in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            s_t, target_dist = s_t.to(device), target_dist.to(device)
            
            optimizer.zero_grad()
            pred_dist = cost_module(s_t)
            
            loss = criterion(pred_dist, target_dist)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        cost_module.eval()
        val_loss = 0
        with torch.no_grad():
            for s_t, target_dist in val_loader:
                s_t, target_dist = s_t.to(device), target_dist.to(device)
                pred_dist = cost_module(s_t)
                loss = criterion(pred_dist, target_dist)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")
        
        with open(log_path, 'a', newline='') as f:
            import time
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch+1, train_loss, val_loss])
            
        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'critic': cost_module.critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(critic_save_path), exist_ok=True)
            torch.save({'critic': cost_module.critic.state_dict()}, best_model_path)
            # Compatibilité
            torch.save({'critic': cost_module.critic.state_dict()}, critic_save_path)
            
    print(f"✅ Entraînement du Critique terminé. Meilleure Val MAE: {best_val_loss:.4f}")
    print(f"💾 Critique sauvegardé dans {best_model_path}")

if __name__ == "__main__":
    train_supervised_critic()
