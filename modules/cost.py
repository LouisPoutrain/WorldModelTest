import torch
import torch.nn as nn
import torch.nn.functional as F

def update_ema(target_net, online_net, tau=0.01):
    """
    Met à jour les poids du Target Network via Moyenne Mobile Exponentielle (EMA).
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def intrinsic_cost(s_t, s_next, s_goal, wall_penalty=5.0):
    """
    Calcule le Coût Intrinsèque (Hard-wired) dans l'espace latent.
    - Distance L2 jusqu'à l'objectif.
    - Pénalité de mur : Si l'état latent ne change pas (ou très peu) entre s_t et s_next, 
      cela signifie que l'agent a heurté un mur (le World Model prédit une position inchangée).
    """
    # Distance géométrique dans l'espace latent
    dist_to_goal = torch.norm((s_next - s_goal).view(s_next.size(0), -1), p=2, dim=1)
    
    # Détection de collision (si l'état change très peu)
    state_change = torch.norm((s_next - s_t).view(s_next.size(0), -1), p=2, dim=1)
    
    # Seuil empirique pour définir le "non-mouvement" dans l'espace latent
    # Dans JEPA, un déplacement devrait modifier significativement le vecteur latent.
    collision_mask = (state_change < 0.1).float() 
    
    cost = dist_to_goal + collision_mask * wall_penalty
    return cost

class SpatialCritic(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64):
        super(SpatialCritic, self).__init__()
        
        # Prend en entrée la concaténation de s_sim et s_goal
        self.conv_net = nn.Sequential(
            nn.Conv2d(latent_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 10 * 10, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1) # Prédit la somme des coûts futurs
        )
        
    def forward(self, s_sim, s_goal):
        is_sequence = len(s_sim.shape) == 5
        
        if is_sequence:
            B, T, C, H, W = s_sim.shape
            s_sim = s_sim.view(B * T, C, H, W)
            if len(s_goal.shape) == 4:
                s_goal = s_goal.unsqueeze(1).expand(B, T, C, H, W)
            s_goal = s_goal.reshape(B * T, C, H, W)
            
        if s_goal.size(0) == 1 and s_sim.size(0) > 1:
            s_goal = s_goal.expand(s_sim.size(0), -1, -1, -1)
            
        x = torch.cat([s_sim, s_goal], dim=1) # [B, 32, H, W]
        
        features = self.conv_net(x)
        pooled = features.view(features.size(0), -1)
        dist = self.mlp(pooled).squeeze(-1) # [B]
        
        # Softplus pour garantir un coût positif
        dist = F.softplus(dist)
        
        if is_sequence:
            dist = dist.view(B, T)
            
        return dist
