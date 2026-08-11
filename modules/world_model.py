import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, z_dim=4):
        super(WorldModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # JEPA-style predictive model
        # Predicts next state from current state, action, and latent z
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, s_t, a_t, z=None):
        # s_t: (B, latent_dim)
        # a_t: (B, action_dim) one-hot encoded
        # z: (B, z_dim) optional
        if len(s_t.shape) == 1:
            s_t = s_t.unsqueeze(0)
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
            
        batch_size = s_t.size(0)
        
        if z is None:
            # Sample z from standard normal distribution
            z = torch.randn(batch_size, self.z_dim, device=s_t.device)
            
        x = torch.cat([s_t, a_t, z], dim=-1)
        s_next = self.predictor(x)
        
        return s_next
