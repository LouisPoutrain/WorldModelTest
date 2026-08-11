import torch
import torch.nn as nn

class Perception(nn.Module):
    def __init__(self, in_channels=4, latent_dim=32):
        super(Perception, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        # x is expected to be of shape (B, C, 10, 10)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        s_t = self.encoder(x)
        return s_t


