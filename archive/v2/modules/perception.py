import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class Perception(nn.Module):
    def __init__(self, in_channels=4, latent_dim=32):
        super(Perception, self).__init__()
        
        # Un petit ResNet pour extraire des features topologiques riches
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ResidualBlock(32),
            ResidualBlock(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Downsample (5x5)
            nn.ReLU(inplace=True),
            
            ResidualBlock(64),
            ResidualBlock(64),
            
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
            # Pas de BatchNorm finale car SIGReg s'occupe de régulariser l'espace latent.
        )
        
    def forward(self, x):
        # x is expected to be of shape (B, C, 10, 10)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        s_t = self.encoder(x)
        return s_t
