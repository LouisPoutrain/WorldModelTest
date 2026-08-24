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
    def __init__(self, in_channels=4, latent_dim=16):
        super(Perception, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ResidualBlock(32),
            ResidualBlock(32),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ResidualBlock(32),
            ResidualBlock(32),
            
            nn.Conv2d(32, self.latent_dim, kernel_size=1, stride=1)
        )
        

        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.encoder(x)
