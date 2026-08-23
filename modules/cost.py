import torch
import torch.nn as nn

class Cost(nn.Module):
    def __init__(self, latent_dim=32):
        super(Cost, self).__init__()
        
        # Le Critique pour estimer le coût futur attendu (TD-Learning)
        # Sortie : Coût attendu
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, s_t, *args, **kwargs):
        """
        Évalue le coût futur attendu de l'état s_t (TD-Learning).
        Plus le coût est faible, meilleur est l'état.
        """
        return self.critic(s_t)
