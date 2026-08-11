import torch
import torch.nn as nn

class Cost(nn.Module):
    def __init__(self, latent_dim=32):
        super(Cost, self).__init__()
        
        # Le Critique est désactivé (Option 4). On peut le garder pour des expériences futures,
        # mais la décision principale se fait sur l'intrinsic_cost.
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def intrinsic_cost(self, s_t, s_goal, w_energy, w_collision, w_goal, sim_energy=None, sim_collision=None):
        """
        Boussole intrinsèque de l'agent.
        On utilise la distance euclidienne dans l'espace latent (structuré par VICReg).
        """
        # Distance géométrique à l'objectif dans l'espace latent
        dist_cost = torch.sum((s_t - s_goal) ** 2, dim=-1, keepdim=True)
        
        cost = w_goal * dist_cost
        
        if sim_energy is not None:
            # Pénalité de famine si l'énergie simulée est trop basse
            famine_penalty = torch.clamp(30.0 - sim_energy, min=0.0) / 30.0
            cost += w_energy * famine_penalty
            
        if sim_collision is not None:
            cost += w_collision * sim_collision
            
        return cost
        
    def forward(self, s_t, *args, **kwargs):
        # Fallback pour compatibilité (si jamais on veut réactiver TD-learning)
        return self.critic(s_t)
