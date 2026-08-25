import torch
import torch.nn.functional as F

def compute_sigreg_loss(s_t, var_weight=25.0, cov_weight=1.0, eps=1e-4):
    """
    Régularisation anti-collapse (Inspirée de VICReg / SIGReg).
    Force l'espace latent à être diversifié (Variance) et décorrélé (Covariance).
    s_t : Tenseur de shape [Batch_size, Latent_dim]
    """
    if len(s_t.shape) == 4:
        # Spatial tensor: flatten spatial dims into batch dim to compute variance over channels
        B, C, H, W = s_t.shape
        s_t = s_t.transpose(1, 3).contiguous().view(B * H * W, C)
        
    B, D = s_t.shape
    
    # 1. Centrer les représentations sur le batch
    s_t_centered = s_t - s_t.mean(dim=0)
    
    # 2. Perte de VARIANCE : Force chaque dimension à avoir un écart-type d'au moins 1.0
    # Cela empêche le réseau de s'effondrer sur un seul point.
    std = torch.sqrt(s_t_centered.var(dim=0) + eps)
    loss_var = torch.mean(F.relu(1.0 - std))
    
    # 3. Perte de COVARIANCE : Empêche les neurones d'apprendre tous la même chose.
    # On veut que la matrice de covariance soit diagonale.
    cov = (s_t_centered.T @ s_t_centered) / (B - 1)
    
    # Créer un masque pour ignorer la diagonale principale
    mask = ~torch.eye(D, device=s_t.device, dtype=torch.bool)
    loss_cov = (cov[mask] ** 2).sum() / D
    
    # On retourne la somme pondérée (les poids 25 et 1 sont les standards de la littérature)
    return (var_weight * loss_var) + (cov_weight * loss_cov)
