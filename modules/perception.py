import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class Perception(nn.Module):
    def __init__(self, in_channels=4, grid_size=10, embed_dim=64, latent_dim=32):
        super(Perception, self).__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # Patch embedding (1x1 patches for the 10x10 grid)
        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Context Encoder (ViT)
        self.context_encoder = nn.Sequential(
            TransformerBlock(embed_dim, num_heads=4),
            TransformerBlock(embed_dim, num_heads=4)
        )
        
        # Target Encoder (EMA copy)
        self.target_encoder = nn.Sequential(
            TransformerBlock(embed_dim, num_heads=4),
            TransformerBlock(embed_dim, num_heads=4)
        )
        # Stop gradient for target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Predictor (ViT)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.predictor = nn.Sequential(
            TransformerBlock(embed_dim, num_heads=4),
            TransformerBlock(embed_dim, num_heads=4)
        )
        
        # Projection head to latent_dim (32)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # Initialize target encoder with context encoder weights
        self.update_target_encoder(momentum=1.0)
        
    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
                
    def forward(self, x):
        """
        Forward method used for downstream tasks (World Model, Critic).
        Returns the latent state s_t.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2) # (B, N, C)
        x = self.patch_embed(x) + self.pos_embed
        
        # Use target encoder for stable downstream representations
        with torch.no_grad():
            features = self.target_encoder(x)
        
        # Global average pooling
        global_feature = features.mean(dim=1)
        s_t = self.head(global_feature)
        return s_t
        
    def forward_loss(self, x):
        """
        Compute the I-JEPA self-supervised loss + VICReg regularization.
        Should be called during the perception pre-training phase.
        """
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2) # (B, N, C)
        patches = self.patch_embed(x) + self.pos_embed
        
        # 1. Target representations (Stop Gradient)
        with torch.no_grad():
            target_features = self.target_encoder(patches) # (B, N, dim)
            
        # 2. Masking strategy
        # For a 10x10 grid, we randomly sample a context and target mask.
        device = x.device
        rand_indices = torch.rand(B, self.num_patches, device=device).argsort(dim=-1)
        
        # e.g., 50% context, 20% target
        num_context = int(self.num_patches * 0.5)
        num_target = int(self.num_patches * 0.2)
        
        context_indices = rand_indices[:, :num_context]
        target_indices = rand_indices[:, num_context:num_context+num_target]
        
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)
        
        # Extract context patches
        context_patches = patches[batch_indices, context_indices] # (B, num_context, dim)
        
        # Encode context
        context_features = self.context_encoder(context_patches) # (B, num_context, dim)
        
        # 3. Predict target blocks
        mask_tokens = self.mask_token.expand(B, num_target, -1)
        target_pos_embed = self.pos_embed.expand(B, -1, -1)[batch_indices, target_indices]
        
        # Predictor input: context features + target mask tokens
        predictor_input = torch.cat([context_features, mask_tokens + target_pos_embed], dim=1)
        predictor_output = self.predictor(predictor_input)
        
        # Extract predictions for the target tokens
        predictions = predictor_output[:, num_context:] # (B, num_target, dim)
        targets = target_features[batch_indices, target_indices] # (B, num_target, dim)
        
        # 4. Losses
        # L1 Loss for I-JEPA prediction
        rec_loss = F.l1_loss(predictions, targets)
        
        # VICReg Regularization on the global representation
        global_rep = self.head(target_features.mean(dim=1)) # (B, latent_dim)
        
        # Variance loss
        std = torch.sqrt(global_rep.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std))
        
        # Covariance loss
        global_rep_centered = global_rep - global_rep.mean(dim=0)
        cov = (global_rep_centered.T @ global_rep_centered) / (B - 1)
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / global_rep.shape[1]
        
        loss = rec_loss + 1.0 * std_loss + 0.04 * cov_loss
        
        return loss, rec_loss, std_loss, cov_loss
