import torch
import torch.nn.functional as F

class Actor:
    def __init__(self, action_dim=4, num_sequences=200, horizon=12, cem_iterations=5, elite_size=30):
        self.action_dim = action_dim
        self.N = num_sequences
        self.H = horizon
        self.M = cem_iterations
        self.K = elite_size
        
    def plan(self, s_t, h_t, world_model, s_goal):
        """
        Planification par CEM (Cross-Entropy Method).
        s_t: (1, latent_dim) État latent actuel
        h_t: (1, hidden_dim) État caché actuel du RNN
        world_model: Le modèle du monde séquentiel
        s_goal: (1, latent_dim) Cible
        """
        device = s_t.device
        best_cost = float('inf')
        best_sequence = None
        
        # Initialize action probabilities uniformly: (H, action_dim)
        action_probs = torch.ones(self.H, self.action_dim, device=device) / self.action_dim
        
        for iteration in range(self.M):
            # Sample N action sequences from the current distribution: (N, H)
            action_sequences = torch.zeros((self.N, self.H), dtype=torch.long, device=device)
            for t in range(self.H):
                p_t = action_probs[t].unsqueeze(0).repeat(self.N, 1)
                action_sequences[:, t] = torch.multinomial(p_t, 1).squeeze(1)
            
            # Rollout dans le World Model
            s_sim = s_t.repeat(self.N, 1)  # (N, latent_dim)
            h_sim = h_t.repeat(self.N, 1)  # (N, hidden_dim)
            
            with torch.no_grad():
                for t in range(self.H):
                    a_t_sim = action_sequences[:, t]
                    a_t_onehot = F.one_hot(a_t_sim, num_classes=self.action_dim).float()
                    # On avance d'un pas en mettant à jour l'état caché
                    s_sim, h_sim = world_model.forward_step(s_sim, a_t_onehot, h_sim)
            
            # Coût = MSE entre le DERNIER état prédit et le goal
            cost = F.mse_loss(
                s_sim,  # (N, latent_dim)
                s_goal.expand(self.N, -1),  # (N, latent_dim) 
                reduction='none'
            ).sum(dim=1)  # (N,)
            
            # Find the top K elite sequences (lowest cost)
            top_costs, top_indices = torch.topk(cost, self.K, largest=False)
            elite_sequences = action_sequences[top_indices]  # (K, H)
            
            # Update action probabilities based on elite sequences
            new_probs = torch.zeros_like(action_probs)
            for t in range(self.H):
                counts = torch.bincount(elite_sequences[:, t], minlength=self.action_dim).float()
                # Laplace smoothing
                new_probs[t] = (counts + 0.1) / (self.K + 0.1 * self.action_dim)
                
            action_probs = new_probs
            
            # Keep track of the best overall
            if top_costs[0].item() < best_cost:
                best_cost = top_costs[0].item()
                best_sequence = elite_sequences[0]
        
        return best_sequence[0].item(), best_sequence, best_cost
