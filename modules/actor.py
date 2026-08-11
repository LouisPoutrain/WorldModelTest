import torch
import torch.nn.functional as F

class Actor:
    def __init__(self, action_dim=4, num_sequences=50, horizon=5, gamma=0.9, cem_iterations=3, elite_size=10):
        self.action_dim = action_dim
        self.N = num_sequences
        self.H = horizon
        self.gamma = gamma
        self.M = cem_iterations
        self.K = elite_size
        
    def plan(self, s_t, world_model, cost_module, s_goal, w_energy, w_collision, w_goal, current_energy):
        """
        Mode-2 (System 2): Planning by simulating N trajectories of length H using Cross-Entropy Method (CEM)
        """
        best_cost = float('inf')
        best_sequence = None
        
        # Initialize action probabilities uniformly: (H, action_dim)
        action_probs = torch.ones(self.H, self.action_dim) / self.action_dim
        
        for iteration in range(self.M):
            # Sample N action sequences from the current distribution: (N, H)
            # torch.multinomial samples from a row vector, so we do it step by step
            action_sequences = torch.zeros((self.N, self.H), dtype=torch.long)
            for t in range(self.H):
                # action_probs[t] shape is (action_dim,)
                # Repeat it N times to sample N actions for this step
                p_t = action_probs[t].unsqueeze(0).repeat(self.N, 1)
                action_sequences[:, t] = torch.multinomial(p_t, 1).squeeze(1)
            
            s_sim = s_t.repeat(self.N, 1) # (N, latent_dim)
            total_costs = torch.zeros(self.N, 1)
            
            for t in range(self.H):
                a_t = action_sequences[:, t]
                a_t_onehot = F.one_hot(a_t, num_classes=self.action_dim).float()
                
                # Predict next latent state, letting the world model sample 'z' internally
                with torch.no_grad():
                    s_next = world_model(s_sim, a_t_onehot)
                    
                with torch.no_grad():
                    c_t = cost_module(s_next)
                
                total_costs += (self.gamma ** t) * c_t
                s_sim = s_next
                
            # Find the top K elite sequences
            # total_costs shape: (N, 1) -> squeeze to (N,)
            top_costs, top_indices = torch.topk(total_costs.squeeze(), self.K, largest=False)
            
            elite_sequences = action_sequences[top_indices] # (K, H)
            
            # Update action probabilities based on elite sequences
            new_probs = torch.zeros_like(action_probs)
            for t in range(self.H):
                # Count frequencies of actions in elite sequences at step t
                counts = torch.bincount(elite_sequences[:, t], minlength=self.action_dim).float()
                # Laplace smoothing to prevent 0 probabilities
                new_probs[t] = (counts + 0.1) / (self.K + 0.1 * self.action_dim)
                
            action_probs = new_probs
            
            # Keep track of the best overall
            if top_costs[0].item() < best_cost:
                best_cost = top_costs[0].item()
                best_sequence = elite_sequences[0]
        
        return best_sequence[0].item(), best_sequence, best_cost
