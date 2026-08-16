import torch
import torch.nn.functional as F

class Actor:
    def __init__(self, action_dim=4, num_sequences=200, horizon=12, cem_iterations=5, elite_size=30, gamma=0.9):
        self.action_dim = action_dim
        self.N = num_sequences
        self.H = horizon
        self.M = cem_iterations
        self.K = elite_size
        self.gamma = gamma
        
    def plan(self, s_t, h_t, world_model, cost_module, s_goal, w_goal=1.0):
        """
        Planification par CEM (Cross-Entropy Method) combinant Boussole et Critique.
        s_t: (1, latent_dim) État latent actuel
        h_t: (1, hidden_dim) État caché actuel du RNN
        world_model: Le modèle du monde séquentiel
        cost_module: Le réseau Critique (Cost)
        s_goal: (1, latent_dim) Cible
        w_goal: Poids de la distance au goal dans le coût total
        """
        device = s_t.device
        best_cost = float('inf')
        best_sequence = None
        
        # Initialisation uniforme des probabilités d'actions: (H, action_dim)
        action_probs = torch.ones(self.H, self.action_dim, device=device) / self.action_dim
        
        for iteration in range(self.M):
            # Échantillonnage de N séquences d'actions
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
                    # Déroulement d'un pas temporel
                    s_sim, h_sim = world_model.forward_step(s_sim, a_t_onehot, h_sim)
                
                # 1. Coût Intrinsèque (Distance euclidienne au goal sur l'état final simulé)
                dist_to_goal = F.mse_loss(
                    s_sim, 
                    s_goal.expand(self.N, -1), 
                    reduction='none'
                ).sum(dim=1)  # (N,)
                
                # 2. Coût prédit par le Critique (Valeur attendue de l'état futur)
                c_critic = cost_module(s_sim).squeeze(-1)  # (N,)
                
                # 3. Combinaison des coûts (Boussole + Critique)
                total_cost = (w_goal * dist_to_goal) + c_critic
            
            # Échantillonnage des K meilleures séquences ("élites")
            top_costs, top_indices = torch.topk(total_cost, self.K, largest=False)
            elite_sequences = action_sequences[top_indices]  # (K, H)
            
            # Mise à jour de la distribution de probabilité avec lissage de Laplace
            new_probs = torch.zeros_like(action_probs)
            for t in range(self.H):
                counts = torch.bincount(elite_sequences[:, t], minlength=self.action_dim).float()
                new_probs[t] = (counts + 0.1) / (self.K + 0.1 * self.action_dim)
                
            action_probs = new_probs
            
            if top_costs[0].item() < best_cost:
                best_cost = top_costs[0].item()
                best_sequence = elite_sequences[0]
        
        return best_sequence[0].item(), best_sequence, best_cost
