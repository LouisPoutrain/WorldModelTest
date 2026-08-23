import random
import torch
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, action_dim=4, num_sequences=500, horizon=25, cem_iterations=10, elite_size=50, w_critic=0.0):
        super().__init__()
        self.action_dim = action_dim
        self.num_sequences = num_sequences
        self.horizon = horizon
        self.cem_iterations = cem_iterations
        self.elite_size = elite_size
        self.w_critic = w_critic

    def plan(self, s_t, h_t, world_model, cost, s_goal, w_goal):
        """Planifie la meilleure action en utilisant la Cross-Entropy Method (CEM) purifiée (V2)."""
        device = s_t.device
        
        # Initialisation de la distribution des actions (logits uniformes)
        action_mean = torch.zeros((self.horizon, self.action_dim), device=device)
        action_std = torch.ones((self.horizon, self.action_dim), device=device) * 2.0
        
        best_action = random_action = torch.randint(0, self.action_dim, (1,)).item()
        best_cost = float('inf')
        best_h_t = None
        
        for it in range(self.cem_iterations):
            # 1. Échantillonner N séquences d'actions depuis la distribution
            noise = torch.randn((self.num_sequences, self.horizon, self.action_dim), device=device)
            action_samples = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise
            
            # Gumbel-Softmax pour obtenir des actions discrètes dérivables (ou argmax pour CEM standard)
            action_probs = F.softmax(action_samples, dim=-1)
            actions_discrete = torch.argmax(action_probs, dim=-1) # (N, H)
            actions_onehot = F.one_hot(actions_discrete, num_classes=self.action_dim).float() # (N, H, 4)
            
            # 2. Simuler les trajectoires avec le World Model
            # On batch le simulateur : (N, H, D)
            s_sim = s_t.expand(self.num_sequences, -1)
            h_sim = world_model.init_hidden(self.num_sequences, device=device)
            # Copy current hidden state to all sequences
            if h_t is not None:
                h_sim = h_t.expand(self.num_sequences, -1).contiguous()
            
            total_costs = torch.zeros(self.num_sequences, device=device)
            
            for t in range(self.horizon):
                a_t_sim = actions_onehot[:, t, :]
                s_sim, h_sim = world_model.forward_step(s_sim, a_t_sim, h_sim)
                
                # Coût : Distance euclidienne vers le but dans l'espace latent
                dist_to_goal = torch.sum((s_sim - s_goal)**2, dim=-1)
                
                # Dans la V2, le World Model est censé simuler les murs correctement.
                # Donc on n'ajoute PLUS de pénalités manuelles (stagnation).
                # Le CEM trouvera de lui-même que taper un mur = pas de rapprochement du but.
                total_costs += w_goal * dist_to_goal
                
                # Si on utilisait un critique (w_critic > 0), ce qui n'est pas le cas en V2 pur
                if self.w_critic > 0.0 and cost is not None:
                    critic_cost = cost(s_sim).squeeze(-1)
                    total_costs += self.w_critic * critic_cost

            # 3. Sélectionner les "élites"
            costs_np = total_costs.detach().cpu().numpy()
            elite_indices = torch.tensor(costs_np.argsort()[:self.elite_size], device=device)
            elite_actions = action_samples[elite_indices] # (Elite, H, 4)
            
            # 4. Mettre à jour la distribution
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0) + 1e-5
            
            if costs_np[elite_indices[0].item()] < best_cost:
                best_cost = costs_np[elite_indices[0].item()]
                best_action = actions_discrete[elite_indices[0], 0].item()
                
        return best_action, best_cost, best_h_t
