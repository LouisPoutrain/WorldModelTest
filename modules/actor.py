import random
import torch
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, action_dim=4, num_sequences=500, horizon=5, cem_iterations=10, elite_size=50, w_critic=0.0):
        super().__init__()
        self.action_dim = action_dim
        self.num_sequences = num_sequences
        self.horizon = horizon
        self.cem_iterations = cem_iterations
        self.elite_size = elite_size
        self.w_critic = w_critic

    def plan(self, s_t, h_t, world_model, cost, s_goal, w_goal):
        device = s_t.device
        
        action_mean = torch.zeros((self.horizon, self.action_dim), device=device)
        action_std = torch.ones((self.horizon, self.action_dim), device=device) * 2.0
        
        best_action = torch.randint(0, self.action_dim, (1,)).item()
        best_cost = float('inf')
        best_h_t = None
        
        for it in range(self.cem_iterations):
            noise = torch.randn((self.num_sequences, self.horizon, self.action_dim), device=device)
            action_samples = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise
            
            action_probs = F.softmax(action_samples, dim=-1)
            actions_discrete = torch.argmax(action_probs, dim=-1)
            actions_onehot = F.one_hot(actions_discrete, num_classes=self.action_dim).float()
            
            # s_t is [1, C, H, W]
            s_sim = s_t.expand(self.num_sequences, -1, -1, -1)
            
            if h_t is not None:
                h_sim = h_t.expand(self.num_sequences, -1, -1, -1).contiguous()
            else:
                h_sim = world_model.init_hidden(self.num_sequences, device=device)
            
            total_costs = torch.zeros(self.num_sequences, device=device)
            
            for t in range(self.horizon):
                a_t_sim = actions_onehot[:, t, :]
                s_sim, h_sim = world_model.forward_step(s_sim, a_t_sim, h_sim)
                
                # Distance to goal
                if cost is not None:
                    dist_to_goal = cost(s_sim, s_goal)
                else:
                    # Manhattan distance using the agent_decoder
                    agent_prob = world_model.decode_agent(s_sim)
                    B_sim = agent_prob.shape[0]
                    agent_idx = agent_prob.view(B_sim, -1).argmax(dim=1)
                    spatial_size = agent_prob.shape[-1]
                    pred_y, pred_x = agent_idx // spatial_size, agent_idx % spatial_size
                    
                    goal_prob = world_model.decode_agent(s_goal)
                    goal_idx = goal_prob.view(goal_prob.shape[0], -1).argmax(dim=1)
                    goal_y, goal_x = goal_idx // spatial_size, goal_idx % spatial_size
                    
                    dist_to_goal = torch.abs(pred_y - goal_y) + torch.abs(pred_x - goal_x)
                    
                total_costs += w_goal * dist_to_goal

            costs_np = total_costs.detach().cpu().numpy()
            elite_indices = torch.tensor(costs_np.argsort()[:self.elite_size], device=device)
            elite_actions = action_samples[elite_indices]
            
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0) + 1e-5
            
            if costs_np[elite_indices[0].item()] < best_cost:
                best_cost = costs_np[elite_indices[0].item()]
                best_action = actions_discrete[elite_indices[0], 0].item()
                
        return best_action, best_cost, best_h_t

class HierarchicalActor:
    def __init__(self, perception, critic, astar):
        self.perception = perception
        self.critic = critic
        self.astar = astar
        
    def plan(self, env, s_t, s_goal):
        """
        1. Trouve la position de l'agent et du but (via l'environnement local).
        2. Si dans la même pièce, A* direct vers le but.
        3. Sinon, liste les portes de la pièce, utilise le Critique pour choisir la meilleure.
        4. A* vers la meilleure porte.
        """
        agent_pos = env.agent_pos
        target_pos = env.target_pos
        
        agent_room = env.get_current_room(agent_pos)
        target_room = env.get_current_room(target_pos)
        
        # Si dans la même pièce ou sur une porte (room=-1)
        if agent_room == target_room or agent_room == -1 or target_room == -1:
            return self.astar.get_path(agent_pos, target_pos, env)
            
        # Pièces différentes : le Critique choisit la porte
        all_doors = env.get_visible_doors()
        
        # Filtrer pour ne garder que les portes de la pièce courante
        doors = []
        mid = env.size // 2
        for d in all_doors:
            dy, dx = d
            if agent_room == 0 and ((dy == mid and dx < mid) or (dx == mid and dy < mid)): doors.append(d)
            elif agent_room == 1 and ((dy == mid and dx > mid) or (dx == mid and dy < mid)): doors.append(d)
            elif agent_room == 2 and ((dy == mid and dx < mid) or (dx == mid and dy > mid)): doors.append(d)
            elif agent_room == 3 and ((dy == mid and dx > mid) or (dx == mid and dy > mid)): doors.append(d)
            
        # Si on ne trouve aucune porte (bizarre), on fallback sur toutes
        if not doors: doors = all_doors
        
        best_door = None
        min_cost = float('inf')
        
        for door in doors:
            # Générer une observation fictive où l'agent est sur la porte
            door_obs = env.get_local_observation().clone()
            door_obs[3, :, :] = 0.0 # Effacer l'agent
            door_obs[3, door[0], door[1]] = 1.0 # Placer sur la porte
            
            door_obs = door_obs.unsqueeze(0).to(next(self.perception.parameters()).device)
            s_door = self.perception(door_obs)
            
            # Le Critique évalue le coût depuis cette porte jusqu'au but final
            cost = self.critic(s_door, s_goal).item()
            
            if cost < min_cost:
                min_cost = cost
                best_door = door
                
        # Le Micro-Planner (A*) calcule le chemin local jusqu'à la porte choisie
        return self.astar.get_path(agent_pos, best_door, env)
