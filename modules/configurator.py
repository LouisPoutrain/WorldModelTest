import torch

class Configurator:
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim
        
        # We need a way to define the latent state for 'target' and 'station'
        # In a full system, these could be learned or extracted from memory.
        # Here, we will simulate them as fixed vectors for simplicity, or 
        # ideally, we would encode the target and station observations directly.
        # To make it clean, we'll allow passing s_target and s_station.
        self.s_target = torch.ones(1, latent_dim) # Placeholder
        self.s_station = -torch.ones(1, latent_dim) # Placeholder
        
    def set_goals(self, s_target, s_station):
        self.s_target = s_target
        self.s_station = s_station

    def get_configuration(self, energy):
        """
        Returns:
            s_goal: The target latent state
            w_energy: Weight for energy cost
            w_collision: Weight for collision cost
            w_goal: Weight for goal distance cost
        """
        w_collision = 1.0
        
        if energy <= 30:
            s_goal = self.s_station
            w_energy = 5.0
            w_goal = 2.0
        else:
            s_goal = self.s_target
            w_energy = 0.1
            w_goal = 1.0
            
        return s_goal, w_energy, w_collision, w_goal
