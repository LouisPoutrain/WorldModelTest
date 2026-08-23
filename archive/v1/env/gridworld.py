import torch
import numpy as np
import random
from collections import deque

class GridWorldEnv:
    def __init__(self, size=10, max_energy=100, procedural=True, obstacle_density=0.15):
        self.size = size
        self.max_energy = max_energy
        self.action_space = 4 # 0: Up, 1: Down, 2: Left, 3: Right
        self.procedural = procedural
        self.obstacle_density = obstacle_density
        
        self.reset()
        
    def _bfs_reachable(self, start, end):
        """Vérifie qu'un chemin existe entre start et end via BFS."""
        visited = set()
        queue = deque([tuple(start)])
        visited.add(tuple(start))
        obstacle_set = set(tuple(o) for o in self.obstacles)
        
        while queue:
            y, x = queue.popleft()
            if (y, x) == tuple(end):
                return True
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.size and 0 <= nx < self.size and (ny,nx) not in visited and (ny,nx) not in obstacle_set:
                    visited.add((ny,nx))
                    queue.append((ny,nx))
        return False
        
    def _generate_random_grid(self):
        """Génère une grille aléatoire avec des obstacles, une cible et une station accessibles."""
        for attempt in range(50):
            # Positions aléatoires pour l'agent, la cible et la station
            all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            random.shuffle(all_cells)
            
            self.agent_pos = list(all_cells[0])
            self.target_pos = list(all_cells[1])
            self.station_pos = list(all_cells[2])
            
            reserved = {all_cells[0], all_cells[1], all_cells[2]}
            
            # Placer des obstacles aléatoirement
            num_obstacles = int(self.size * self.size * self.obstacle_density)
            available = [c for c in all_cells[3:] if c not in reserved]
            random.shuffle(available)
            self.obstacles = [list(c) for c in available[:num_obstacles]]
            
            # Vérifier que la cible ET la station sont accessibles depuis l'agent
            if self._bfs_reachable(self.agent_pos, self.target_pos) and self._bfs_reachable(self.agent_pos, self.station_pos):
                return
        
        # Fallback : grille sans obstacles
        self.obstacles = []
        
    def _generate_fixed_grid(self):
        """Grille fixe originale (pour référence et débogage)."""
        self.agent_pos = [0, 0]
        self.target_pos = [self.size - 1, self.size - 1]
        self.station_pos = [self.size // 2, self.size // 2]
        
        self.obstacles = []
        for i in range(2, 8):
            self.obstacles.append([i, 3])
            self.obstacles.append([3, i])
        self.obstacles = [obs for obs in self.obstacles if obs != self.target_pos and obs != self.station_pos]
        
    def reset(self):
        self.energy = self.max_energy
        self.done = False
        
        if self.procedural:
            self._generate_random_grid()
        else:
            self._generate_fixed_grid()
        
        return self.get_local_observation()
        
    def step(self, action):
        if self.done:
            return self.get_local_observation(), 0.0, True

        # Action: 0: Up (-1, 0), 1: Down (+1, 0), 2: Left (0, -1), 3: Right (0, +1)
        dy, dx = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1
            
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx
        
        collision = False
        if 0 <= new_y < self.size and 0 <= new_x < self.size:
            if [new_y, new_x] not in self.obstacles:
                self.agent_pos = [new_y, new_x]
            else:
                collision = True
        else:
            collision = True
            
        self.energy -= 1
        
        # Reward signal
        reward = -1.0  # Coût de base par pas (encourage l'efficacité)
        
        if collision:
            reward = -5.0  # Pénalité pour collision
        
        # Check station
        if self.agent_pos == self.station_pos:
            self.energy = self.max_energy
            reward = 10.0  # Récompense pour recharge
            
        # Check target
        if self.agent_pos == self.target_pos:
            reward = 100.0  # Récompense pour atteindre la cible
            self.done = True
            
        if self.energy <= 0:
            self.done = True
        
        return self.get_local_observation(), reward, self.done

    def get_local_observation(self):
        # Retourne une observation GLOBALE (4, 10, 10)
        obs = torch.zeros((4, self.size, self.size), dtype=torch.float32)
        
        # Channel 0: Obstacles
        for o in self.obstacles:
            obs[0, o[0], o[1]] = 1.0
            
        # Channel 1: Target
        obs[1, self.target_pos[0], self.target_pos[1]] = 1.0
        
        # Channel 2: Station
        obs[2, self.station_pos[0], self.station_pos[1]] = 1.0
        
        # Channel 3: Agent
        obs[3, self.agent_pos[0], self.agent_pos[1]] = 1.0
                    
        return obs
        
    def render(self):
        """
        Returns a 2D numpy array of the global grid state.
        0: Empty, 1: Obstacle (Wall), 2: Target, 3: Station, 4: Agent
        """
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
            
        grid[self.target_pos[0], self.target_pos[1]] = 2
        grid[self.station_pos[0], self.station_pos[1]] = 3
        grid[self.agent_pos[0], self.agent_pos[1]] = 4
        
        return grid
