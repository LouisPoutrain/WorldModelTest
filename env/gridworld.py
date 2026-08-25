import torch
import numpy as np
import random
from collections import deque

class GridWorldEnv:
    def __init__(self, size=20, max_energy=200, procedural=True, obstacle_density=0.1):
        self.size = size
        self.max_energy = max_energy
        self.action_space = 4 # 0: Up, 1: Down, 2: Left, 3: Right
        self.procedural = procedural
        self.obstacle_density = obstacle_density
        
        self.reset()
        
    def _bfs_reachable(self, start, end):
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
        
    def _generate_dungeon_grid(self):
        """Génère un donjon 20x20 divisé en 4 pièces par une croix centrale."""
        for attempt in range(50):
            self.obstacles = []
            
            # Murs centraux
            mid_y, mid_x = self.size // 2, self.size // 2
            
            for i in range(self.size):
                self.obstacles.append([mid_y, i]) # Mur horizontal
                self.obstacles.append([i, mid_x]) # Mur vertical
                
            # Retirer les doublons (le centre)
            self.obstacles = [list(x) for x in set(tuple(x) for x in self.obstacles)]
            
            # Portes (1 trou aléatoire par segment de mur)
            doors = []
            # Porte TL -> TR (Mur vertical haut)
            doors.append([random.randint(1, mid_y - 2), mid_x])
            # Porte BL -> BR (Mur vertical bas)
            doors.append([random.randint(mid_y + 2, self.size - 2), mid_x])
            # Porte TL -> BL (Mur horizontal gauche)
            doors.append([mid_y, random.randint(1, mid_x - 2)])
            # Porte TR -> BR (Mur horizontal droit)
            doors.append([mid_y, random.randint(mid_x + 2, self.size - 2)])
            
            self.doors = doors
            
            # Percer les portes dans les obstacles
            for door in doors:
                if door in self.obstacles:
                    self.obstacles.remove(door)
                    
            # Cellules valides pour le spawn (ni mur, ni porte)
            valid_cells = [(r, c) for r in range(self.size) for c in range(self.size) if [r, c] not in self.obstacles and [r, c] not in doors]
            
            # Ajouter des obstacles internes
            num_internal_obs = int(len(valid_cells) * self.obstacle_density)
            random.shuffle(valid_cells)
            internal_obstacles = [list(c) for c in valid_cells[:num_internal_obs]]
            self.obstacles.extend(internal_obstacles)
            
            # Mettre à jour les cellules valides
            valid_cells = [c for c in valid_cells[num_internal_obs:]]
            
            # Placer Agent, Cible, Station dans des pièces DIFFÉRENTES si possible, ou aléatoirement
            self.agent_pos = list(valid_cells[0])
            self.target_pos = list(valid_cells[1])
            self.station_pos = list(valid_cells[2])
            
            # Vérifier que tout est globalement connecté
            if self._bfs_reachable(self.agent_pos, self.target_pos) and self._bfs_reachable(self.agent_pos, self.station_pos):
                return
                
        print("Warning: Impossible de générer un donjon connecté après 50 essais.")
        
    def reset(self):
        self.energy = self.max_energy
        self.done = False
        
        if self.procedural:
            self._generate_dungeon_grid()
        else:
            self._generate_dungeon_grid() # Fallback temporaire
        
        return self.get_local_observation()
        
    def step(self, action):
        if self.done:
            return self.get_local_observation(), 0.0, True

        dy, dx = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
            
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
        reward = -1.0
        if collision:
            reward = -5.0
            
        if self.agent_pos == self.station_pos:
            self.energy = self.max_energy
            reward = 10.0
            
        if self.agent_pos == self.target_pos:
            reward = 100.0
            self.done = True
            
        if self.energy <= 0:
            self.done = True
        
        return self.get_local_observation(), reward, self.done

    def get_local_observation(self):
        obs = torch.zeros((4, self.size, self.size), dtype=torch.float32)
        for o in self.obstacles: obs[0, o[0], o[1]] = 1.0
        obs[1, self.target_pos[0], self.target_pos[1]] = 1.0
        obs[2, self.station_pos[0], self.station_pos[1]] = 1.0
        obs[3, self.agent_pos[0], self.agent_pos[1]] = 1.0
        return obs
        
    def get_current_room(self, pos):
        y, x = pos
        mid = self.size // 2
        if y < mid and x < mid: return 0
        if y < mid and x > mid: return 1
        if y > mid and x < mid: return 2
        if y > mid and x > mid: return 3
        return -1 # Sur un mur/porte
        
    def get_visible_doors(self):
        """Retourne toutes les portes. Dans un donjon plus grand, on filtrerait par pièce courante."""
        return self.doors
        
    def render(self):
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        grid[self.target_pos[0], self.target_pos[1]] = 2
        grid[self.station_pos[0], self.station_pos[1]] = 3
        grid[self.agent_pos[0], self.agent_pos[1]] = 4
        return grid
