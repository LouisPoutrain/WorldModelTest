import os
import sys
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

# Ajouter le parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.gridworld import GridWorldEnv

def bfs_distance(env, start, end):
    """Calcule la distance réelle (plus court chemin) entre start et end avec BFS."""
    visited = set()
    queue = deque([(tuple(start), 0)]) # (position, distance)
    visited.add(tuple(start))
    obstacle_set = set(tuple(o) for o in env.obstacles)
    
    while queue:
        (y, x), dist = queue.popleft()
        if (y, x) == tuple(end):
            return dist
            
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < env.size and 0 <= nx < env.size:
                if (ny, nx) not in visited and (ny, nx) not in obstacle_set:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), dist + 1))
                    
    return -1 # Pas de chemin trouvé

def generate_dataset(num_samples=100000, save_path="dataset/geodesic_data.pt"):
    print(f"🌍 Génération du dataset géodésique : {num_samples} échantillons")
    env = GridWorldEnv(size=10, max_energy=100, procedural=True)
    
    observations = []
    distances = []
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for _ in tqdm(range(num_samples)):
        obs = env.reset()
        dist = bfs_distance(env, env.agent_pos, env.target_pos)
        
        # Parfois la cible n'est pas accessible, on ignore ces cas
        if dist != -1:
            observations.append(obs.clone())
            distances.append(dist)
            
    observations = torch.stack(observations)
    distances = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1) # (N, 1)
    
    print(f"✅ Génération terminée. Exemples valides : {len(distances)} / {num_samples}")
    
    torch.save({
        'observations': observations,
        'distances': distances
    }, save_path)
    
    print(f"💾 Dataset sauvegardé dans {save_path}")

if __name__ == "__main__":
    generate_dataset(num_samples=50000) # 50k pour commencer, modifiable si besoin
