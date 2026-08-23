import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

def compute_bfs_distances(size, obstacles, target_pos):
    distances = np.full((size, size), 100.0, dtype=np.float32)
    distances[target_pos[0], target_pos[1]] = 0.0
    queue = deque([target_pos])
    
    obs_set = set(tuple(o) for o in obstacles)
    
    while queue:
        y, x = queue.popleft()
        d = distances[y, x]
        
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < size and 0 <= nx < size:
                if (ny, nx) not in obs_set and distances[ny, nx] == 100.0:
                    distances[ny, nx] = d + 1.0
                    queue.append((ny, nx))
                    
    return distances

def main():
    print("🚀 Génération du dataset Géodésique (Supervised Critic)")
    dataset_path = "dataset/grids_v2.pt"
    out_path = "dataset/grids_v2_bfs.pt"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Impossible de trouver {dataset_path}")
        return
        
    grids = torch.load(dataset_path)
    new_grids = []
    
    for grid in tqdm(grids, desc="Calcul BFS"):
        obstacles = grid['obstacles']
        target_pos = grid['target_pos']
        
        dist_matrix = compute_bfs_distances(10, obstacles, target_pos)
        
        new_grid = grid.copy()
        new_grid['distances'] = torch.from_numpy(dist_matrix)
        new_grids.append(new_grid)
        
    torch.save(new_grids, out_path)
    print(f"✅ {len(new_grids)} grilles traitées et sauvegardées dans {out_path}")

if __name__ == "__main__":
    main()
