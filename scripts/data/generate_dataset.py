import os
import torch
import random
from collections import deque
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SIZE = 10

def bfs_reachable(agent_pos, target_pos, obstacles):
    """Vérifie qu'un chemin existe entre start et end via BFS."""
    visited = set()
    queue = deque([tuple(agent_pos)])
    visited.add(tuple(agent_pos))
    obstacle_set = set(tuple(o) for o in obstacles)
    
    while queue:
        y, x = queue.popleft()
        if (y, x) == tuple(target_pos):
            return True
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < SIZE and 0 <= nx < SIZE and (ny,nx) not in visited and (ny,nx) not in obstacle_set:
                visited.add((ny,nx))
                queue.append((ny,nx))
    return False

def generate_random_positions(obstacles):
    """Place l'agent, la cible et la station sur des cases vides."""
    all_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if [r, c] not in obstacles]
    random.shuffle(all_cells)
    return list(all_cells[0]), list(all_cells[1]), list(all_cells[2])

def generate_u_trap():
    """Génère un piège en U avec une orientation et une position aléatoires."""
    obstacles = []
    # Choisir le centre du U (en évitant les bords pour avoir la place)
    cy = random.randint(2, SIZE - 3)
    cx = random.randint(2, SIZE - 3)
    
    orientation = random.choice(["up", "down", "left", "right"])
    
    # Construire le U de taille 5x3 ou 3x5
    if orientation == "up": # Ouvert vers le haut
        for i in range(cx - 2, cx + 3): obstacles.append([cy + 1, i]) # Bas
        for i in range(cy - 1, cy + 2):
            obstacles.append([i, cx - 2]) # Gauche
            obstacles.append([i, cx + 2]) # Droite
            
    elif orientation == "down": # Ouvert vers le bas
        for i in range(cx - 2, cx + 3): obstacles.append([cy - 1, i]) # Haut
        for i in range(cy - 1, cy + 2):
            obstacles.append([i, cx - 2]) 
            obstacles.append([i, cx + 2])
            
    elif orientation == "left": # Ouvert vers la gauche
        for i in range(cy - 2, cy + 3): obstacles.append([i, cx + 1]) # Droite
        for i in range(cx - 1, cx + 2):
            obstacles.append([cy - 2, i]) # Haut
            obstacles.append([cy + 2, i]) # Bas
            
    elif orientation == "right": # Ouvert vers la droite
        for i in range(cy - 2, cy + 3): obstacles.append([i, cx - 1]) # Gauche
        for i in range(cx - 1, cx + 2):
            obstacles.append([cy - 2, i])
            obstacles.append([cy + 2, i])
            
    return obstacles, [cy, cx] # Retourne les obstacles et le "centre" du piège

def generate_maze():
    """Génère un labyrinthe par ajout de murs aléatoires (lignes horizontales/verticales)."""
    obstacles = []
    num_walls = random.randint(3, 6)
    
    for _ in range(num_walls):
        is_horizontal = random.random() > 0.5
        length = random.randint(3, 7)
        if is_horizontal:
            row = random.randint(1, SIZE - 2)
            col_start = random.randint(0, SIZE - length)
            for c in range(col_start, col_start + length):
                obstacles.append([row, c])
        else:
            col = random.randint(1, SIZE - 2)
            row_start = random.randint(0, SIZE - length)
            for r in range(row_start, row_start + length):
                obstacles.append([r, col])
                
    # Remove duplicates
    unique_obs = []
    for o in obstacles:
        if o not in unique_obs:
            unique_obs.append(o)
    return unique_obs

def generate_dense():
    """Grille aléatoire avec 20% à 35% d'obstacles."""
    density = random.uniform(0.2, 0.35)
    num_obstacles = int(SIZE * SIZE * density)
    all_cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    random.shuffle(all_cells)
    return [list(c) for c in all_cells[:num_obstacles]]

def main():
    print("🚀 Génération du Dataset de Grilles (V2)")
    num_grids = 10000
    dataset = []
    
    os.makedirs("dataset", exist_ok=True)
    
    pbar = tqdm(total=num_grids, desc="Génération")
    while len(dataset) < num_grids:
        # Choisir le type de grille (33% U-Trap, 33% Maze, 34% Dense)
        grid_type = random.random()
        
        if grid_type < 0.33:
            obstacles, center = generate_u_trap()
            agent_pos = center # Placer l'agent DANS le piège
            target_pos, station_pos, _ = generate_random_positions(obstacles + [agent_pos])
        elif grid_type < 0.66:
            obstacles = generate_maze()
            agent_pos, target_pos, station_pos = generate_random_positions(obstacles)
        else:
            obstacles = generate_dense()
            agent_pos, target_pos, station_pos = generate_random_positions(obstacles)
            
        # Vérifier la solvabilité
        if bfs_reachable(agent_pos, target_pos, obstacles) and bfs_reachable(agent_pos, station_pos, obstacles):
            dataset.append({
                'obstacles': obstacles,
                'agent_pos': agent_pos,
                'target_pos': target_pos,
                'station_pos': station_pos
            })
            pbar.update(1)
            
    pbar.close()
    
    # Sauvegarde
    save_path = "dataset/grids_v2.pt"
    torch.save(dataset, save_path)
    print(f"✅ Dataset généré avec succès !")
    print(f"📦 10 000 grilles sauvegardées dans '{save_path}'")
    print(f"💡 Pour charger : dataset = torch.load('{save_path}')")

if __name__ == "__main__":
    main()
