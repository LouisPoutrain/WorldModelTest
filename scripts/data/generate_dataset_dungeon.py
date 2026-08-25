import torch
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from env.gridworld import GridWorldEnv

def main():
    print("Génération du dataset Donjon 20x20...")
    num_grids = 20000 # On génère 20k grilles pour bien couvrir les configurations des pièces
    
    env = GridWorldEnv(size=20, obstacle_density=0.08) # Densité légèrement plus faible pour un 20x20
    
    dataset = []
    
    for _ in tqdm(range(num_grids)):
        obs = env.reset()
        dataset.append(obs.unsqueeze(0))
        
    dataset_tensor = torch.cat(dataset, dim=0)
    
    os.makedirs('dataset', exist_ok=True)
    torch.save(dataset_tensor, 'dataset/grids_dungeon.pt')
    
    print(f"✅ Dataset Donjon généré : {dataset_tensor.shape}")

if __name__ == "__main__":
    main()
