# Plan d'Implémentation : Donjon Multi-Pièce (Hierarchical JEPA)

Votre idée est brillante et correspond **exactement** à la définition d'un H-JEPA (Hierarchical Joint Embedding Predictive Architecture) de Yann LeCun ! 

Au lieu d'opposer A* et le Critique, nous allons les faire collaborer dans une vraie architecture hiérarchique :
- **Le Critique (High-Level)** évalue la stratégie globale : *Quelle porte dois-je prendre pour me rapprocher de l'objectif final ?*
- **A* (Low-Level)** exécute la tactique locale : *Quel est le plus court chemin géométrique pour atteindre cette porte depuis ma position, sans me cogner ?*

Pour justifier cette séparation, nous allons limiter la "vision" ou la "portée" de A*.

---

## Le Concept : Le Donjon Multi-Pièce

1. **L'Environnement (GridWorld 20x20)**
   - La grille est divisée en plusieurs "pièces" (ex: 4 pièces de 10x10) séparées par des murs infranchissables.
   - Les pièces sont connectées entre elles par des "Portes" (cases vides spécifiques).
   - *Limitation de A** : L'algorithme A* sera artificiellement bridé pour ne pouvoir planifier qu'**à l'intérieur de la pièce courante**. Il est incapable de trouver le chemin jusqu'à la cible finale si celle-ci est dans une autre pièce.

2. **Le Rôle du Critique JEPA (Macro-Planner)**
   - Le Critique a été entraîné sur l'espace latent global. Il sait estimer la "distance latente" (Cost-to-Go) entre n'importe quel état et la cible.
   - Le système identifie les portes accessibles dans la pièce courante.
   - Le système utilise le **Critique** pour évaluer chaque porte : $V(\text{Porte}_A, \text{Cible})$ vs $V(\text{Porte}_B, \text{Cible})$.
   - Le système choisit la porte avec le coût le plus faible (celle qui mène vers la bonne partie du donjon).

3. **Le Rôle de A* (Micro-Planner)**
   - Une fois la porte choisie par le Critique, A* calcule le chemin géométrique parfait pour esquiver les petits obstacles locaux et atteindre cette porte.

---

## Modifications Proposées

### 1. Environnement (`env/gridworld.py`)
- Étendre la taille de la grille (ex: `size=20`).
- Modifier `_generate_random_grid()` pour générer un layout de "Donjon" (4 pièces de 10x10 avec des murs croisés au centre, et des trous aléatoires faisant office de portes).
- Ajouter des obstacles aléatoires à l'intérieur des pièces.

### 2. Adaptation du World Model et du Critique
- Générer un nouveau dataset (`dataset_dungeon.pt`) de ces donjons.
- Ré-entraîner `agent_h_jepa.pth` sur cette grille 20x20. L'espace latent devra capturer la sémantique "Pièce A est connectée à Pièce B".
- Ré-entraîner `train_critic_td.py` (avec la Fitted Value Iteration) pour qu'il apprenne la vraie distance géodésique inter-pièces.

### 3. Nouvel Acteur Hiérarchique (`modules/actor.py`)
Créer une classe `HierarchicalActor` avec le pseudo-code suivant :
```python
def plan_hierarchical(env_state, target_state, critic, astar):
    if target_in_same_room():
        return astar.get_path(env_state, target_state)
    else:
        doors = get_visible_doors()
        best_door = None
        min_cost = inf
        
        # Le Critique évalue quelle porte est la meilleure
        for door in doors:
            door_latent = perception(door)
            cost = critic(door_latent, target_latent)
            if cost < min_cost:
                min_cost = cost
                best_door = door
                
        # A* exécute le chemin vers la porte
        return astar.get_path(env_state, best_door)
```

### 4. Évaluation (`eval_dungeon.py`)
Créer un script d'évaluation démontrant que :
- **A* tout seul (bridé)** : Échoue à atteindre la cible.
- **Critique tout seul (CEM)** : Échoue car le labyrinthe 20x20 est trop complexe pour CEM à horizon court.
- **H-JEPA (Critique + A*)** : Réussit avec un taux de 100%, prouvant la synergie de l'architecture.

---

## User Review Required

> [!IMPORTANT]
> Cette évolution est majeure et va demander de ré-entraîner les modèles (Perception, World Model et Critique) sur des grilles 20x20. 
> Êtes-vous d'accord avec ce design en "4 pièces" (qui prouve magnifiquement la synergie H-JEPA) pour remplacer les petites grilles 10x10 actuelles ?
