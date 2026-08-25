import torch
import heapq
import numpy as np

class MacroPlanner:
    def __init__(self, waypoint_lookahead=5):
        self.lookahead = waypoint_lookahead
        
    def a_star(self, grid, start, goal, env=None):
        # grid: numpy array, 1 = wall, 0 = free
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_node = (current[0] + dx, current[1] + dy)
                if 0 <= next_node[0] < grid.shape[0] and 0 <= next_node[1] < grid.shape[1]:
                    if grid[next_node[0], next_node[1]] == 1:
                        continue # Wall
                    
                    # Brider A* : si env est fourni, on ne peut pas changer de pièce, SAUF si le next_node est le goal (la porte)
                    if env is not None:
                        start_room = env.get_current_room(start)
                        next_room = env.get_current_room(next_node)
                        if start_room != -1 and next_room != -1 and next_room != start_room and next_node != goal:
                            continue # Impossible de traverser les pièces
                            
                    new_cost = cost_so_far[current] + 1
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + heuristic(goal, next_node)
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current
                        
        if goal not in came_from:
            return [] # No path
            
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def get_path(self, start, goal, env):
        start = tuple(start)
        goal = tuple(goal)
        grid = env.render()
        walls = (grid == 1).astype(int)
        path = self.a_star(walls, start, goal, env)
        return path
    def get_waypoint_obs(self, x_t):
        """
        x_t: (1, 4, 10, 10) tensor
        Returns: x_waypoint (1, 4, 10, 10) tensor
        """
        obs = x_t[0].cpu().numpy() # (4, 10, 10)
        walls = obs[0]
        goals = obs[1]
        agents = obs[3]
        
        # Find positions
        agent_pos = tuple(np.argwhere(agents == 1)[0]) if np.any(agents == 1) else (0,0)
        goal_pos = tuple(np.argwhere(goals == 1)[0]) if np.any(goals == 1) else (0,0)
        
        path = self.a_star(walls, agent_pos, goal_pos)
        
        if not path:
            return x_t # Fallback
            
        # Pick waypoint
        waypoint_idx = min(self.lookahead - 1, len(path) - 1)
        waypoint_pos = path[waypoint_idx]
        
        # Create new observation with agent at waypoint
        x_waypoint = x_t.clone()
        x_waypoint[0, 3, :, :] = 0 # Clear agent
        x_waypoint[0, 3, waypoint_pos[0], waypoint_pos[1]] = 1 # Set agent at waypoint
        
        return x_waypoint
