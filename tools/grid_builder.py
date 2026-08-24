import pygame
import sys
import os
import torch
import torch.nn.functional as F
import imageio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.macro_planner import MacroPlanner
from modules.actor import Actor

# --- Configuration ---
CELL_SIZE = 60
GRID_SIZE = 10
MARGIN = 20
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE + 2 * MARGIN, GRID_SIZE * CELL_SIZE + 2 * MARGIN + 100)

COLORS = {
    "bg": (30, 30, 30),
    "grid": (60, 60, 60),
    "free": (40, 40, 40),
    "wall": (150, 150, 150),
    "agent": (50, 150, 255),
    "target": (50, 255, 50),
    "station": (255, 200, 50),
    "text": (200, 200, 200),
    "waypoint": (255, 100, 255),
}

class GridBuilder:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("H-JEPA GridBuilder")
        self.font = pygame.font.SysFont(None, 24)
        
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.agent_pos = [0, 0]
        self.target_pos = [9, 9]
        self.station_pos = [5, 5]
        
        self.mode = "EDIT" # "EDIT" or "RUN"
        self.model_loaded = False
        self.waypoint = None
        self.start_pos = list(self.agent_pos)
        self.step_count = 0
        self.frames = []
        
    def load_models(self):
        print("Loading models...")
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.perception = Perception(in_channels=4, latent_dim=16).to(self.device)
        self.world_model = WorldModel(latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10).to(self.device)
        
        checkpoint = torch.load('checkpoints/agent_h_jepa.pth', map_location=self.device)
        self.perception.load_state_dict(checkpoint['perception'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        
        self.perception.eval()
        self.world_model.eval()
        
        self.macro_planner = MacroPlanner(waypoint_lookahead=1) # Horizon 2 for max success
        self.actor = Actor(action_dim=4, num_sequences=1024, horizon=1, cem_iterations=3, elite_size=100, w_critic=0.0)
        
        self.env = GridWorldEnv(size=GRID_SIZE, max_energy=500, procedural=False)
        self.env.reset()
        
        # Inject custom grid
        obstacles = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    obstacles.append([y, x])
                    
        self.env.obstacles = obstacles
        self.start_pos = list(self.agent_pos)
        self.env.agent_pos = list(self.agent_pos)
        self.env.target_pos = list(self.target_pos)
        self.env.station_pos = list(self.station_pos)
        
        self.h_t = None
        self.step_count = 0
        self.frames = [] # Reset frames
        self.model_loaded = True
        print("Models loaded and Environment initialized!")

    def run_step(self):
        if self.env.done:
            return
            
        x_t = self.env.get_local_observation().unsqueeze(0).to(self.device)
        x_waypoint = self.macro_planner.get_waypoint_obs(x_t).to(self.device)
        
        self.waypoint = tuple(torch.nonzero(x_waypoint[0, 3])[0].tolist())
        
        with torch.no_grad():
            s_t = self.perception(x_t)
            s_waypoint = self.perception(x_waypoint)
            
        best_action, _, best_h_t = self.actor.plan(s_t, self.h_t, self.world_model, None, s_waypoint, w_goal=1.0)
        
        _, reward, done = self.env.step(best_action)
        self.agent_pos = self.env.agent_pos
        self.step_count += 1
        
        a_t_onehot = F.one_hot(torch.tensor([best_action]), num_classes=4).float().to(self.device)
        with torch.no_grad():
            _, self.h_t = self.world_model.forward_step(s_t, a_t_onehot, self.h_t)

    def save_gif(self):
        if not self.frames:
            print("No frames to save!")
            return
            
        os.makedirs("runs", exist_ok=True)
        
        # Find next available run number
        i = 1
        while os.path.exists(f"runs/run_{i}.gif"):
            i += 1
            
        filename = f"runs/run_{i}.gif"
        print(f"Saving {filename}...")
        imageio.mimsave(filename, self.frames, fps=3)
        print(f"{filename} saved!")

    def draw(self):
        self.screen.fill(COLORS["bg"])
        
        # Draw instructions
        if self.mode == "EDIT":
            inst = "Click: Wall | A: Agent | T: Target | C: Clear Grid | ENTER: Run"
        else:
            status = "Finished!" if getattr(self, 'env', None) and self.env.done else "Running..."
            inst = f"Steps: {self.step_count} | {status} | R: Reset | G: Export GIF"
            
        text = self.font.render(inst, True, COLORS["text"])
        self.screen.blit(text, (MARGIN, 10))
        
        # Draw grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(MARGIN + x * CELL_SIZE, MARGIN + 40 + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Base color
                color = COLORS["free"]
                if self.grid[y][x] == 1:
                    color = COLORS["wall"]
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)
                
                # Entities
                center = (MARGIN + x * CELL_SIZE + CELL_SIZE // 2, MARGIN + 40 + y * CELL_SIZE + CELL_SIZE // 2)
                if [y, x] == list(self.target_pos):
                    pygame.draw.circle(self.screen, COLORS["target"], center, CELL_SIZE // 3)
                elif [y, x] == list(self.station_pos):
                    pygame.draw.rect(self.screen, COLORS["station"], (center[0]-10, center[1]-10, 20, 20))
                
                if [y, x] == list(self.agent_pos):
                    pygame.draw.circle(self.screen, COLORS["agent"], center, CELL_SIZE // 2 - 4)
                    
                if self.mode == "RUN" and self.waypoint and [y, x] == list(self.waypoint):
                    pygame.draw.circle(self.screen, COLORS["waypoint"], center, 5)

        pygame.display.flip()
        
        # Save frame if in run mode
        if self.mode == "RUN":
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2)) # Pygame uses (x,y,c), imageio needs (y,x,c)
            self.frames.append(frame)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if self.mode == "EDIT":
                if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
                    mx, my = pygame.mouse.get_pos()
                    x = (mx - MARGIN) // CELL_SIZE
                    y = (my - MARGIN - 40) // CELL_SIZE
                    
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_a]:
                            self.agent_pos = [y, x]
                            self.grid[y][x] = 0
                        elif keys[pygame.K_t]:
                            self.target_pos = [y, x]
                            self.grid[y][x] = 0
                        elif keys[pygame.K_s]:
                            self.station_pos = [y, x]
                            self.grid[y][x] = 0
                        else:
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                self.grid[y][x] = 1 - self.grid[y][x]
                                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.mode = "RUN"
                        self.load_models()
                    elif event.key == pygame.K_c:
                        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            elif self.mode == "RUN":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.mode = "EDIT"
                        self.agent_pos = list(self.start_pos)
                        self.waypoint = None
                    elif event.key == pygame.K_g:
                        self.save_gif()
                        
        return True

    def loop(self):
        clock = pygame.time.Clock()
        run_timer = 0
        
        running = True
        self.draw() # Initial draw
        while running:
            running = self.handle_events()
            
            needs_draw = False
            if self.mode == "RUN" and self.model_loaded:
                run_timer += clock.get_time()
                if run_timer > 300: # 1 step every 300ms
                    self.run_step()
                    run_timer = 0
                    needs_draw = True
            elif self.mode == "EDIT":
                needs_draw = True
                
            if needs_draw:
                self.draw()
            clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    app = GridBuilder()
    app.loop()
