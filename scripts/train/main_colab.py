import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import random
import csv
import time

from env.gridworld import GridWorldEnv
from modules.perception import Perception
from modules.configurator import Configurator
from modules.world_model import WorldModel
from modules.cost import Cost
from modules.actor import Actor
from modules.memory import ShortTermMemory
from modules.sigreg import SIGReg

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_synthetic_target_obs(env, target_type='target'):
    obs = torch.zeros((4, env.size, env.size), dtype=torch.float32)
    for o in env.obstacles:
        obs[0, o[0], o[1]] = 1.0
    obs[1, env.target_pos[0], env.target_pos[1]] = 1.0
    obs[2, env.station_pos[0], env.station_pos[1]] = 1.0
    if target_type == 'target':
        obs[3, env.target_pos[0], env.target_pos[1]] = 1.0
    elif target_type == 'station':
        obs[3, env.station_pos[0], env.station_pos[1]] = 1.0
    return obs

def main():
    device = torch.device("cpu")
    print(f"Initialisation des modules (Deep-JEPA) sur {device}...")
    env = GridWorldEnv(size=10, max_energy=100)
    
    latent_dim = 32
    hidden_dim = 128
    
    # ResNet Encoder
    perception = Perception(in_channels=4, latent_dim=latent_dim).to(device)
    
    # Target Encoder (EMA)
    target_encoder = copy.deepcopy(perception).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False
        
    configurator = Configurator(latent_dim=latent_dim)
    # RNN World Model
    world_model = WorldModel(latent_dim=latent_dim, action_dim=4, hidden_dim=hidden_dim).to(device)
    cost = Cost(latent_dim=latent_dim).to(device)
    
    # --- Phase 1 : Target Critic (EMA) pour stabiliser le TD-Learning ---
    target_cost = copy.deepcopy(cost).to(device)
    for param in target_cost.parameters():
        param.requires_grad = False
    
    # Phase 3 : CEM avec compromis exploration/coût (500 séquences, horizon 10)
    actor = Actor(action_dim=4, num_sequences=500, horizon=10, cem_iterations=10, elite_size=50)
    
    # Replay Buffer
    memory = ShortTermMemory(capacity=10000)
    sigreg = SIGReg().to(device)
    
    # Phase 3 : LR ajustés (WM ralenti car déjà convergé, Critic accéléré)
    optimizer_wm = optim.Adam(list(world_model.parameters()) + list(perception.parameters()), lr=5e-5)
    optimizer_critic = optim.Adam(cost.critic.parameters(), lr=3e-4)
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "agent_checkpoint.pth")
    
    # --- Logger CSV ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    csv_headers = [
        "timestamp", "episode", "success_rate", "avg_steps",
        "l_pred", "l_sigreg", "l_critic",
        "v_target", "v_agent", "delta_v",
        "latent_dist", "latent_std"
    ]
    # Créer le fichier avec headers si il n'existe pas
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(csv_headers)
    
    start_episode = 0
    if os.path.exists(checkpoint_path):
        print(f"Chargement des poids depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        perception.load_state_dict(checkpoint['perception'])
        target_encoder.load_state_dict(checkpoint['target_encoder'])
        world_model.load_state_dict(checkpoint['world_model'])
        cost.load_state_dict(checkpoint['cost'])
        # Phase 1 : Initialiser le Target Critic avec les poids du Critic chargé
        target_cost.load_state_dict(checkpoint['cost'])
        # On recrée les optimizers avec les nouveaux LR (Phase 3)
        # Les anciens optimizer states ne sont plus compatibles avec les nouveaux LR
        start_episode = checkpoint.get('episode', 0)
        print(f"Modèles restaurés avec succès ! Reprise à l'épisode {start_episode}.")
    else:
        print("Aucun modèle existant trouvé. L'agent part de zéro.")
        
    num_episodes = 25000
    batch_size = 64
    seq_len = 8 # BPTT length
    exploration_episodes = 15000
    
    print("Début de l'entraînement (Génération Procédurale)...")
    print(f"  Phase 1 (Exploration): épisodes 1-{exploration_episodes}")
    print(f"  Phase 2 (Planification): épisodes {exploration_episodes+1}-{num_episodes}")
    print("=" * 70)
    
    recent_successes = 0
    avg_loss_pred = 0.0
    avg_loss_sigreg = 0.0
    avg_loss_critic = 0.0
    log_count = 0
    total_steps_in_window = 0
    
    for episode in range(start_episode, num_episodes):
        x_t = env.reset().to(device)
        
        perception.eval()
        world_model.eval()
        cost.eval()
        
        with torch.no_grad():
            target_obs = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
            station_obs = create_synthetic_target_obs(env, 'station').unsqueeze(0).to(device)
            s_target = perception(target_obs)
            s_station = perception(station_obs)
            configurator.set_goals(s_target, s_station)
        
        # Initialiser l'état caché du RNN pour le début de l'épisode
        h_t = world_model.init_hidden(1, device=device)
        
        for step in range(200):  # max steps
            x_t_tensor = x_t.unsqueeze(0)
            with torch.no_grad():
                s_t = perception(x_t_tensor)
                
            s_goal, _, _, _ = configurator.get_configuration(env.energy)
            
            # Action Selection
            if episode < exploration_episodes:
                a_t = random.randint(0, 3)
            else:
                # L'acteur planifie en projetant h_t dans le futur (Boussole + Critique)
                a_t, _, _ = actor.plan(s_t, h_t, world_model, cost, s_goal)
                
                if random.random() < 0.10: # Epsilon-greedy
                    a_t = random.randint(0, 3)
            
            # Interaction environnement
            x_next, reward, done = env.step(a_t)
            x_next_tensor = x_next.unsqueeze(0).to(device)
            
            # Phase 2 : Reward Scaling (ramener les récompenses dans [-1, 1])
            scaled_reward = reward / 100.0
            
            # Stockage dans le Replay Buffer (avec récompense normalisée)
            memory.push(x_t_tensor, a_t, x_next_tensor, scaled_reward, done)
            
            # Mise à jour de l'état caché avec l'action réelle (inférence)
            a_t_onehot = F.one_hot(torch.tensor([a_t], device=device), num_classes=4).float()
            with torch.no_grad():
                _, h_next = world_model.forward_step(s_t, a_t_onehot, h_t)
                
            x_t = x_next.to(device)
            h_t = h_next
            
            # Apprentissage BPTT (Backprop Through Time)
            if len(memory) > batch_size + seq_len:
                perception.train()
                world_model.train()
                cost.train()
                
                try:
                    # Extraction d'une séquence de longueur seq_len (ex: 8)
                    x_0_batch, a_seq_batch, x_next_seq_batch, reward_seq_batch, done_seq_batch = memory.sample_sequences(batch_size, seq_len=seq_len)
                    
                    x_0_batch = x_0_batch.to(device)
                    a_seq_batch = a_seq_batch.to(device)
                    x_next_seq_batch = x_next_seq_batch.to(device)
                    reward_seq_batch = reward_seq_batch.to(device)
                    done_seq_batch = done_seq_batch.to(device)
                    
                    a_seq_onehot = F.one_hot(a_seq_batch, num_classes=4).float() # (B, T, 4)
                    
                    # 1. Encodeur Spatial (Perception sur s_0)
                    s_0_batch = perception(x_0_batch) # (B, latent_dim)
                    
                    # 2. Encodeur Cible (Target CNN) pour toute la séquence
                    B, T, C, H, W = x_next_seq_batch.size()
                    x_next_flat = x_next_seq_batch.view(B * T, C, H, W)
                    
                    with torch.no_grad():
                        s_next_target_flat = target_encoder(x_next_flat)
                        s_next_target = s_next_target_flat.view(B, T, -1) # (B, T, latent_dim)
                    
                    # 3. World Model (Prédiction Séquentielle RNN)
                    optimizer_wm.zero_grad()
                    optimizer_critic.zero_grad()
                    
                    s_preds, h_seq = world_model.forward_seq(s_0_batch, a_seq_onehot) # (B, T, latent_dim)
                    
                    # 4. Calcul des pertes
                    # JEPA Pred Loss sur la séquence entière
                    loss_pred = F.mse_loss(s_preds, s_next_target)
                    
                    # SIGReg Loss (on l'applique sur s_0_batch pour structurer l'encodeur)
                    loss_sigreg = sigreg(s_0_batch)
                    
                    loss_wm = loss_pred + 1.0 * loss_sigreg
                    
                    # Gradient Clipping pour le RNN
                    loss_wm.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
                    
                    # 5. Entraînement du Critique (Sur la séquence entière)
                    # V(s_t) = cost + gamma * V(s_{t+1})
                    s_t_flat = s_preds.detach().view(B * T, -1)
                    s_next_target_flat_detached = s_next_target.view(B * T, -1)
                    
                    step_cost_flat = -reward_seq_batch.view(B * T, 1)
                    done_flat = done_seq_batch.view(B * T, 1)
                    
                    V_t = cost(s_t_flat)
                    # Phase 1 : Utiliser le TARGET Critic pour évaluer V(s_{t+1})
                    with torch.no_grad():
                        V_next = target_cost(s_next_target_flat_detached)
                        
                    target_value = step_cost_flat + 0.9 * V_next * (1.0 - done_flat)
                    loss_critic = F.mse_loss(V_t, target_value)
                    
                    loss_critic.backward()
                    
                    optimizer_wm.step()
                    optimizer_critic.step()
                    
                    # Métriques
                    avg_loss_pred += loss_pred.item()
                    avg_loss_sigreg += loss_sigreg.item()
                    avg_loss_critic += loss_critic.item()
                    log_count += 1
                    
                    # EMA Target Encoder
                    tau = 0.01
                    for p_tgt, p_main in zip(target_encoder.parameters(), perception.parameters()):
                        p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                    
                    # Phase 1 : EMA Target Critic
                    for p_tgt, p_main in zip(target_cost.parameters(), cost.parameters()):
                        p_tgt.data.mul_(1.0 - tau).add_(p_main.data, alpha=tau)
                        
                except ValueError:
                    pass # Pas encore assez de séquences valides, on saute l'apprentissage
                
                perception.eval()
                world_model.eval()
                cost.eval()
                
            if done:
                break
                
        # --- Tracking ---
        reached = env.agent_pos == env.target_pos
        if reached:
            recent_successes += 1
        total_steps_in_window += step + 1
        
        phase = "🟢 EXPLORE" if episode < exploration_episodes else "🟠 PLAN"
        status = "✅" if reached else "❌"
        print(f"{phase} Ep {episode+1}/{num_episodes} | Steps: {step+1:3d} | Energy: {env.energy:3d} | {status}")
        
        if (episode + 1) % 50 == 0:
            n = max(log_count, 1)
            success_rate = recent_successes / 50.0
            avg_steps = total_steps_in_window / 50.0
            
            # Barre de progression visuelle du taux de succès
            bar_len = 25
            filled = int(bar_len * success_rate)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            # Progression globale
            progress = (episode + 1) / num_episodes
            prog_filled = int(30 * progress)
            prog_bar = "▓" * prog_filled + "░" * (30 - prog_filled)
            
            print(f"\n  {'━'*60}")
            print(f"  📊 RAPPORT (Épisodes {episode+2-50} à {episode+1})")
            print(f"  {'━'*60}")
            print(f"  📈 Progression: [{prog_bar}] {progress*100:.1f}%")
            print(f"  {'─'*60}")
            
            # Taux de succès avec barre visuelle
            if success_rate >= 0.7:
                emoji = "🟢"
            elif success_rate >= 0.4:
                emoji = "🟡"
            else:
                emoji = "🔴"
            print(f"  {emoji} Succès: [{bar}] {recent_successes}/50 ({success_rate*100:.0f}%)")
            print(f"  🦶 Pas moyens: {avg_steps:.1f} / 200")
            
            print(f"  {'─'*60}")
            print(f"  📉 Pertes:")
            print(f"    L_pred (BPTT):   {avg_loss_pred/n:.6f}")
            print(f"    L_sigreg:        {avg_loss_sigreg/n:.6f}")
            
            # Indicateur visuel pour L_critic
            l_critic_val = avg_loss_critic / n
            if l_critic_val < 1.0:
                critic_status = "🟢 Excellent"
            elif l_critic_val < 5.0:
                critic_status = "🟡 En convergence"
            elif l_critic_val < 15.0:
                critic_status = "🟠 Instable"
            else:
                critic_status = "🔴 Diverge"
            print(f"    L_critic (TD):   {l_critic_val:.6f}  {critic_status}")
            
            # Analyse du Critique (V-values)
            with torch.no_grad():
                test_obs_target = create_synthetic_target_obs(env, 'target').unsqueeze(0).to(device)
                test_obs_agent = env.reset().unsqueeze(0).to(device)
                s_target_test = perception(test_obs_target)
                s_agent_test = perception(test_obs_agent)
                
                v_target = cost(s_target_test).item()
                v_agent = cost(s_agent_test).item()
                dist = torch.sum((s_target_test - s_agent_test)**2).item()
                s_std = s_target_test.std().item()
            
            print(f"  {'─'*60}")
            print(f"  🧠 Critique (V-values):")
            print(f"    V(target):  {v_target:+.4f}  (devrait être bas = bon état)")
            print(f"    V(agent):   {v_agent:+.4f}  (devrait être haut = loin du but)")
            print(f"    ΔV:         {v_agent - v_target:+.4f}  (positif = le Critic sait où aller)")
            
            print(f"  {'─'*60}")
            print(f"  🔬 Espace latent:")
            print(f"    Distance (target↔agent): {dist:.4f}")
            print(f"    Std vecteur latent:       {s_std:.4f}  {'✅' if 0.5 < s_std < 2.0 else '⚠️'}")
            
            # --- Écriture CSV ---
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    episode + 1,
                    f"{success_rate:.4f}",
                    f"{avg_steps:.1f}",
                    f"{avg_loss_pred/n:.6f}",
                    f"{avg_loss_sigreg/n:.6f}",
                    f"{l_critic_val:.6f}",
                    f"{v_target:.4f}",
                    f"{v_agent:.4f}",
                    f"{v_agent - v_target:.4f}",
                    f"{dist:.4f}",
                    f"{s_std:.4f}"
                ])
            print(f"  {'━'*60}\n")
            
            recent_successes = 0
            avg_loss_pred = 0.0
            avg_loss_sigreg = 0.0
            avg_loss_critic = 0.0
            log_count = 0
            total_steps_in_window = 0
        
        if (episode + 1) % 50 == 0 or (episode + 1) == num_episodes:
            torch.save({
                'episode': episode + 1,
                'perception': perception.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'world_model': world_model.state_dict(),
                'cost': cost.state_dict(),
                'target_cost': target_cost.state_dict(),
                'optimizer_wm': optimizer_wm.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict()
            }, checkpoint_path)
            print(f"💾 Poids sauvegardés à l'épisode {episode + 1}.")

if __name__ == "__main__":
    main()
