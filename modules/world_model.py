import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=128):
        super(WorldModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Projeter (s_t, a_t) vers l'espace d'entrée du RNN
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Le cœur de la mémoire (RNN)
        # Il prend (s_t, a_t) projeté et l'état caché précédent h_{t-1}
        self.rnn_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # Prédire s_{t+1} à partir de l'état caché mis à jour h_t
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def init_hidden(self, batch_size, device):
        """Retourne un état caché initial (vecteur de zéros)."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def forward_step(self, s_t, a_t, h_t):
        """
        Passe un pas temporel. Utilisé pour l'inférence et la planification.
        s_t: (B, latent_dim)
        a_t: (B, action_dim) one-hot
        h_t: (B, hidden_dim) l'état caché courant
        
        Returns:
            s_next: (B, latent_dim) l'état prédit
            h_next: (B, hidden_dim) le nouvel état caché
        """
        if len(s_t.shape) == 1:
            s_t = s_t.unsqueeze(0)
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
            
        x = torch.cat([s_t, a_t], dim=-1)
        x_proj = self.input_proj(x)
        
        # Mise à jour de la mémoire
        h_next = self.rnn_cell(x_proj, h_t)
        
        # Prédiction (résiduelle par rapport à s_t pour aider l'apprentissage)
        delta = self.predictor(h_next)
        s_next = s_t + delta
        
        return s_next, h_next
        
    def forward_seq(self, s_0, a_seq, h_0=None):
        """
        Traite une séquence entière. Utilisé pour l'entraînement (BPTT).
        s_0: (B, latent_dim) L'état latent au début de la séquence (t=0)
        a_seq: (B, T, action_dim) Les actions prises aux temps t=0 à T-1
        h_0: (B, hidden_dim) État caché initial optionnel
        
        Returns:
            s_preds: (B, T, latent_dim) Les prédictions pour t=1 à T
            h_seq: (B, T, hidden_dim) Les états cachés pour t=1 à T
        """
        B, T, _ = a_seq.size()
        if h_0 is None:
            h_0 = self.init_hidden(B, s_0.device)
            
        s_t = s_0
        h_t = h_0
        s_preds = []
        h_seq = []
        
        for t in range(T):
            a_t = a_seq[:, t, :]
            s_next, h_t = self.forward_step(s_t, a_t, h_t)
            
            s_preds.append(s_next.unsqueeze(1))
            h_seq.append(h_t.unsqueeze(1))
            
            # Autoregressif : on utilise notre propre prédiction comme entrée pour le pas suivant
            # (Cela force le modèle à apprendre une dynamique robuste sur le long terme)
            s_t = s_next
            
        return torch.cat(s_preds, dim=1), torch.cat(h_seq, dim=1)
