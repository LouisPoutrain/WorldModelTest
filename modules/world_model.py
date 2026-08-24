import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h):
        stacked = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(stacked))
        r = torch.sigmoid(self.conv_r(stacked))
        stacked_h = torch.cat([x, r * h], dim=1)
        h_candidate = torch.tanh(self.conv_h(stacked_h))
        h_new = (1 - z) * h + z * h_candidate
        return h_new

class WorldModel(nn.Module):
    def __init__(self, latent_dim=16, action_dim=4, hidden_dim=32, spatial_size=10):
        super(WorldModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.proj_dim = 64
        self.action_encoder = nn.Embedding(action_dim, latent_dim)
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        
        # Projeter (s_t, a_t) vers l'espace d'entrée du ConvGRU
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_dim + latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Le cœur de la mémoire (ConvGRU)
        self.rnn_cell = ConvGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=3)
        
        # Prédire s_{t+1} à partir de l'état caché mis à jour h_t
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        )
        
        # Décodeur d'agent pour forcer la représentation latente à suivre l'agent
        self.agent_decoder = nn.Conv2d(latent_dim, 1, kernel_size=1)
        

        
    def decode_agent(self, s):
        # s: [B, C, H, W]
        # returns: [B, 1, H, W] probabilités d'agent
        if len(s.shape) == 5:
            B, T, C, H, W = s.shape
            s = s.view(B*T, C, H, W)
            return torch.sigmoid(self.agent_decoder(s)).view(B, T, 1, H, W)
        return torch.sigmoid(self.agent_decoder(s))
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, self.spatial_size, self.spatial_size, device=device)
        
    def forward_step(self, s_t, a_t, h_t):
        if h_t is None:
            h_t = self.init_hidden(s_t.size(0), s_t.device)
        if len(s_t.shape) == 3:
            s_t = s_t.unsqueeze(0)
        B, C, H, W = s_t.shape
        
        # a_t is now expected to be discrete [B], not one-hot!
        # wait, if a_t is discrete [B], what if it's one-hot?
        # In actor.py and train.py, we pass one_hot. So let's extract argmax to use Embedding.
        if len(a_t.shape) == 2 and a_t.shape[1] == self.action_dim:
            a_t_idx = torch.argmax(a_t, dim=1)
        elif len(a_t.shape) == 3 and a_t.shape[2] == self.action_dim:
            a_t_idx = torch.argmax(a_t, dim=2).squeeze(1) # [B]
        else:
            a_t_idx = a_t
            if len(a_t_idx.shape) == 0:
                a_t_idx = a_t_idx.unsqueeze(0)
                
        # Get action embedding and broadcast
        a_emb = self.action_encoder(a_t_idx) # [B, latent_dim]
        a_t_spatial = a_emb.view(B, self.latent_dim, 1, 1).expand(B, self.latent_dim, H, W)
        
        x = torch.cat([s_t, a_t_spatial], dim=1)
        x_proj = self.input_proj(x)
        
        # Mise à jour de la mémoire
        h_next = self.rnn_cell(x_proj, h_t)
        
        # Prédiction résiduelle
        delta = self.predictor(h_next)
        s_next = delta
        
        return s_next, h_next
        
    def forward_seq(self, s_0, a_seq, h_0=None):
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
            
            s_t = s_next
            
        return torch.cat(s_preds, dim=1), torch.cat(h_seq, dim=1)
