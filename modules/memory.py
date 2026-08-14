import random
import torch

class ShortTermMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, x_t, a_t, x_next, reward, done):
        """Stocke une transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        # x_t et x_next doivent rester sur CPU ou être convertis
        # pour ne pas exploser la VRAM
        self.buffer[self.position] = (
            x_t.cpu(), 
            a_t, 
            x_next.cpu(), 
            reward, 
            done
        )
        self.position = (self.position + 1) % self.capacity
        
    def sample_transitions(self, batch_size):
        """Échantillonnage classique (1 pas)."""
        batch = random.sample(self.buffer, batch_size)
        x_batch = torch.cat([b[0] for b in batch], dim=0)
        a_batch = torch.tensor([b[1] for b in batch], dtype=torch.long)
        x_next_batch = torch.cat([b[2] for b in batch], dim=0)
        reward_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32).unsqueeze(1)
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)
        
        return x_batch, a_batch, x_next_batch, reward_batch, done_batch

    def sample_sequences(self, batch_size, seq_len=8):
        """
        Extrait des séquences de longueur 'seq_len' (BPTT).
        Retourne :
        - x_0_batch: (B, C, H, W) L'observation initiale (t=0)
        - a_seq_batch: (B, T) Les actions de t=0 à T-1
        - x_next_seq_batch: (B, T, C, H, W) Les observations cibles de t=1 à T
        - reward_seq_batch: (B, T)
        - done_seq_batch: (B, T)
        """
        # On ne peut tirer que des séquences qui ne dépassent pas la fin du buffer
        # et qui ne "traversent" pas le curseur (position) si le buffer a wrappé.
        # Pour faire simple : on tire un indice de départ valide.
        valid_indices = []
        n_items = len(self.buffer)
        
        for i in range(n_items - seq_len):
            # Vérifier que la séquence ne coupe pas la fin d'un épisode (done)
            # et ne coupe pas l'endroit de wrap
            is_valid = True
            for j in range(i, i + seq_len - 1):
                if self.buffer[j][4]: # done is True au milieu de la séquence
                    is_valid = False
                    break
            
            # Éviter que l'indice de wrap soit au milieu de la séquence
            if self.position > i and self.position < i + seq_len:
                is_valid = False
                
            if is_valid:
                valid_indices.append(i)
                
        if len(valid_indices) < batch_size:
            # Pas assez de séquences valides, fallback sur des transitions simples ou pad (rare si grand buffer)
            raise ValueError("Pas assez de séquences valides dans la mémoire.")
            
        start_indices = random.sample(valid_indices, batch_size)
        
        x_0_batch = []
        a_seq_batch = []
        x_next_seq_batch = []
        reward_seq_batch = []
        done_seq_batch = []
        
        for idx in start_indices:
            x_0_batch.append(self.buffer[idx][0])
            
            a_seq = []
            x_next_seq = []
            r_seq = []
            d_seq = []
            
            for step in range(seq_len):
                _, a, x_next, r, d = self.buffer[idx + step]
                a_seq.append(a)
                x_next_seq.append(x_next)
                r_seq.append(r)
                d_seq.append(d)
                
            a_seq_batch.append(torch.tensor(a_seq, dtype=torch.long))
            x_next_seq_batch.append(torch.cat(x_next_seq, dim=0).unsqueeze(0)) # (1, T, C, H, W)
            reward_seq_batch.append(torch.tensor(r_seq, dtype=torch.float32))
            done_seq_batch.append(torch.tensor(d_seq, dtype=torch.float32))
            
        return (
            torch.cat(x_0_batch, dim=0),                    # (B, C, H, W)
            torch.stack(a_seq_batch, dim=0),                # (B, T)
            torch.cat(x_next_seq_batch, dim=0),             # (B, T, C, H, W)
            torch.stack(reward_seq_batch, dim=0),           # (B, T)
            torch.stack(done_seq_batch, dim=0)              # (B, T)
        )

    def __len__(self):
        return len(self.buffer)
