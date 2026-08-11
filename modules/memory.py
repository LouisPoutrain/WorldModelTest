import random
import torch

class ShortTermMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, x_t, a_t, x_next, c_intr):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_t, a_t, x_next, c_intr)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_t_batch, a_t_batch, x_next_batch, c_intr_batch = zip(*batch)
        
        return (
            torch.cat(x_t_batch),
            torch.tensor(a_t_batch),
            torch.cat(x_next_batch),
            torch.cat(c_intr_batch)
        )
        
    def __len__(self):
        return len(self.buffer)
