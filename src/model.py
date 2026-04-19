'''
architecture:
    encoder: LSTM compresses a 24 step sequence into a fixed size hidden state
    decoder: LSTM reconstructs the original sequence from a hidden state

model learns to reconstruct normal price patterns, 
anomalous inputs produce high reconstruction error, which is used as the anomaly score
'''

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, n_layers: int = 1):
        '''
        args:
            input_dim: number of features per timestep (1 = just price)
            hidden_dim: size of the LSTM hidden state (the bottleneck)
            n_layers: number of stacked LSTM layers
        '''
    super().__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    # encoder: reads the input sequence and compresses it

    