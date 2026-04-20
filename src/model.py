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
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # decoder: reconstructs the sequence from the compressed representation
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # output layer: maps hidden state back to original feature dimension
        self.output_layer = nn.Linear(hidden_dim,input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        forward pass

        args:
            x: input tensor of shape (batch_size, seq_len, input_dim)

        returns:
            reconstructed tensor of same shape as x
        '''
        batch_size, seq_len, _ = x.shape

        # encode
        # run input through encoder, keep only final hidden state
        _, (hidden, cell) = self.encoder(x)

        # decode
        # feed the input to the decoder, initialized with encoders final state
        decoder_output, _ = self.decoder(x, (hidden, cell))

        # reconstruct
        # map each timesteps hidden state back to the original dimension
        reconstructed = self.output_layer(decoder_output)

        return reconstructed