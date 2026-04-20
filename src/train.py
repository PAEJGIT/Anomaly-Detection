'''
train the LSTM autoencoder on normal price windows

    1. loads processed training data
    2. filter out anomalous windows (trains only on normal patterns)
    3. train the autoencoder to reconstruct normal windows
    4. calculate anomaly threshold from training reconstruction errors
    5. save model weights and threshold

outputs in models/
    autoencoder.pt: trained model weights
    threshold.npy: reconstruction error threshold for anomaly detection
'''


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from model import LSTMAutoencoder


def train_model(
        X_train: np.ndarray,
        hidden_dim: int = 32,
        n_layers: int = 1,
        epochs: int = 50,
        batch_size: int = 64,
        learinng_rate: float = 1e-3,
) -> tuple[LSTMAutoencoder, list[float]]:
    '''
    train the autoencoder on normal windows

    returns:
        model: trained LSTMautoencoder
        losses: list of average loss per epoch
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')

    #conver to pytorch tensor
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model, loss function and optimizer
    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learinng_rate)


    # training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in dataloader:
            # forward pass: reconstruct the input
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            # backward pass: update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches = 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {epochs} - Loss: {avg_loss:.6f}')

    return model, losses


def calculate_threshold(
        model: LSTMAutoencoder,
        X_train: np.ndarray,
        percentile: float = 95.0
) -> float:
    '''
    calculate the anomaly threshold from training reconstruction errors

    the threshold is set at a percentile of the training errors, any test window with reconstruction error above this threshold is flagged as anomalous
    '''
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(device)
        reconstructed = model(X_tensor)
        # mean squared error per window
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1,2))
        errors = errors.cpu().numpy()

    threshold = np.percentile(errors, percentile)
    print(f'Threshold (p{percentile:.0f}): {threshold:.10f}')
    print(f'training errors - min: {errors.min():.10f}, max: {errors.max():.10f}, mean: {errors.mean():.10f}')

    return threshold

def main() -> None:
    #load training data
    data_dir = Path('data/processed')
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')

    #keep only normal windows for training
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    print(f'Training on {X_train_normal.shape[0]} normal windows (removed {(~normal_mask).sum()}) anomalous')

    # train
    model, losses = train_model(X_train_normal)

    # calculate threshold
    threshold = calculate_threshold(model, X_train_normal)

    # save
    out_dir = Path('models')
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / 'autoencoder.pt')
    np.save(out_dir / 'threshold.npy', threshold)

    print(f'Saved model to {out_dir / 'autoencoder.pt'}')
    print(f'Saved model to {out_dir / 'threshold.npy'}')

if __name__ == '__main__':
    main()




