'''
evaluate the trained LSTM autoencoder on test data
    1. load trained model, threshold and test data
    2. compute reconstruction errors on test window
    3. flag anomalies (error > threshold)
    4. calculate precision, recall F1
    5. plot reconstruction errors with flagged anomalies highlighted

outputs:
    results/anomaly_plot.png, visual showing reconstruction errors and deteced anomalies
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
 
from model import LSTMAutoencoder


def compute_errors(model: LSTMAutoencoder, X: np.ndarray) -> np.ndarray:
    #compute per window reconstruction errors
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1,2))
        errors = errors.cpu().numpy()

    return errors


def plot_anomalies(errors: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
    # plot reconstruction errors with detected anomalies highlighted
    fig, axes = plt.subplots(2, 1, figsize=(14,8), sharex=True)

    #top plot: reconstruction errors
    ax1 = axes[0]
    ax1.plot(errors, color="steelblue", linewidth=0.5, label='reconstruction error')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.2e})')
    ax1.set_ylabel('reconstruction error MSE')
    ax1.set_title('reconstruction error per window')
    ax1.legend()

    #bottom plot: actual vs predicted anomalies
    ax2 = axes[1]

    #true anomalies in green
    true_anomaly_idx = np.where(y_true == 1)[0]
    ax2.scatter(true_anomaly_idx, np.ones_like(true_anomaly_idx) * 1.0,
                color="green", alpha=0.4, s=4, label="Actual anomalies")
 
    # predicted anomalies in red
    pred_anomaly_idx = np.where(y_pred == 1)[0]
    ax2.scatter(pred_anomaly_idx, np.ones_like(pred_anomaly_idx) * 0.5,
                color="red", alpha=0.4, s=4, label="Detected anomalies")
 
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Anomaly")
    ax2.set_yticks([0.5, 1.0])
    ax2.set_yticklabels(["Detected", "Actual"])
    ax2.set_title("Actual vs Detected Anomalies")
    ax2.legend()
 
    plt.tight_layout()
 
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved plot to {out_path}")
 
def main() -> None:
    # load everything
    data_dir = Path("data/processed")
    model_dir = Path("models")
 
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    threshold = np.load(model_dir / "threshold.npy").item()
 
    model = LSTMAutoencoder(input_dim=1, hidden_dim=32, n_layers=1)
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
 
    print(f"Loaded model and threshold ({threshold:.2e})")
    print(f"Test windows: {X_test.shape[0]}, Anomalous: {y_test.sum():.0f}")
 
    #compute errors and predict
    errors = compute_errors(model, X_test)
    y_pred = (errors > threshold).astype(int)

    # metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
 
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
 

 
    # plot
    plot_anomalies(errors, y_test, y_pred, threshold)
 
 
if __name__ == "__main__":
    main()
 