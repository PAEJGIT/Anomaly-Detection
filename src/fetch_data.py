import numpy as np
import pandas as pd
from pathlib import Path

"""
    Creating syntethic data. 
    Base price, 40 EU/mwh
    Outputs .csv, colums: timestamp, price, is_anomaly
    Weekly seasonality
    Gaussian noise
    Injected anomaly spikes, random, labelled for evaluation
"""


def generate_prices(
    n_days: int = 365,
    base_price: float = 40.0,
    noise_std: float = 5.0,
    anomaly_fraction: float = 0.02,
    spike_range: tuple[float, float] = (80.0, 200.0),
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    # daily seasonality: two peaks

    hour = np.arange(n_hours) % 24
    daily = (
        6.0 * np.exp(-0.5 * ((hour - 9) / 2.0) ** 2)
        + 8.0 * np.exp(-0.5 * ((hour - 18) / 2.5 ) ** 2)
        -4.0 * np.exp(-0.5 * ((hour - 3) / 2.0 ) ** 2)
    )

    # weekly prices are 15% cheaper
    day_of_week = timestamps.dayofweek.to_numpy()   
    weekly = np.where(day_of_week >= 5, -0.15 * base_price, 0.0)

    # combine
    prices = base_price + daily + weekly + rng.normal(0, noise_std, n_hours)
    prices = np.clip(prices, 0, None) # prices cant go negative in this model


    # injecting anomalies
    is_anomaly = np.zeros(n_hours, dtype=int) # array of n-hour zeroes
    n_anomalies = int(n_hours * anomaly_fraction) # number of anomalies
    anomaly_indices = rng.choice(n_hours, size=n_anomalies, replace=False) # picks unique random positions
    spike_values = rng.uniform(spike_range[0], spike_range[1], size=n_anomalies) # overwrites the positions with the spike values
    prices[anomaly_indices] = spike_values
    is_anomaly[anomaly_indices] = 1

    return pd.DataFrame({
        "timestamp": timestamps,
        "price": np.round(prices, 2),
        "is_anomaly": is_anomaly,
    })

def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
 
    df = generate_prices()
    out_path = out_dir / "raw_prices.csv"
    df.to_csv(out_path, index=False)
 
    n_anomalies = df["is_anomaly"].sum()
    print(f"Generated {len(df)} hourly prices ({df['timestamp'].min()} → {df['timestamp'].max()})")
    print(f"Injected {n_anomalies} anomalies ({100 * n_anomalies / len(df):.1f}%)")
    print(f"Saved to {out_path}")
 
if __name__ == "__main__":
    main()

