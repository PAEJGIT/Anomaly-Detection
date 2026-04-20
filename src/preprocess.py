"""
preparing raw price data for the LSTM autoencoder
outputs:
X_train.npy: training window
X_test.npy: test window
Y_test.npy
scaler.pkl
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(path: str = "data/raw/raw_prices.csv") -> pd.DataFrame:
    """load the raw price csv"""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    """split into train/trest chronologically, return two dataframes"""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def create_windows(values: np.ndarray, labels:np.ndarray, window_size: int = 24):
    """
    create sliding windows from a 1d array of values

    args:
        values> scaled price values, shape (n,)
        labels: anomaly labels, shape (n,)
        window_size: number of hours per window

    returns:
        x: array of shape (n_windows, window_size, 1)
        y: array of shape (n_windows, ), 1 if 
    """
    X = []
    y = []
    
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        label = labels[i : i + window_size]
        X.append(window)
        # a window is anomalous if it contains at least one anomalous hour
        y.append(1 if label.sum() > 0 else 0)

    X = np.array(X)
    y = np.array(y)

    # reshaping X to (n_windows, window) for LSTM input
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def main() -> None:
    # load
    df = load_raw_data()
    print(f'loaded {len(df)} rows')

    # split
    train_df, test_df = split_train_test(df)
    print(f'train: {len(train_df)} rows, test: {len(test_df)} rows')

    # scale, fit on train only
    scaler = MinMaxScaler()
    train_prices = scaler.fit_transform(train_df[['price']].values)
    test_prices = scaler.transform(test_df[['price']].values)

    # flatten back to 1d for windowing
    train_prices = train_prices.flatten()
    test_prices = test_prices.flatten()

    # create windows
    window_size = 24
    X_train, y_train = create_windows(
        train_prices,
        train_df['is_anomaly'].to_numpy(),
        window_size,
    )
    X_test, y_test = create_windows(
        test_prices,
        test_df['is_anomaly'].values,
        window_size,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Anomalous windows in test: {y_test.sum()} / {len(y_test)}")


    # save

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
 
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / 'y_train.npy', y_train)
    np.save(out_dir / "y_test.npy", y_test)
    
 
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
 
    print(f"Saved processed data to {out_dir}")
 
 
if __name__ == "__main__":
    main()