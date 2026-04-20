'''
FastAPI REST endpoint for the anomaly detection

exposes a /predict endpoint, that accepts 24 hour price sequence and returns the resconstrution error and anomaly flag

instruction:
    run with:
    uvicorn api:app --reload
'''

 
import numpy as np
import torch
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
 
from src.model import LSTMAutoencoder
 
# load model, threshold and scaler at startup
model_dir = Path('models')
data_dir = Path('data/processed')
 
model = LSTMAutoencoder(input_dim=1, hidden_dim=32, n_layers=1)
model.load_state_dict(torch.load(model_dir / 'autoencoder.pt', weights_only=True))
model.eval()
 
threshold = np.load(model_dir / 'threshold.npy').item()
 
with open(data_dir / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
 
app = FastAPI(
    title='Energy Anomaly Detector',
    description='Detects anomalous electricity price patterns using an LSTM autoencoder.',
)
 
 
class PriceWindow(BaseModel):
    '''requesting: a sequence of 24 hourly prices in EUR/MWh.'''
    prices: list[float]
 
 
class PredictionResponse(BaseModel):
    '''response: anomaly score and flag.'''
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
 
 
@app.post('/predict', response_model=PredictionResponse)
def predict(window: PriceWindow) -> PredictionResponse:
    '''
    accepts 24 hour price window and returns anomaly detection results

    prices are scaled, using same scalar from training, passing through LSTM autoencoder. reconstruction error is compared against the learned threshold
    '''
    
    if len(window.prices) != 24:
        raise HTTPException(
            status_code=400,
            detail=f'Expected 24 prices, got {len(window.prices)}',
        )
 
    # scale input prices
    prices_array = np.array(window.prices).reshape(-1, 1)
    scaled = scaler.transform(prices_array).flatten()
 
    # reshape to (1, 24, 1), single sample, 24 timesteps, 1 feature
    X = torch.FloatTensor(scaled).reshape(1, 24, 1)
 
    # predict
    with torch.no_grad():
        reconstructed = model(X)
        error = torch.mean((X - reconstructed) ** 2).item()
 
    return PredictionResponse(
        reconstruction_error=error,
        threshold=threshold,
        is_anomaly=error > threshold,
    )
 
 
@app.get('/health')
def health():
    '''Health check endpoint.'''
    return {'status': 'healthy', 'model_loaded': True}