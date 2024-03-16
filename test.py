import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split, KFold
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import ChromophoreSolventGNN, ChromophoreSolventTransformerGNN, DoubleGraphGNN, ChromophoreSolventTransformerGNN_GAP_GMP
import os
import matplotlib.pyplot as plt

def getpredictions(model, test_loader, solvent_feature_dim):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data, solvent_feature_dim = solvent_feature_dim)

            predictions.append(out.cpu().numpy())

    predictions = np.vstack(predictions)

    return predictions