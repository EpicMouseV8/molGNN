import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split, KFold
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import ChromophoreSolventGNN, ChromophoreSolventTransformerGNN
import os
import matplotlib.pyplot as plt


def train(loader, model, optimizer, loss_fn, num_targets, solvent_feature_dim, device):
    model.train()
    total_loss = 0
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        output = model(data, solvent_feature_dim=solvent_feature_dim)
        
        data.y = data.y.view(-1, num_targets) # Reshape targets to match output shape

        loss = loss_fn(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(loader, model, loss_fn, num_targets, solvent_feature_dim, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data.to(device)
            output = model(data, solvent_feature_dim=solvent_feature_dim)

            data.y = data.y.view(-1, num_targets)

            loss = loss_fn(output, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def run_training(data, epochs, target_names, model_name):

    num_targets = len(target_names)

    #training and validation sets
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15

    data_train, data_test_val = train_test_split(data, test_size= 1 - train_ratio, random_state=0)
    data_test, data_val = train_test_split(data_test_val, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)

    train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=32, shuffle=False)


    print("Instantiating model...")

    node_feature_dim = data[0].num_node_features
    edge_feature_dim = data[0].num_edge_features
    solvent_feature_dim = data[0].solvent_fingerprint.size(0)
    output_dim = num_targets

    model = ChromophoreSolventTransformerGNN(node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}.")
    model.to(device)

    print("Training model...")

    save_folder = 'models/' + model_name

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer, loss_fn, num_targets, solvent_feature_dim, device)
        val_loss = evaluate(val_loader, model, loss_fn, num_targets, solvent_feature_dim, device)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            file_name = model_name + 'epoch_' + str(epoch) + '.pth'
            save_path = os.path.join(save_folder, file_name)
            torch.save(model.state_dict(), save_path)
            print("Model saved as " + model_name + '_epoch_' + str(epoch) + '.pth')

    print("Training complete. Saving model...")

    file_name = model_name +'.pth'
    save_path = os.path.join(save_folder, file_name)
    torch.save(model.state_dict(), save_path)
    print("Model saved as gcn_" + model_name + ".pth.")

    print("Testing model...")

    test_model(model, test_loader, num_targets, target_names, solvent_feature_dim)


def test_model(model, test_loader, num_targets, target_names, solvent_feature_dim):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    model.eval() 
    predictions = []
    actuals = []

    bad_smiles = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data, solvent_feature_dim=solvent_feature_dim)

            prediction = output.cpu().numpy()
            actual = data.y.view(-1, 1).cpu().numpy()  # Adjust shape as necessary

            # Calculate error; adjust this based on your specific needs
            error = np.abs(prediction - actual)

            for i, err in enumerate(error):
                if np.any(err > 100):  # Using np.any() to capture any target exceeding the threshold
                    chromo_smiles = data.chromo_smiles[i]
                    solvent_smiles = data.solvent_smiles[i]
                    bad_smiles.append((chromo_smiles, solvent_smiles))

            # No need to reshape if your model output and targets are already in the correct shape
            predictions.append(output.cpu().numpy())
            actuals.append(data.y.view(-1, num_targets).cpu().numpy())  # Reshape targets to match output shape
            
    predictions = np.vstack(predictions)  # Shape: [N, 3], where N is the number of samples
    actuals = np.vstack(actuals)  # Shape: [N, 3]
    
    # Compute metrics for each output
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        mse = mean_squared_error(actuals[:, i], predictions[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals[:, i], predictions[:, i])
        
        print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")
        
        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values for {name}')
        max_val = max(actuals[:, i].max(), predictions[:, i].max())
        plt.plot([0, max_val], [0, max_val], 'k--')  # Diagonal line indicating perfect predictions
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
        plt.show()

    return bad_smiles
    





