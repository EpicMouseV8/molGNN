import torch
import model, preprocess, train
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader

targets = ["Absorption max (nm)"]
file_path = 'data/raw/prep2.csv'

df = pd.read_csv(file_path)

chromophores, solvents, targetsVal = preprocess.preprocess(df, targets)

data = preprocess.featurize(chromophores, solvents, targetsVal, model_name="abs_maxfp128_smiles")

# train.run_training(data, epochs=1200, target_names=targets, model_name="abs_ems_maxfp128")


# test
node_feature_dim = data[0].num_node_features
edge_feature_dim = data[0].num_edge_features
solvent_feature_dim = data[0].solvent_fingerprint.size(0)
output_dim = 1
model = model.ChromophoreSolventTransformerGNN(node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim)

model_path = 'models/transformer_abs_maxfp128/transformer_abs_maxfp128epoch_1200.pth'
model.load_state_dict(torch.load(model_path))


# training and validation sets
# train_ratio = 0.70
# validation_ratio = 0.15
# test_ratio = 0.15

# data_train, data_test_val = train_test_split(data, test_size= 1 - train_ratio, random_state=0)
# data_test, data_val = train_test_split(data_test_val, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)

# train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
# val_loader = DataLoader(data_val, batch_size=32, shuffle=False)
# test_loader = DataLoader(data_test, batch_size=32, shuffle=False)


# smiles_list = []
# bad_smiles = train.test_model(model, test_loader=test_loader, num_targets=1, target_names=targets, solvent_feature_dim=solvent_feature_dim)

# print(bad_smiles)

# import csv


# # Specify the path to your output file
# csv_file_path = 'bad_smiles.csv'

# # Open the file in write mode ('w') and create a CSV writer object
# with open(csv_file_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     # Write a header row (optional)
#     writer.writerow(['Chromophore SMILES', 'Solvent SMILES'])
#     # Write the SMILES data
#     writer.writerows(bad_smiles)

# print(f"Saved bad SMILES to {csv_file_path}")
