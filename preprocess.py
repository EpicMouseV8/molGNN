import pandas as pd
import deepchem
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import os


def preprocess(df, target_columns):
    print("Preprocessing data...")
    
    # Drop rows where 'Solvent' is 'O'
    # df = df[df['Solvent'] != 'O']
    
    # Drop rows with missing values in the target columns

    df = df.dropna(subset=target_columns)

    chromophores = df['Chromophore'].to_numpy()
    solvents = df['Solvent'].to_numpy()
    targets = df[target_columns].to_numpy()

    print("Data preprocessed.")

    return chromophores, solvents, targets


def featurizeAsGraphs(chromophores, solvents, targets, save_filename):

    data = []

    save_dir = 'data/processed/' + save_filename
    if not os.path.exists(save_dir):
        print("Creating directory for processed data.")
        os.makedirs(save_dir)

    all_files_exist = True
    for i in range(len(chromophores)):
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        if not os.path.isfile(save_path):
            all_files_exist = False
            break
        else:
            graph = torch.load(save_path)
            data.append(graph)

    if all_files_exist:
        print("Loaded preprocessed graphs from disk.")
        return data


    print("Featurizing data...")

    featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    featuresChromo = featurizer.featurize(chromophores)
    featuresSolvent = featurizer.featurize(solvents)

    print("Data featurized.")

    print("Creating graph data...")

    for i, (featChromo, featSolvent) in enumerate(zip(featuresChromo, featuresSolvent)):
        node_featuresChromo = torch.tensor(featChromo.node_features, dtype=torch.float)
        edge_featuresChromo = torch.tensor(featChromo.edge_features, dtype=torch.float)
        edge_indexChromo = torch.tensor(featChromo.edge_index, dtype=torch.long)

        node_featuresSolvent = torch.tensor(featSolvent.node_features, dtype=torch.float)
        edge_featuresSolvent = torch.tensor(featSolvent.edge_features, dtype=torch.float)
        edge_indexSolvent = torch.tensor(featSolvent.edge_index, dtype=torch.long)

        chromo_smiles = chromophores[i]
        solvent_smiles = solvents[i]

        graph1 = Data(x=node_featuresChromo, edge_index=edge_indexChromo.t().contiguous(), edge_attr=edge_featuresChromo,
                     chromo_smiles=chromo_smiles,   y=torch.tensor(targets[i], dtype=torch.float))
        
        graph2 = Data(x_solvent=node_featuresSolvent, edge_index_solvent=edge_indexSolvent.t().contiguous(), edge_attr_solvent=edge_featuresSolvent, 
                      solvent_smiles = solvent_smiles)

        data_list = [graph1, graph2]
        data.append(data_list)

        # Save each graph object to a file
        # save_path = os.path.join(save_dir, f'graph_{i}.pt')
        # torch.save(graph, save_path)

    return data


def featurize(chromophores, solvents, targets, save_filename):
    
    data = []

    save_dir = 'data/processed/' + save_filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files_exist = True
    for i in range(len(chromophores)):
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        if not os.path.isfile(save_path):
            all_files_exist = False
            break
        else:
            graph = torch.load(save_path)
            data.append(graph)

    if all_files_exist:
        print("Loaded preprocessed graphs from disk.")

        for graph in data:
            graph.edge_index = graph.edge_index.t()

        return data


    print("Featurizing data...")
    featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    features = featurizer.featurize(chromophores)

    print("Data featurized.")

    print("Creating graph data...")

    for i, feat in enumerate(features):
        node_features = torch.tensor(feat.node_features, dtype=torch.float)
        edge_features = torch.tensor(feat.edge_features, dtype=torch.float)
        edge_index = torch.tensor(feat.edge_index, dtype=torch.long)

        chromo_smiles = chromophores[i]
        solvent_smiles = solvents[i]
        solvent_mol = Chem.MolFromSmiles(solvent_smiles)
        solvent_fingerprint = AllChem.GetMorganFingerprintAsBitVect(solvent_mol, radius=2, nBits=128)
        solvent_fingerprint = torch.tensor((solvent_fingerprint), dtype=torch.float)

        y = torch.tensor([targets[i]], dtype=torch.float)

        graph = Data(x=node_features, solvent_fingerprint = solvent_fingerprint, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, 
                     chromo_smiles=chromo_smiles, solvent_smiles = solvent_smiles,  y=torch.tensor(targets[i], dtype=torch.float))
        data.append(graph)

        # Save each graph object to a file
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        torch.save(graph, save_path)

    for graph in data:
        graph.edge_index = graph.edge_index.t()

    return data


