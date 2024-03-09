import pandas as pd
import deepchem
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import os
from torch_geometric.data import Batch


def preprocess(df, target_columns):
    print("Preprocessing data...")
    
    # Drop rows where 'Solvent' is 'O'
    df = df[df['Solvent'] != 'O']
    
    # Drop rows with missing values in the target columns
    df = df.dropna(subset=target_columns)

    chromophores = df['Chromophore'].to_numpy()
    solvents = df['Solvent'].to_numpy()
    targets = df[target_columns].to_numpy()

    print("Data preprocessed.")

    return chromophores, solvents, targets

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, chromophores, solvents, targets, featurizer):
        assert len(chromophores) == len(solvents)
        self.chromophores = chromophores
        self.solvents = solvents
        self.targets = targets
        self.featurizer = featurizer

    def __len__(self):
        return len(self.chromophores)

    def __getitem__(self, idx):
        chromophore = self.chromophores[idx]
        solvent = self.solvents[idx]
        target = self.targets[idx]

        return self.featurizer(chromophore, solvent), target


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB