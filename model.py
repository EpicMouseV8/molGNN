import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool
from torch.nn import Linear, Dropout, BatchNorm1d

class ChromophoreSolventGNN1(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(ChromophoreSolventGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.bn1 = BatchNorm1d(128)  # Batch normalization after first conv layer
        self.conv2 = GCNConv(128, 256)
        self.bn2 = BatchNorm1d(256)  # Batch normalization after second conv layer
        self.fc_solvent = Linear(solvent_feature_dim, 128)  # Solvent features are 32, matching this
        self.fc1 = Linear(256 + 128, 128)  # Combining features
        self.bn_fc1 = BatchNorm1d(128)  # Batch normalization after combining features
        self.dropout = Dropout(dropout_rate)  # Dropout layer
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.solvent_fingerprint

        # Process the graph with dropout and batch normalization
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        solvent_fingerprint = data.solvent_fingerprint.view(-1, solvent_feature_dim)  # Assuming batch size is 32

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, data.batch)  # Ensure 'batch' is passed correctly

        # Directly process the batched solvent fingerprints without changing its shape
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # Ensure x and solvent_features are correctly aligned in batch dimension
        x = torch.cat([x, solvent_features], dim=1)

        # Apply batch normalization and dropout before the final fully connected layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x


class ChromophoreSolventGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(ChromophoreSolventGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.bn1 = BatchNorm1d(128)  # Batch normalization after first conv layer
        self.conv2 = GCNConv(128, 256)
        self.bn2 = BatchNorm1d(256)  # Batch normalization after second conv layer
        self.fc_solvent = Linear(solvent_feature_dim, 128)  # Solvent features are 32, matching this
        self.fc1 = Linear(256 + 128, 128)  # Combining features
        self.bn_fc1 = BatchNorm1d(128)  # Batch normalization after combining features
        self.dropout = Dropout(dropout_rate)  # Dropout layer
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.solvent_fingerprint

        # Process the graph with dropout and batch normalization
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        solvent_fingerprint = data.solvent_fingerprint.view(-1, solvent_feature_dim)  # Assuming batch size is 32

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, data.batch)  # Ensure 'batch' is passed correctly

        # Directly process the batched solvent fingerprints without changing its shape
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # Ensure x and solvent_features are correctly aligned in batch dimension
        x = torch.cat([x, solvent_features], dim=1)

        # Apply batch normalization and dropout before the final fully connected layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x


class ChromophoreSolventTransformerGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(ChromophoreSolventTransformerGNN, self).__init__()
        self.transformer_conv1 = TransformerConv(node_feature_dim, 128, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn1 = BatchNorm1d(128 * 4)  # Account for the number of heads in the output
        self.transformer_conv2 = TransformerConv(128 * 4, 256, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn2 = BatchNorm1d(256 * 4)  # Account for the number of heads in the output
        self.fc_solvent = Linear(solvent_feature_dim, 128)
        self.fc1 = Linear(256 * 4 + 128, 128)  # Combine features from the second transformer layer and solvent
        self.bn_fc1 = BatchNorm1d(128)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        # TransformerConv processes
        x = F.relu(self.bn1(self.transformer_conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.transformer_conv2(x, edge_index, edge_attr)))

        # Global pooling
        x = global_mean_pool(x, batch)  # Ensure 'batch' is passed correctly

        # Solvent features
        solvent_fingerprint = data.solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # Combine features
        x = torch.cat([x, solvent_features], dim=1)

        # Apply batch normalization and dropout before final layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)

        return x
    


import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class ChromophoreSolventTransformerGNN_GAP_GMP(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(ChromophoreSolventTransformerGNN_GAP_GMP, self).__init__()  # Corrected the class name in super()
        self.transformer_conv1 = TransformerConv(node_feature_dim, 128, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn1 = BatchNorm1d(128 * 4)  # Account for the number of heads in the output
        self.transformer_conv2 = TransformerConv(128 * 4, 256, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn2 = BatchNorm1d(256 * 4)  # Account for the number of heads in the output
        self.fc_solvent = Linear(solvent_feature_dim, 128)
        self.fc1 = Linear((256 * 4) * 2 + 128, 128)  # Adjusted for combined features from GAP and GMP
        self.bn_fc1 = BatchNorm1d(128)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        # TransformerConv processes
        x = F.relu(self.bn1(self.transformer_conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.transformer_conv2(x, edge_index, edge_attr)))

        # Global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)  # Global Average Pooling
        x_gmp = global_max_pool(x, batch)  # Global Max Pooling

        # Combine pooled features
        x_combined = torch.cat([x_gap, x_gmp], dim=1)  # Concatenate GAP and GMP features

        # Solvent features
        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # Combine features from GNN and solvent
        x = torch.cat([x_combined, solvent_features], dim=1)

        # Apply batch normalization and dropout before final layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)

        return x

    
class DoubleGraphGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim, dropout_rate=0.3):
        super(DoubleGraphGNN, self).__init__()

        self.conv1_chromophore = GCNConv(node_feature_dim, 128)
        self.bn1_chromophore = BatchNorm1d(128)
        self.conv2_chromophore = GCNConv(128, 64)
        self.bn2_chromophore = BatchNorm1d(64)

        self.conv1_solvent = GCNConv(node_feature_dim, 128)
        self.bn1_solvent = BatchNorm1d(128)
        self.conv2_solvent = GCNConv(128, 64)
        self.bn2_solvent = BatchNorm1d(64)

        self.fc1 = Linear(64 * 2, 64)
        self.bn_fc1 = BatchNorm1d(64)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(64, output_dim)

    def forward(self, data):
        x_chromophore, edge_index_chromophore, edge_attr_chromophore, x_solvent, edge_index_solvent, edge_attr_solvent, batch = data.x_chromophore, data.edge_index_chromophore, data.edge_attr_chromophore, data.x_solvent, data.edge_index_solvent, data.edge_attr_solvent, data.batch
        
        # Process the graph with dropout and batch normalization
        x_chromophore = F.relu(self.bn1_chromophore(self.conv1_chromophore(x_chromophore, edge_index_chromophore)))
        x_chromophore = F.relu(self.bn2_chromophore(self.conv2_chromophore(x_chromophore, edge_index_chromophore)))

        x_solvent = F.relu(self.bn1_solvent(self.conv1_solvent(x_solvent, edge_index_solvent)))
        x_solvent = F.relu(self.bn2_solvent(self.conv2_solvent(x_solvent, edge_index_solvent)))

        # Global pooling to get graph-level representation
        x_chromophore = global_mean_pool(x_chromophore, batch)
        x_solvent = global_mean_pool(x_solvent, batch)

        # Combine features
        x = torch.cat([x_chromophore, x_solvent], dim=1)

        # Apply batch normalization and dropout before the final fully connected layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)

        return x


