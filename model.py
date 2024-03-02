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