"""
GNN-based Link Prediction Model

This module contains the Graph Neural Network architecture
for predicting edges in functional group molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictorGNN(nn.Module):
    """
    GCN-based Link Prediction model
    Trained to predict edges in molecular FG graphs

    Architecture:
    - GCN layer for node embedding
    - MLP for edge prediction from concatenated node pairs
    """

    def __init__(self, in_channels=128, hidden_channels=128):
        """
        Initialize the link predictor model

        Args:
            in_channels (int): Input feature dimension
            hidden_channels (int): Hidden layer dimension
        """
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        """
        Forward pass for link prediction

        Args:
            data: PyG Data with x (node features) and edge_index

        Returns:
            edge_logits: Predicted logits for each edge
            edge_pairs: List of node pairs (i, j)
            node_embeddings: Node embeddings from GCN (N, hidden_channels)
        """
        # GCN forward
        x = self.gcn(data.x, data.edge_index)
        x = F.relu(x)

        # Generate all possible edge pairs
        n_nodes = x.size(0)
        edge_pairs = []
        edge_features = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_pairs.append([i, j])
                # Concatenate node embeddings
                edge_feat = torch.cat([x[i], x[j]])
                edge_features.append(edge_feat)

        if len(edge_features) == 0:
            return torch.tensor([]), torch.tensor([]), x

        edge_features = torch.stack(edge_features)
        edge_pairs = torch.tensor(edge_pairs)

        # Predict edge existence
        edge_logits = self.edge_mlp(edge_features).squeeze(-1)

        return edge_logits, edge_pairs, x
