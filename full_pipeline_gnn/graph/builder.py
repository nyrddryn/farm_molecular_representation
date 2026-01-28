"""
FG Graph Builder

This module builds functional group graphs from SMILES strings
using GNN-based structure prediction and KGE-based node features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from rdkit import Chem
from helpers import get_new_smiles_rep

from full_pipeline_gnn.models.gnn import LinkPredictorGNN
from full_pipeline_gnn.embeddings.kge import KGEmbeddings

# Aliases
s2m = Chem.MolFromSmiles
m2s = Chem.MolToSmiles


class FGGraphBuilder:
    """
    Build FG Graph using trained GCN + KGE

    Components:
    - GCN: Predicts graph structure (edges between functional groups)
    - KGE: Provides semantic node features from knowledge graph
    """

    def __init__(self,
                 gnn_path='gnn_real_weight/link_prediction_model_epoch49.pth',
                 kg_path='data/fgkg.pkl',
                 kge_path='kge_model/checkpoints86.pth'):
        """
        Initialize the FG graph builder

        Args:
            gnn_path (str): Path to trained GNN checkpoint
            kg_path (str): Path to knowledge graph vocabulary
            kge_path (str): Path to KGE embeddings
        """
        print(f"Loading GNN model from {gnn_path}...")

        # Load GNN model
        self.gnn = LinkPredictorGNN(in_channels=128, hidden_channels=128)
        self.gnn.load_state_dict(torch.load(gnn_path, map_location='cpu'))
        self.gnn.eval()

        # Load KGE for semantic features
        self.kge = KGEmbeddings(kg_path=kg_path, kge_path=kge_path)

        print(f"✓ GNN model loaded")

    def fg_token_to_feature(self, fg_token):
        """
        Convert FG token to feature vector

        Uses KGE embeddings (256-dim) projected to 128-dim for GNN input

        Args:
            fg_token (str): Functional group token

        Returns:
            torch.Tensor: 128-dim feature vector
        """
        # Get KGE embedding (256-dim)
        kge_emb = self.kge.get_fg_embedding(fg_token)

        # Project to 128-dim for GNN
        # Simple approach: average pool pairs
        feat = (kge_emb[:128] + kge_emb[128:]) / 2.0

        return feat

    def smiles_to_tokens(self, smiles):
        """
        Convert SMILES to FG-enhanced tokens

        This is the ONLY RDKit usage: SMILES → FG-enhanced SMILES conversion

        Args:
            smiles (str): Input SMILES string

        Returns:
            tuple: (tokens, canonical_smiles)
        """
        mol = s2m(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        canonical_smiles = m2s(mol)
        farm_smiles = get_new_smiles_rep(mol)
        tokens = farm_smiles.split()

        # Return tokens only - no Mol object beyond this point
        return tokens, canonical_smiles

    def build_fg_graph(self, smiles, edge_threshold=0.3):
        """
        Build FG Graph from SMILES using GNN

        Pipeline:
        1. Convert SMILES to FG tokens
        2. Extract FG nodes (atoms with underscore)
        3. Create initial fully-connected graph
        4. Use GNN to predict which edges exist
        5. Return FG graph structure

        Args:
            smiles (str): Input SMILES string
            edge_threshold (float): Threshold for edge prediction (0-1)

        Returns:
            dict: Graph structure with keys:
                - nodes: List of FG tokens
                - node_indices: Token positions
                - edges: List of (i, j) node pairs
                - node_embeddings: (N, 128) tensor
                - tokens: All tokens
                - canonical: Canonical SMILES
        """
        # Step 1: Get FG tokens (ONLY RDKit usage here)
        tokens, canonical = self.smiles_to_tokens(smiles)

        # Step 2: Extract FG nodes
        fg_nodes = []
        fg_node_indices = []  # Map to token position

        for i, token in enumerate(tokens):
            if '_' in token:  # FG atom
                fg_nodes.append(token)
                fg_node_indices.append(i)

        if len(fg_nodes) == 0:
            return {
                'nodes': [],
                'edges': [],
                'tokens': tokens,
                'canonical': canonical,
                'node_embeddings': torch.zeros((0, 128))
            }

        # Step 3: Convert to node features
        node_features = []
        for fg in fg_nodes:
            feat = self.fg_token_to_feature(fg)
            node_features.append(feat)

        node_features = torch.stack(node_features)  # (N, 128)

        # Step 4: Create initial fully-connected edge_index for GNN
        n_nodes = len(fg_nodes)
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append([i, j])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index).t()  # (2, E)

        # Step 5: Create PyG Data object
        data = Data(x=node_features, edge_index=edge_index)

        # Step 6: Run GNN to predict edges
        with torch.no_grad():
            edge_logits, edge_pairs, node_embeddings = self.gnn(data)

        # Step 7: Filter edges by threshold (lower threshold to get more edges)
        predicted_edges = []
        if len(edge_logits) > 0:
            edge_probs = torch.sigmoid(edge_logits)

            for i, prob in enumerate(edge_probs):
                if prob > edge_threshold:
                    u, v = edge_pairs[i].tolist()
                    predicted_edges.append((u, v))

        return {
            'nodes': fg_nodes,  # List of FG tokens
            'node_indices': fg_node_indices,  # Token positions
            'edges': predicted_edges,  # List of (i, j) node pairs
            'node_embeddings': node_embeddings,  # (N, 128) tensor
            'tokens': tokens,
            'canonical': canonical
        }
