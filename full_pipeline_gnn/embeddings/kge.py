"""
Knowledge Graph Embeddings (KGE) Handler

This module loads and uses ComplEx embeddings to provide
semantic similarity between functional groups.
"""

import torch
import torch.nn.functional as F
import pickle


class KGEmbeddings:
    """
    Load and use ComplEx Knowledge Graph Embeddings
    Provides semantic similarity between functional groups

    ComplEx embeddings have real and imaginary components
    that capture complex relationships in the knowledge graph.
    """

    def __init__(self, kg_path='data/fgkg.pkl', kge_path='kge_model/checkpoints86.pth'):
        """
        Initialize KGE embeddings

        Args:
            kg_path (str): Path to knowledge graph vocabulary file
            kge_path (str): Path to ComplEx embedding checkpoint
        """
        print(f"Loading KGE embeddings from {kge_path}...")

        # Load KG vocabulary
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)
        self.node_to_idx = kg_data['node_to_idx']
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}

        # Load ComplEx embeddings
        checkpoint = torch.load(kge_path, map_location='cpu')

        # ComplEx has real and imaginary parts
        self.entity_embeddings_real = checkpoint['ent_real.weight']  # (N, 128)
        self.entity_embeddings_imag = checkpoint['ent_imag.weight']  # (N, 128)

        print(f"✓ KGE loaded: {self.entity_embeddings_real.shape[0]} entities, "
              f"{self.entity_embeddings_real.shape[1]}*2 dims")

    def get_fg_embedding(self, fg_token):
        """
        Get embedding for a functional group token

        Args:
            fg_token (str): Functional group token (e.g., 'C_alcohol')

        Returns:
            torch.Tensor: Concatenated [real, imag] vector (256-dim)
        """
        if fg_token not in self.node_to_idx:
            # Unknown FG - return zeros
            return torch.zeros(256)

        idx = self.node_to_idx[fg_token]

        # Concatenate real and imaginary parts
        real = self.entity_embeddings_real[idx]
        imag = self.entity_embeddings_imag[idx]
        emb = torch.cat([real, imag])

        # Normalize
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)

        return emb

    def compute_similarity(self, fg1, fg2):
        """
        Compute semantic similarity between two FG types using KGE

        Args:
            fg1 (str): First functional group token
            fg2 (str): Second functional group token

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Exact match
        if fg1 == fg2:
            return 1.0

        # Get embeddings
        emb1 = self.get_fg_embedding(fg1)
        emb2 = self.get_fg_embedding(fg2)

        # Cosine similarity
        sim = torch.dot(emb1, emb2).item()

        return sim
