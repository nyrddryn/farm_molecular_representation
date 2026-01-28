#!/usr/bin/env python3
"""
FARM + GCN Molecular Comparison Pipeline

Pipeline:
1. Input SMILES → FG-enhanced representation (rule-based)
2. Build FG Graph using GCN (learned structure prediction)
3. Compare 2 FG graphs → Find MCS + differences
4. Output differences as atom-level groups
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import argparse
import time
import pickle
from collections import defaultdict
import numpy as np

from rdkit import Chem
from helpers import get_new_smiles_rep

# Aliases
s2m = Chem.MolFromSmiles
m2s = Chem.MolToSmiles


class KGEmbeddings:
    """
    Load and use ComplEx Knowledge Graph Embeddings
    Provides semantic similarity between functional groups
    """
    
    def __init__(self, kg_path='data/fgkg.pkl', kge_path='ccheckpoints/checkpoints86.pth'):
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
        
        print(f"✓ KGE loaded: {self.entity_embeddings_real.shape[0]} entities, {self.entity_embeddings_real.shape[1]}*2 dims")
    
    def get_fg_embedding(self, fg_token):
        """
        Get embedding for a functional group token
        Returns concatenated [real, imag] vector (256-dim)
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
        Returns cosine similarity score
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


class LinkPredictorGNN(nn.Module):
    """
    GCN-based Link Prediction model
    Trained to predict edges in molecular FG graphs
    """
    def __init__(self, in_channels=128, hidden_channels=128):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        """
        Input: PyG Data with x (node features) and edge_index
        Output: edge_logits, edge_pairs, node_embeddings
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


class FGGraphBuilder:
    """
    Build FG Graph using trained GCN + KGE
    GCN: Predicts graph structure (edges)
    KGE: Provides semantic node features
    """
    
    def __init__(self, 
                 gnn_path='gnn_real_weight/link_prediction_model_epoch49.pth',
                 kg_path='data/fgkg.pkl',
                 kge_path='kge_model/checkpoints86.pth'):
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
        Use KGE embeddings (256-dim) projected to 128-dim
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
        ONLY RDKit usage: SMILES → FG-enhanced SMILES conversion
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


class FGGraphComparator:
    """
    Compare two FG graphs and extract MCS + differences
    Uses both GCN structure and KGE similarity
    """
    
    def __init__(self, builder):
        self.builder = builder
        self.kge = builder.kge
    
    def match_graphs(self, graph1, graph2, similarity_threshold=0.65):
        """
        Match two FG graphs using HYBRID scoring:
        - KGE semantic similarity (primary)
        - GCN structural embeddings (secondary)
        
        Returns:
        - matched_pairs: [(i, j, score), ...]
        - unmatched1: [i, ...]
        - unmatched2: [j, ...]
        """
        nodes1 = graph1['nodes']
        nodes2 = graph2['nodes']
        emb1 = graph1['node_embeddings']  # GCN embeddings
        emb2 = graph2['node_embeddings']
        
        if len(nodes1) == 0 or len(nodes2) == 0:
            return [], list(range(len(nodes1))), list(range(len(nodes2)))
        
        # Compute GCN structural similarity
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        gcn_similarity = torch.matmul(emb1_norm, emb2_norm.T)  # (N1, N2)
        
        # Greedy matching with hybrid scoring
        matched_pairs = []
        matched1 = set()
        matched2 = set()
        
        # Score all potential matches
        scores = []
        for i in range(len(nodes1)):
            for j in range(len(nodes2)):
                fg1 = nodes1[i]
                fg2 = nodes2[j]
                
                # Hybrid scoring:
                # 1. KGE semantic similarity (weight: 0.7)
                kge_sim = self.kge.compute_similarity(fg1, fg2)
                
                # 2. GCN structural similarity (weight: 0.3)
                gcn_sim = gcn_similarity[i, j].item()
                
                # Combined score
                combined_score = 0.7 * kge_sim + 0.3 * gcn_sim
                
                # Boost for exact match
                if fg1 == fg2:
                    combined_score = max(combined_score, 0.99)
                
                if combined_score >= similarity_threshold:
                    scores.append((combined_score, i, j))
        
        # Sort by score (highest first)
        scores.sort(reverse=True)
        
        # Greedy assignment
        for score, i, j in scores:
            if i not in matched1 and j not in matched2:
                matched_pairs.append((i, j, score))
                matched1.add(i)
                matched2.add(j)
        
        # Unmatched nodes
        unmatched1 = [i for i in range(len(nodes1)) if i not in matched1]
        unmatched2 = [j for j in range(len(nodes2)) if j not in matched2]
        
        return matched_pairs, unmatched1, unmatched2
    
    def extract_mcs_atoms(self, graph, matched_node_indices):
        """
        Extract atom-level MCS from matched FG nodes
        
        Returns: List of atom indices in original molecule
        """
        if not matched_node_indices:
            return []
        
        mol = graph['mol']
        tokens = graph['tokens']
        node_indices = graph['node_indices']
        
        # Get token positions of matched nodes
        matched_token_positions = [node_indices[i] for i in matched_node_indices]
        
        # Map token positions to atom indices in mol
        mcs_atoms = []
        
        # Simple mapping: FG token at position i corresponds to atom
        # This is approximate - need proper mapping from helpers.py
        for token_pos in matched_token_positions:
            # Find corresponding atom in mol
            # For simplicity, use token position as approximation
            if token_pos < mol.GetNumAtoms():
                mcs_atoms.append(token_pos)
        
        return mcs_atoms
    
    def group_unmatched_nodes(self, graph, unmatched_node_indices):
        """
        Group unmatched nodes based on:
        1. Graph connectivity (GNN edges)
        2. Token position proximity (for disconnected groups)
        """
        if not unmatched_node_indices:
            return []
        
        edges = graph['edges']
        node_indices = graph['node_indices']  # Token positions
        
        # Build adjacency for unmatched nodes
        adj = defaultdict(set)
        for i, j in edges:
            if i in unmatched_node_indices and j in unmatched_node_indices:
                adj[i].add(j)
                adj[j].add(i)
        
        # Find connected components using GNN edges
        visited = set()
        components = []
        
        for node in unmatched_node_indices:
            if node not in visited:
                # DFS to find connected component
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adj[current] - visited)
                
                components.append(sorted(component))
        
        # Further split components by token position if they are far apart
        # This handles cases where GNN predicts edges but groups are structurally separate
        final_components = []
        for comp in components:
            if len(comp) <= 1:
                final_components.append(comp)
                continue
            
            # Get token positions
            positions = [node_indices[i] for i in comp]
            
            # Sort by position
            comp_with_pos = sorted(zip(comp, positions), key=lambda x: x[1])
            
            # Split if gap > 5 tokens (different structural regions)
            sub_groups = []
            current_group = [comp_with_pos[0][0]]
            
            for i in range(1, len(comp_with_pos)):
                node, pos = comp_with_pos[i]
                prev_pos = comp_with_pos[i-1][1]
                
                if pos - prev_pos > 5:  # Large gap - different region
                    sub_groups.append(sorted(current_group))
                    current_group = [node]
                else:
                    current_group.append(node)
            
            if current_group:
                sub_groups.append(sorted(current_group))
            
            final_components.extend(sub_groups)
        
        # Sort components by their minimum token position
        final_components.sort(key=lambda comp: min(node_indices[i] for i in comp))
        
        return final_components
    
    def extract_atom_groups(self, graph, node_indices):
        """
        Extract atom-level representation from FG node indices
        Returns list of base atoms (without FG labels)
        """
        atoms = []
        for node_idx in node_indices:
            fg_token = graph['nodes'][node_idx]
            # Get base atom (before first underscore)
            if '_' in fg_token:
                base_atom = fg_token.split('_')[0]
                atoms.append(base_atom)
            else:
                atoms.append(fg_token)
        return atoms
    
    def format_diff_group_atoms(self, graph, node_indices):
        """
        Format difference group at atom-level (not FG-level)
        Returns base atoms only
        """
        atoms = self.extract_atom_groups(graph, node_indices)
        
        if len(atoms) == 0:
            return ""
        elif len(atoms) == 1:
            return atoms[0]
        else:
            # Group consecutive atoms
            return f"({''.join(atoms)})"
    
    def format_diff_group_with_fg(self, graph, node_indices):
        """
        Format difference group with FG labels
        Returns full FG names
        """
        fg_tokens = [graph['nodes'][i] for i in node_indices]
        
        if len(fg_tokens) == 0:
            return ""
        elif len(fg_tokens) == 1:
            return fg_tokens[0]
        else:
            # Join with underscore
            return f"({'_'.join(fg_tokens)})"
    
    def compare_smiles(self, smiles1, smiles2, verbose=False, use_fg_labels=False):
        """
        Complete comparison pipeline
        
        Args:
            smiles1, smiles2: SMILES strings
            verbose: Print detailed info
            use_fg_labels: If True, use FG labels; if False, use atom-level
        """
        start_time = time.time()
        
        # Step 1: Build FG graphs
        print("Building FG graphs...")
        graph1 = self.builder.build_fg_graph(smiles1)
        graph2 = self.builder.build_fg_graph(smiles2)
        
        # Step 2: Match graphs
        print("Matching graphs...")
        matched, unmatched1, unmatched2 = self.match_graphs(graph1, graph2)
        
        # Step 3: Extract MCS
        matched_nodes1 = [pair[0] for pair in matched]
        matched_nodes2 = [pair[1] for pair in matched]
        
        # Step 4: Group differences
        diff_groups1 = self.group_unmatched_nodes(graph1, unmatched1)
        diff_groups2 = self.group_unmatched_nodes(graph2, unmatched2)
        
        # Step 5: Format output - atom-level representation
        # Build MCS from matched nodes (atom-level, no FG labels)
        if matched_nodes1:
            # Extract base atoms from matched nodes
            mcs_atoms = []
            for node_idx in matched_nodes1:
                fg_token = graph1['nodes'][node_idx]
                base_atom = fg_token.split('_')[0] if '_' in fg_token else fg_token
                mcs_atoms.append(base_atom)
            
            # Simple concatenation - no RDKit validation
            mcs_string = ''.join(mcs_atoms)
        else:
            mcs_string = ""
        
        # Format diff groups (choose format based on flag)
        serial_parts1 = []
        for i, group in enumerate(diff_groups1, 1):
            if use_fg_labels:
                group_label = self.format_diff_group_with_fg(graph1, group)
            else:
                group_label = self.format_diff_group_atoms(graph1, group)
            
            if group_label:
                # Get position for ordering
                min_pos = min(graph1['node_indices'][node_idx] for node_idx in group)
                serial_parts1.append((min_pos, group_label, i))
        
        serial_parts2 = []
        for i, group in enumerate(diff_groups2, 1):
            if use_fg_labels:
                group_label = self.format_diff_group_with_fg(graph2, group)
            else:
                group_label = self.format_diff_group_atoms(graph2, group)
            
            if group_label:
                min_pos = min(graph2['node_indices'][node_idx] for node_idx in group)
                serial_parts2.append((min_pos, group_label, i))
        
        # Sort by position
        serial_parts1.sort(key=lambda x: x[0])
        serial_parts2.sort(key=lambda x: x[0])
        
        # Build output with proper positioning
        # Determine MCS position based on first matched node
        if matched_nodes1:
            mcs_pos1 = min(graph1['node_indices'][i] for i in matched_nodes1)
        else:
            mcs_pos1 = 0
        
        if matched_nodes2:
            mcs_pos2 = min(graph2['node_indices'][i] for i in matched_nodes2)
        else:
            mcs_pos2 = 0
        
        # Build mol1 output - insert diff groups and MCS at proper positions
        all_parts1 = []
        for pos, fg, serial in serial_parts1:
            all_parts1.append((pos, f"{fg}:{serial}"))
        all_parts1.append((mcs_pos1, mcs_string))
        all_parts1.sort(key=lambda x: x[0])
        
        output1 = " ".join([part[1] for part in all_parts1])
        
        # Build mol2 output
        all_parts2 = []
        for pos, fg, serial in serial_parts2:
            all_parts2.append((pos, f"{fg}:{serial}"))
        all_parts2.append((mcs_pos2, mcs_string))
        all_parts2.sort(key=lambda x: x[0])
        
        output2 = " ".join([part[1] for part in all_parts2])
        
        result = f"{output1} and {output2}"
        
        # Metrics
        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'matched_nodes': len(matched),
            'diff_groups_mol1': len(diff_groups1),
            'diff_groups_mol2': len(diff_groups2),
            'canonical1': graph1['canonical'],
            'canonical2': graph2['canonical']
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"GNN-BASED FG GRAPH COMPARISON")
            print(f"{'='*80}")
            print(f"Molecule 1: {smiles1}")
            print(f"  Canonical: {graph1['canonical']}")
            print(f"  FG nodes: {len(graph1['nodes'])}")
            print(f"  GNN predicted edges: {len(graph1['edges'])}")
            
            print(f"\nMolecule 2: {smiles2}")
            print(f"  Canonical: {graph2['canonical']}")
            print(f"  FG nodes: {len(graph2['nodes'])}")
            print(f"  GNN predicted edges: {len(graph2['edges'])}")
            
            print(f"\nGraph Matching:")
            print(f"  Matched nodes: {len(matched)}")
            print(f"  Unmatched mol1: {len(unmatched1)}")
            print(f"  Unmatched mol2: {len(unmatched2)}")
            
            print(f"\nMCS: {mcs_string}")
            print(f"\nTime: {total_time*1000:.2f} ms")
        
        return result, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compare SMILES using GNN-based FG graph construction'
    )
    parser.add_argument('--smiles1', required=True, help='First SMILES string')
    parser.add_argument('--smiles2', required=True, help='Second SMILES string')
    parser.add_argument('--gnn', default='gnn_real_weight/link_prediction_model_epoch49.pth',
                       help='Path to GNN checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Edge prediction threshold (default: 0.5)')
    parser.add_argument('--fg-labels', action='store_true',
                       help='Use FG labels instead of atom-level output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = FGGraphBuilder(gnn_path=args.gnn)
    
    # Initialize comparator
    comparator = FGGraphComparator(builder)
    
    # Compare
    result, metrics = comparator.compare_smiles(
        args.smiles1, args.smiles2,
        verbose=args.verbose,
        use_fg_labels=args.fg_labels
    )
    
    print(f"\nInput 1: {args.smiles1}")
    print(f"Input 2: {args.smiles2}")
    print(f"\nOutput: {result}")
    
    if not args.verbose and metrics:
        print(f"\nMetrics:")
        print(f"  Matched nodes: {metrics['matched_nodes']}")
        print(f"  Diff groups: Mol1={metrics['diff_groups_mol1']}, Mol2={metrics['diff_groups_mol2']}")
        print(f"  Time: {metrics['total_time']*1000:.2f} ms")


if __name__ == '__main__':
    main()
