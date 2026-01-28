#!/usr/bin/env python3
"""
FARM-based Molecular Comparison using Functional Group Graph
Pipeline:
1. Build FG graph from FG-enhanced SMILES using learned embeddings
2. Find maximum common subgraph using similarity from KGE
3. Identify corresponding but different FG positions
4. Group atoms and format output
"""

import sys
sys.path.append('src')

from rdkit import Chem
from helpers import get_new_smiles_rep
import argparse
import time
from collections import defaultdict
import torch
import torch.nn.functional as F
import pickle
import numpy as np


# Aliases
s2m = Chem.MolFromSmiles
m2s = Chem.MolToSmiles


class KGEmbeddings:
    """Load and manage KG embeddings for FG similarity"""
    
    def __init__(self, kg_path='data/fgkg.pkl', ckpt_path='ccheckpoints/checkpoints86.pth'):
        print(f"Loading KG embeddings from {ckpt_path}...")
        
        # Load KG data (node_to_idx mapping)
        with open(kg_path, 'rb') as f:
            kg_data = pickle.load(f)
        
        self.node_to_idx = kg_data['node_to_idx']
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.relation_to_idx = kg_data['relation_to_idx']
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Get entity embeddings (complex-valued)
        self.ent_real = checkpoint['ent_real.weight']  # (num_entities, 128)
        self.ent_imag = checkpoint['ent_imag.weight']  # (num_entities, 128)
        
        print(f"✓ Loaded {self.ent_real.shape[0]} FG embeddings (dim={self.ent_real.shape[1]})")
    
    def get_fg_embedding(self, fg_type):
        """Get embedding for a functional group"""
        if fg_type not in self.node_to_idx:
            # Return zero embedding for unknown FG
            return torch.zeros(self.ent_real.shape[1] * 2)
        
        idx = self.node_to_idx[fg_type]
        # Combine real and imaginary parts
        emb = torch.cat([self.ent_real[idx], self.ent_imag[idx]])
        # Normalize
        emb = F.normalize(emb, p=2, dim=0)
        return emb
    
    def compute_similarity(self, fg1, fg2):
        """Compute similarity between two FG types using embeddings"""
        if fg1 == fg2:
            return 1.0  # Exact match
        
        emb1 = self.get_fg_embedding(fg1)
        emb2 = self.get_fg_embedding(fg2)
        
        # Cosine similarity
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return sim


class FGNode:
    """Represents a functional group node"""
    def __init__(self, idx, fg_type, atom_indices, kg_embeddings=None):
        self.idx = idx  # Position in token sequence
        self.fg_type = fg_type  # e.g., "C_alkyl", "O_ester"
        self.atom_indices = atom_indices  # List of token indices for this FG
        self.element = fg_type.split('_')[0] if '_' in fg_type else fg_type
        self.kg_embeddings = kg_embeddings
        
    def __repr__(self):
        return f"FGNode({self.idx}, {self.fg_type})"
    
    def similarity_to(self, other):
        """Compute learned similarity to another FG node using KGE"""
        if self.kg_embeddings is None:
            # Fallback: exact match
            return 1.0 if self.fg_type == other.fg_type else 0.0
        
        return self.kg_embeddings.compute_similarity(self.fg_type, other.fg_type)


class FGGraph:
    """Functional Group Graph representation"""
    
    def __init__(self, tokens, kg_embeddings=None):
        self.tokens = tokens
        self.kg_embeddings = kg_embeddings
        self.nodes = []  # List of FGNode
        self.edges = []  # List of (i, j) where i, j are node indices
        self._build_graph()
    
    def _build_graph(self):
        """Build FG graph from tokens"""
        # Step 1: Extract FG nodes (atoms with underscore)
        for i, token in enumerate(self.tokens):
            if '_' in token:
                # This is a functional group atom
                node = FGNode(i, token, [i], self.kg_embeddings)
                self.nodes.append(node)
        
        # Step 2: Build edges based on connectivity in token sequence
        # Two FG nodes are connected if they are close in sequence
        # and not separated by major structural boundaries
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                
                # Check if connected (within certain distance and connectivity)
                if self._are_connected(node_i.idx, node_j.idx):
                    self.edges.append((i, j))
    
    def _are_connected(self, idx1, idx2):
        """
        Check if two token positions are connected
        Connected if:
        1. Adjacent (distance <= 3)
        2. Not separated by ring closure (e.g., "1" in "c1ccc1")
        3. Same parenthesis level or connected through structure
        """
        if abs(idx1 - idx2) <= 3:
            # Check tokens in between
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            
            # Count parentheses depth
            paren_depth = 0
            for i in range(start, end):
                token = self.tokens[i]
                if token == '(':
                    paren_depth += 1
                elif token == ')':
                    paren_depth -= 1
            
            # Connected if same level or going down one level
            return abs(paren_depth) <= 1
        
        return False
    
    def get_node_at_token_idx(self, token_idx):
        """Get FG node that contains this token index"""
        for i, node in enumerate(self.nodes):
            if token_idx in node.atom_indices:
                return i
        return None


class FARMGraphComparator:
    """
    Compare molecules using FG Graph matching with learned embeddings
    """
    
    def __init__(self, use_kge=True):
        if use_kge:
            self.kg_embeddings = KGEmbeddings()
        else:
            self.kg_embeddings = None
    
    def smiles_to_tokens(self, smiles):
        """Convert SMILES to FG-enhanced tokens"""
        mol = s2m(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        canonical_smiles = m2s(mol)
        farm_smiles = get_new_smiles_rep(mol)
        tokens = farm_smiles.split()
        
        return tokens, canonical_smiles
    
    def find_maximum_common_subgraph(self, graph1, graph2, similarity_threshold=0.99):
        """
        Find maximum common subgraph between two FG graphs using learned similarity
        Returns: (matched_pairs, unmatched1, unmatched2)
        matched_pairs: list of (node_idx1, node_idx2, similarity_score)
        """
        matched_pairs = []
        matched1 = set()
        matched2 = set()
        
        # Build adjacency similarity matrix
        n1 = len(graph1.nodes)
        n2 = len(graph2.nodes)
        
        # Score each potential match using KGE similarity
        scores = []
        for i in range(n1):
            for j in range(n2):
                node1 = graph1.nodes[i]
                node2 = graph2.nodes[j]
                
                # Compute learned similarity
                base_sim = node1.similarity_to(node2)
                
                # Only consider matches above threshold
                if base_sim < similarity_threshold:
                    continue
                
                # Additional score based on neighborhood compatibility
                score = base_sim * 10.0
                
                # Neighborhood similarity (structural compatibility)
                neighbors1 = [k for k, e in enumerate(graph1.edges) if i in (graph1.edges[k][0], graph1.edges[k][1])]
                neighbors2 = [k for k, e in enumerate(graph2.edges) if j in (graph2.edges[k][0], graph2.edges[k][1])]
                
                # Bonus for similar connectivity
                if len(neighbors1) == len(neighbors2):
                    score += 1.0
                
                scores.append((score, i, j, base_sim))
        
        # Sort by score (descending)
        scores.sort(reverse=True)
        
        # Greedy matching
        for score, i, j, sim in scores:
            if i not in matched1 and j not in matched2:
                matched_pairs.append((i, j, sim))
                matched1.add(i)
                matched2.add(j)
        
        # Find unmatched nodes
        unmatched1 = [i for i in range(n1) if i not in matched1]
        unmatched2 = [j for j in range(n2) if j not in matched2]
        
        return matched_pairs, unmatched1, unmatched2
    
    def build_mcs_string(self, tokens, matched_node_indices, unmatched_node_indices):
        """
        Build MCS string from matched nodes
        Include all tokens except those in unmatched FG nodes
        """
        # Get all token indices that are unmatched FG atoms
        unmatched_token_indices = set()
        for node_idx in unmatched_node_indices:
            # This is an FG node index in the graph
            # Find corresponding token index
            for i, token in enumerate(tokens):
                if '_' in token:
                    # Map back to original token position
                    fg_count = sum(1 for t in tokens[:i] if '_' in t)
                    if fg_count == node_idx:
                        unmatched_token_indices.add(i)
                        break
        
        # Build MCS by skipping unmatched regions
        mcs_parts = []
        skip_depth = 0
        
        for i, token in enumerate(tokens):
            # Check if this token is in an unmatched FG
            if i in unmatched_token_indices:
                continue
            
            # Handle parentheses - skip groups containing unmatched atoms
            if token == '(':
                # Look ahead to see if group contains unmatched
                j = i + 1
                depth = 1
                has_unmatched = False
                
                while j < len(tokens) and depth > 0:
                    if tokens[j] == '(':
                        depth += 1
                    elif tokens[j] == ')':
                        depth -= 1
                    elif j in unmatched_token_indices:
                        has_unmatched = True
                        break
                    j += 1
                
                if has_unmatched:
                    skip_depth = 1
                    continue
                else:
                    mcs_parts.append(token)
            
            elif token == ')':
                if skip_depth > 0:
                    skip_depth = 0
                    continue
                else:
                    mcs_parts.append(token)
            
            elif skip_depth > 0:
                continue
            
            elif '_' in token:
                # FG atom - remove suffix
                base = token.split('_')[0]
                mcs_parts.append(base)
            
            else:
                # Structure token
                mcs_parts.append(token)
        
        mcs_string = ''.join(mcs_parts)
        
        # Convert to canonical SMILES with better starting point
        # Try to rewrite starting from alkyl chain if possible
        try:
            print("using MOL")
            mol = s2m(mcs_string)
            if mol:
                # Find best starting atom (prefer non-ring atoms)
                best_start = 0
                for atom in mol.GetAtoms():
                    if not atom.IsInRing():
                        best_start = atom.GetIdx()
                        break
                
                # Generate SMILES from this starting point
                mcs_string = m2s(mol, rootedAtAtom=best_start)
        except:
            pass
        
        return mcs_string
    
    def group_unmatched_nodes(self, graph, unmatched_node_indices):
        """
        Group unmatched FG nodes that are connected
        Returns list of groups (each group is list of node indices)
        """
        if not unmatched_node_indices:
            return []
        
        # Build adjacency for unmatched nodes
        adj = defaultdict(set)
        for i, j in graph.edges:
            if i in unmatched_node_indices and j in unmatched_node_indices:
                adj[i].add(j)
                adj[j].add(i)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        for node in unmatched_node_indices:
            if node not in visited:
                # DFS to find component
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adj[current] - visited)
                
                components.append(sorted(component))
        
        return components
    
    def format_diff_group(self, graph, node_indices):
        """
        Format a group of unmatched FG nodes
        """
        if not node_indices:
            return ""
        
        # Get all tokens for these nodes
        tokens_list = []
        for node_idx in node_indices:
            node = graph.nodes[node_idx]
            tokens_list.append(node.fg_type)
        
        # Format
        if len(tokens_list) == 1:
            return tokens_list[0]
        else:
            # Join with underscore
            return f"({('_'.join(tokens_list))})"
    
    def compare_smiles(self, smiles1, smiles2, verbose=False):
        """
        Compare two SMILES using FG Graph matching
        """
        start_time = time.time()
        
        # Step 1: Convert to FG-enhanced tokens
        tokens1, canonical1 = self.smiles_to_tokens(smiles1)
        tokens2, canonical2 = self.smiles_to_tokens(smiles2)
        
        # Step 2: Build FG graphs with KGE
        graph1 = FGGraph(tokens1, self.kg_embeddings)
        graph2 = FGGraph(tokens2, self.kg_embeddings)
        
        # Step 3: Find maximum common subgraph
        matched_pairs, unmatched1, unmatched2 = self.find_maximum_common_subgraph(graph1, graph2)
        
        # Step 4: Build MCS string
        mcs_string = self.build_mcs_string(tokens1, 
                                           [pair[0] for pair in matched_pairs],
                                           unmatched1)
        
        # Step 5: Group unmatched nodes
        diff_groups1 = self.group_unmatched_nodes(graph1, unmatched1)
        diff_groups2 = self.group_unmatched_nodes(graph2, unmatched2)
        
        # Step 6: Format output
        serial_parts1 = []
        for i, group in enumerate(diff_groups1, 1):
            fg_label = self.format_diff_group(graph1, group)
            if fg_label:
                # Get position (min token index)
                pos = min([graph1.nodes[idx].idx for idx in group])
                serial_parts1.append((pos, f"{fg_label}:{i}"))
        
        serial_parts2 = []
        for i, group in enumerate(diff_groups2, 1):
            fg_label = self.format_diff_group(graph2, group)
            if fg_label:
                pos = min([graph2.nodes[idx].idx for idx in group])
                serial_parts2.append((pos, f"{fg_label}:{i}"))
        
        # Determine MCS position
        if matched_pairs:
            mcs_pos1 = min([graph1.nodes[pair[0]].idx for pair in matched_pairs])
            mcs_pos2 = min([graph2.nodes[pair[1]].idx for pair in matched_pairs])
        else:
            mcs_pos1 = 0
            mcs_pos2 = 0
        
        # Build output
        output1_parts = serial_parts1 + [(mcs_pos1, mcs_string)]
        output2_parts = serial_parts2 + [(mcs_pos2, mcs_string)]
        output1_parts.sort()
        output2_parts.sort()
        
        output_str1 = " ".join([p[1] for p in output1_parts])
        output_str2 = " ".join([p[1] for p in output2_parts])
        result = f"{output_str1} and {output_str2}"
        
        # Metrics
        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'matched_nodes': len(matched_pairs),
            'diff_groups_mol1': len(diff_groups1),
            'diff_groups_mol2': len(diff_groups2),
            'fg_nodes_mol1': len(graph1.nodes),
            'fg_nodes_mol2': len(graph2.nodes),
            'canonical1': canonical1,
            'canonical2': canonical2
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FARM FG GRAPH COMPARISON")
            print(f"{'='*80}")
            print(f"Input 1: {smiles1}")
            print(f"Canonical 1: {canonical1}")
            print(f"  Tokens: {len(tokens1)}")
            print(f"  FG nodes: {len(graph1.nodes)}")
            print(f"  FG graph edges: {len(graph1.edges)}")
            print(f"  FARM representation: {' '.join(tokens1)}")
            
            print(f"\nInput 2: {smiles2}")
            print(f"Canonical 2: {canonical2}")
            print(f"  Tokens: {len(tokens2)}")
            print(f"  FG nodes: {len(graph2.nodes)}")
            print(f"  FG graph edges: {len(graph2.edges)}")
            print(f"  FARM representation: {' '.join(tokens2)}")
            
            print(f"\nFG Graph Matching:")
            print(f"  Matched FG nodes: {len(matched_pairs)}")
            print(f"  Unmatched mol1: {len(unmatched1)}")
            print(f"  Unmatched mol2: {len(unmatched2)}")
            
            print(f"\nMatched pairs:")
            for pair in matched_pairs:
                if len(pair) == 3:
                    i, j, sim = pair
                    node1 = graph1.nodes[i]
                    node2 = graph2.nodes[j]
                    print(f"    {node1.fg_type} <-> {node2.fg_type} (sim={sim:.3f})")
                else:
                    i, j = pair
                    node1 = graph1.nodes[i]
                    node2 = graph2.nodes[j]
                    print(f"    {node1.fg_type} <-> {node2.fg_type}")
            
            print(f"\nUnmatched mol1:")
            for i in unmatched1:
                print(f"    {graph1.nodes[i].fg_type}")
            
            print(f"\nUnmatched mol2:")
            for j in unmatched2:
                print(f"    {graph2.nodes[j].fg_type}")
            
            print(f"\nMCS (from FG graph): {mcs_string}")
            print(f"\nPerformance:")
            print(f"  Total time: {total_time*1000:.2f} ms")
        
        return result, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compare SMILES using FARM FG Graph matching'
    )
    parser.add_argument('--smiles1', help='First SMILES string')
    parser.add_argument('--smiles2', help='Second SMILES string')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = FARMGraphComparator()
    
    if args.smiles1 and args.smiles2:
        result, metrics = comparator.compare_smiles(
            args.smiles1, args.smiles2, 
            verbose=args.verbose
        )
        
        print(f"\nInput 1: {args.smiles1}")
        print(f"Input 2: {args.smiles2}")
        if metrics:
            print(f"\nCanonical 1: {metrics['canonical1']}")
            print(f"Canonical 2: {metrics['canonical2']}")
        print(f"\nOutput: {result}")
        
        if not args.verbose and metrics:
            print(f"\nMetrics:")
            print(f"  Matched FG nodes: {metrics['matched_nodes']}")
            print(f"  Diff groups: Mol1={metrics['diff_groups_mol1']}, Mol2={metrics['diff_groups_mol2']}")
            print(f"  Time: {metrics['total_time']*1000:.2f} ms")
    else:
        print("Usage: python compare_farm_graph.py --smiles1 'SMILES1' --smiles2 'SMILES2'")
        print("Example:")
        print("  python compare_farm_graph.py --smiles1 'COC(=O)N' --smiles2 'COC(=O)C=C' --verbose")


if __name__ == '__main__':
    main()
