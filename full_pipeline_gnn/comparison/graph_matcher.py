"""
Graph Matching Logic

This module handles matching between two FG graphs using
hybrid scoring (KGE semantic + GCN structural similarity).
"""

import torch
import torch.nn.functional as F
from collections import defaultdict


class GraphMatcher:
    """
    Match two FG graphs using hybrid scoring
    """

    def __init__(self, kge):
        """
        Initialize graph matcher

        Args:
            kge: KGEmbeddings instance
        """
        self.kge = kge

    def match_graphs(self, graph1, graph2, similarity_threshold=0.65):
        """
        Match two FG graphs using HYBRID scoring:
        - KGE semantic similarity (primary, weight: 0.7)
        - GCN structural embeddings (secondary, weight: 0.3)

        Args:
            graph1 (dict): First FG graph
            graph2 (dict): Second FG graph
            similarity_threshold (float): Minimum similarity for matching

        Returns:
            tuple: (matched_pairs, unmatched1, unmatched2)
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

    def group_unmatched_nodes(self, graph, unmatched_node_indices):
        """
        Group unmatched nodes based on:
        1. Graph connectivity (GNN edges)
        2. Token position proximity (for disconnected groups)

        Args:
            graph (dict): FG graph structure
            unmatched_node_indices (list): List of unmatched node indices

        Returns:
            list: List of node groups [[i, j], [k], ...]
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

    def verify_edge_consistency(self, graph1, graph2, matched_pairs):
        """
        Verify edge consistency between matched nodes

        This detects topological differences even when all nodes match.
        Example: Branched vs Linear structures

        Args:
            graph1 (dict): First FG graph
            graph2 (dict): Second FG graph
            matched_pairs (list): [(i, j, score), ...] matched node pairs

        Returns:
            dict: {
                'missing_edges': [(i, j), ...],  # Edges in G1 not in G2
                'extra_edges': [(i, j), ...],    # Edges in G2 not in G1
                'topology1': str,                # 'linear', 'branched', 'cyclic'
                'topology2': str,
                'edge_consistency': float         # 0-1 score
            }
        """
        if not matched_pairs:
            return {
                'missing_edges': [],
                'extra_edges': [],
                'topology1': 'unknown',
                'topology2': 'unknown',
                'edge_consistency': 1.0
            }

        # Create mapping: graph1_idx -> graph2_idx
        node_map_1to2 = {i: j for i, j, _ in matched_pairs}
        node_map_2to1 = {j: i for i, j, _ in matched_pairs}

        # Get edges for matched nodes only
        edges1 = set()
        edges2 = set()

        for src, dst in graph1['edges']:
            if src in node_map_1to2 and dst in node_map_1to2:
                # Map to graph2 indices
                mapped_edge = (node_map_1to2[src], node_map_1to2[dst])
                edges1.add(mapped_edge)

        for src, dst in graph2['edges']:
            if src in node_map_2to1 and dst in node_map_2to1:
                edges2.add((src, dst))

        # Find differences
        missing_edges = list(edges1 - edges2)  # In G1 but not in G2
        extra_edges = list(edges2 - edges1)    # In G2 but not in G1

        # Compute edge consistency score
        if len(edges1) == 0 and len(edges2) == 0:
            edge_consistency = 1.0
        else:
            total_expected = len(edges1) + len(edges2)
            total_diff = len(missing_edges) + len(extra_edges)
            edge_consistency = 1.0 - (total_diff / total_expected) if total_expected > 0 else 1.0

        # Determine topology types
        topology1 = self._determine_topology(graph1, list(node_map_1to2.keys()))
        topology2 = self._determine_topology(graph2, list(node_map_2to1.keys()))

        return {
            'missing_edges': missing_edges,
            'extra_edges': extra_edges,
            'topology1': topology1,
            'topology2': topology2,
            'edge_consistency': edge_consistency
        }

    def _determine_topology(self, graph, node_subset):
        """
        Determine topology type of graph

        Args:
            graph (dict): FG graph
            node_subset (list): Subset of nodes to analyze

        Returns:
            str: 'linear', 'branched', 'cyclic', or 'complex'
        """
        if len(node_subset) <= 1:
            return 'single'

        # Build adjacency for subset
        adj = defaultdict(set)
        for src, dst in graph['edges']:
            if src in node_subset and dst in node_subset:
                adj[src].add(dst)
                adj[dst].add(src)

        # Calculate degree distribution
        degrees = [len(adj[node]) for node in node_subset]
        max_degree = max(degrees) if degrees else 0
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        # Detect cycles (simple check)
        has_cycle = self._has_cycle(adj, node_subset)

        # Classify
        if has_cycle:
            return 'cyclic'
        elif max_degree >= 3:
            return 'branched'  # Hub node exists
        elif max_degree == 2 and avg_degree >= 1.5:
            return 'linear'    # Chain structure
        elif max_degree <= 1:
            return 'disconnected'
        else:
            return 'complex'

    def _has_cycle(self, adj, nodes):
        """
        Check if graph has cycle using DFS

        Args:
            adj (dict): Adjacency list
            nodes (list): Node indices

        Returns:
            bool: True if cycle exists
        """
        if len(nodes) < 3:
            return False

        visited = set()

        def dfs(node, parent):
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True  # Back edge found = cycle
            return False

        for start_node in nodes:
            if start_node not in visited:
                if dfs(start_node, None):
                    return True

        return False
