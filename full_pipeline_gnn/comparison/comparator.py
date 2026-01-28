"""
FG Graph Comparator

This module compares two functional group graphs to find
the Maximum Common Substructure (MCS) and differences.
"""

import torch
import torch.nn.functional as F
import time
from collections import defaultdict

from full_pipeline_gnn.comparison.graph_matcher import GraphMatcher
from full_pipeline_gnn.comparison.output_formatter import OutputFormatter


class FGGraphComparator:
    """
    Compare two FG graphs and extract MCS + differences
    Uses both GCN structure and KGE similarity
    """

    def __init__(self, builder):
        """
        Initialize the comparator

        Args:
            builder: FGGraphBuilder instance
        """
        self.builder = builder
        self.kge = builder.kge
        self.matcher = GraphMatcher(self.kge)
        self.formatter = OutputFormatter()

    def compare_smiles(self, smiles1, smiles2, verbose=False, use_fg_labels=False):
        """
        Complete comparison pipeline

        Args:
            smiles1 (str): First SMILES string
            smiles2 (str): Second SMILES string
            verbose (bool): Print detailed information
            use_fg_labels (bool): If True, use FG labels; if False, use atom-level

        Returns:
            tuple: (result_string, metrics_dict)
        """
        start_time = time.time()

        # Step 1: Build FG graphs
        print("Building FG graphs...")
        graph1 = self.builder.build_fg_graph(smiles1)
        graph2 = self.builder.build_fg_graph(smiles2)

        # Step 2: Match graphs
        print("Matching graphs...")
        matched, unmatched1, unmatched2 = self.matcher.match_graphs(graph1, graph2)

        # Step 3: Verify edge consistency (NEW!)
        print("Verifying edge consistency...")
        edge_analysis = self.matcher.verify_edge_consistency(graph1, graph2, matched)

        # Step 4: Extract MCS
        matched_nodes1 = [pair[0] for pair in matched]
        matched_nodes2 = [pair[1] for pair in matched]

        # Step 5: Group differences
        diff_groups1 = self.matcher.group_unmatched_nodes(graph1, unmatched1)
        diff_groups2 = self.matcher.group_unmatched_nodes(graph2, unmatched2)

        # Step 6: Format output
        result = self.formatter.format_comparison_result(
            graph1, graph2,
            matched_nodes1, matched_nodes2,
            diff_groups1, diff_groups2,
            use_fg_labels,
            edge_analysis  # NEW: Include edge analysis
        )

        # Metrics
        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'matched_nodes': len(matched),
            'diff_groups_mol1': len(diff_groups1),
            'diff_groups_mol2': len(diff_groups2),
            'canonical1': graph1['canonical'],
            'canonical2': graph2['canonical'],
            # NEW: Edge metrics
            'missing_edges': len(edge_analysis['missing_edges']),
            'extra_edges': len(edge_analysis['extra_edges']),
            'edge_consistency': edge_analysis['edge_consistency'],
            'topology1': edge_analysis['topology1'],
            'topology2': edge_analysis['topology2']
        }

        if verbose:
            self._print_verbose_output(
                smiles1, smiles2,
                graph1, graph2,
                matched, unmatched1, unmatched2,
                edge_analysis,  # NEW: Pass edge analysis
                result, total_time
            )

        return result, metrics

    def _print_verbose_output(self, smiles1, smiles2, graph1, graph2,
                              matched, unmatched1, unmatched2, edge_analysis, result, total_time):
        """Print detailed comparison information"""
        # Extract MCS for display
        mcs_atoms = []
        for i, j, score in matched:
            fg_token = graph1['nodes'][i]
            base_atom = fg_token.split('_')[0] if '_' in fg_token else fg_token
            mcs_atoms.append(base_atom)
        mcs_string = ''.join(mcs_atoms)

        print(f"\n{'='*80}")
        print(f"GNN-BASED FG GRAPH COMPARISON (With Edge Verification)")
        print(f"{'='*80}")
        print(f"Molecule 1: {smiles1}")
        print(f"  Canonical: {graph1['canonical']}")
        print(f"  FG nodes: {len(graph1['nodes'])}")
        print(f"  GNN predicted edges: {len(graph1['edges'])}")
        print(f"  Topology: {edge_analysis['topology1']}")

        print(f"\nMolecule 2: {smiles2}")
        print(f"  Canonical: {graph2['canonical']}")
        print(f"  FG nodes: {len(graph2['nodes'])}")
        print(f"  GNN predicted edges: {len(graph2['edges'])}")
        print(f"  Topology: {edge_analysis['topology2']}")

        print(f"\nGraph Matching:")
        print(f"  Matched nodes: {len(matched)}")
        print(f"  Unmatched mol1: {len(unmatched1)}")
        print(f"  Unmatched mol2: {len(unmatched2)}")

        print(f"\nEdge Consistency:")
        print(f"  Missing edges (in G1, not in G2): {len(edge_analysis['missing_edges'])}")
        print(f"  Extra edges (in G2, not in G1): {len(edge_analysis['extra_edges'])}")
        print(f"  Edge consistency score: {edge_analysis['edge_consistency']:.3f}")
        if edge_analysis['topology1'] != edge_analysis['topology2']:
            print(f"  ⚠️  TOPOLOGY MISMATCH: {edge_analysis['topology1']} vs {edge_analysis['topology2']}")

        print(f"\nMCS: {mcs_string}")
        print(f"\nTime: {total_time*1000:.2f} ms")
