#!/usr/bin/env python3
"""
FARM + GCN Molecular Comparison - Main Entry Point

Pipeline:
1. Input SMILES → FG-enhanced representation (rule-based)
2. Build FG Graph using GCN (learned structure prediction)
3. Compare 2 FG graphs → Find MCS + differences
4. Output differences as atom-level groups

Usage:
    cd /home/hoangcm462/farm_molecular_representation
    python -m full_pipeline_gnn.main --smiles1 "CCO" --smiles2 "CC(O)C" --verbose
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_pipeline_gnn.graph.builder import FGGraphBuilder
from full_pipeline_gnn.comparison.comparator import FGGraphComparator


def main():
    """Main entry point for SMILES comparison"""
    parser = argparse.ArgumentParser(
        description='Compare SMILES using GNN-based FG graph construction'
    )
    parser.add_argument('--smiles1', required=True,
                       help='First SMILES string')
    parser.add_argument('--smiles2', required=True,
                       help='Second SMILES string')
    parser.add_argument('--gnn', default='gnn_real_weight/link_prediction_model_epoch49.pth',
                       help='Path to GNN checkpoint')
    parser.add_argument('--kg', default='data/fgkg.pkl',
                       help='Path to knowledge graph vocabulary')
    parser.add_argument('--kge', default='kge_model/checkpoints86.pth',
                       help='Path to KGE embeddings')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Edge prediction threshold (default: 0.5)')
    parser.add_argument('--fg-labels', action='store_true',
                       help='Use FG labels instead of atom-level output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')

    args = parser.parse_args()

    try:
        # Initialize builder
        builder = FGGraphBuilder(
            gnn_path=args.gnn,
            kg_path=args.kg,
            kge_path=args.kge
        )

        # Initialize comparator
        comparator = FGGraphComparator(builder)

        # Compare
        result, metrics = comparator.compare_smiles(
            args.smiles1, args.smiles2,
            verbose=args.verbose,
            use_fg_labels=args.fg_labels
        )

        # Output results
        print(f"\nInput 1: {args.smiles1}")
        print(f"Input 2: {args.smiles2}")
        print(f"\nOutput: {result}")

        if not args.verbose and metrics:
            print(f"\nMetrics:")
            print(f"  Matched nodes: {metrics['matched_nodes']}")
            print(f"  Diff groups: Mol1={metrics['diff_groups_mol1']}, Mol2={metrics['diff_groups_mol2']}")
            print(f"  Time: {metrics['total_time']*1000:.2f} ms")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
