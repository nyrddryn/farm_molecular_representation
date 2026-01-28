"""
FARM + GCN Molecular Comparison Pipeline

This package provides modular components for:
- Knowledge Graph Embeddings (KGE)
- GNN-based link prediction
- FG graph construction
- Molecular comparison
"""

from .models.gnn import LinkPredictorGNN
from .embeddings.kge import KGEmbeddings
from .graph.builder import FGGraphBuilder
from .comparison.comparator import FGGraphComparator

__all__ = [
    'LinkPredictorGNN',
    'KGEmbeddings',
    'FGGraphBuilder',
    'FGGraphComparator',
]

__version__ = '1.0.0'
