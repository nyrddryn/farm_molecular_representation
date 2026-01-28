# FARM + GCN Molecular Comparison Pipeline

A modular pipeline for comparing molecular structures using functional group (FG) graphs with GNN-based structure prediction and Knowledge Graph Embeddings (KGE).

## RDKit Usage

**Important:** RDKit is ONLY used for the initial SMILES tokenization step:
- Location: `utils/smiles_tokenizer.py` - function `smiles_to_fg_tokens()`
- Purpose: Convert SMILES string → FG-enhanced tokens
- Process:
  1. Parse SMILES to RDKit Mol object
  2. Detect functional groups using `get_new_smiles_rep()` from helpers.py
  3. Generate canonical SMILES
  4. Return token list (strings only)

After tokenization, the entire pipeline operates on string tokens without any RDKit Mol objects. This ensures clear separation between chemical structure parsing and graph-based analysis.

## Architecture

```
full_pipeline_gnn/
├── models/              # Neural network models
│   ├── __init__.py
│   └── gnn.py          # GCN-based link prediction model
├── embeddings/          # Knowledge Graph Embeddings
│   ├── __init__.py
│   └── kge.py          # ComplEx embeddings handler
├── graph/               # Graph construction
│   ├── __init__.py
│   └── builder.py      # FG graph builder using GNN
├── comparison/          # Graph comparison logic
│   ├── __init__.py
│   ├── comparator.py   # Main comparison orchestrator
│   ├── graph_matcher.py    # Graph matching algorithm
│   └── output_formatter.py # Result formatting
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── rdkit_utils.py      # RDKit wrappers
│   └── smiles_tokenizer.py # SMILES → tokens (ONLY RDKit usage)
├── main.py              # Main entry point
└── README.md            # This file
```

## Module Overview

### 1. Models (`models/`)
- **`gnn.py`**: Contains the `LinkPredictorGNN` class
  - GCN layer for node embedding learning
  - MLP for edge prediction from node pairs
  - Trained to predict connectivity in FG graphs

### 2. Embeddings (`embeddings/`)
- **`kge.py`**: Contains the `KGEmbeddings` class
  - Loads ComplEx knowledge graph embeddings
  - Provides semantic similarity between functional groups
  - Used for node feature initialization

### 3. Graph (`graph/`)
- **`builder.py`**: Contains the `FGGraphBuilder` class
  - Converts SMILES to FG-enhanced tokens (via utils/smiles_tokenizer.py)
  - Extracts functional group nodes
  - Uses GNN to predict graph structure
  - Returns graph with nodes, edges, and embeddings

### 4. Comparison (`comparison/`)
- **`comparator.py`**: Main comparison orchestrator
  - Coordinates the comparison pipeline
  - Handles verbose output and metrics

- **`graph_matcher.py`**: Graph matching logic
  - Hybrid scoring (KGE semantic + GCN structural)
  - Groups unmatched nodes by connectivity

- **`output_formatter.py`**: Result formatting
  - Formats MCS and difference groups
  - Supports both atom-level and FG-level output

### 5. Utils (`utils/`)
- **`rdkit_utils.py`**: RDKit wrapper functions
  - Minimal interface for SMILES handling

### 6. Main (`main.py`)
- Command-line interface
- Argument parsing
- Pipeline orchestration

## Usage

### Basic Usage
```bash
python main.py --smiles1 "CCO" --smiles2 "CC(O)C"
```

### With Options
```bash
python main.py \
    --smiles1 "CCO" \
    --smiles2 "CC(O)C" \
    --verbose \
    --fg-labels \
    --threshold 0.5
```

### As a Library
```python
from graph.builder import FGGraphBuilder
from comparison.comparator import FGGraphComparator

# Initialize
builder = FGGraphBuilder(
    gnn_path='gnn_real_weight/link_prediction_model_epoch49.pth',
    kg_path='data/fgkg.pkl',
    kge_path='kge_model/checkpoints86.pth'
)

comparator = FGGraphComparator(builder)

# Compare
result, metrics = comparator.compare_smiles(
    smiles1="CCO",
    smiles2="CC(O)C",
    verbose=True
)

print(result)
```

## Pipeline Flow

```
SMILES Input
    ↓
FG-Enhanced Representation (rule-based)
    ↓
FG Graph Construction (GNN)
    ├── Node features from KGE
    └── Edge prediction from GCN
    ↓
Graph Matching
    ├── Hybrid scoring (KGE + GCN)
    └── Group unmatched nodes
    ↓
Output Formatting
    ├── MCS extraction
    └── Difference groups
    ↓
Result String
```

## Key Features

1. **Modular Design**: Each component has a single responsibility
2. **Hybrid Scoring**: Combines semantic (KGE) and structural (GCN) similarity
3. **Flexible Output**: Supports both atom-level and FG-level formatting
4. **Extensible**: Easy to add new models or comparison strategies

## Dependencies

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy

## Command-Line Arguments

- `--smiles1`: First SMILES string (required)
- `--smiles2`: Second SMILES string (required)
- `--gnn`: Path to GNN checkpoint (default: `gnn_real_weight/link_prediction_model_epoch49.pth`)
- `--kg`: Path to KG vocabulary (default: `data/fgkg.pkl`)
- `--kge`: Path to KGE embeddings (default: `kge_model/checkpoints86.pth`)
- `--threshold`: Edge prediction threshold (default: 0.5)
- `--fg-labels`: Use FG labels instead of atom-level output
- `--verbose`, `-v`: Show detailed information

## Output Format

### Atom-level (default)
```
C(N):1 CCO and CCO C(OH):1
```

### FG-level (with --fg-labels)
```
(C_amine):1 CCO and CCO (C_alcohol_OH):1
```

## Metrics

- `matched_nodes`: Number of matched FG nodes
- `diff_groups_mol1`: Number of difference groups in molecule 1
- `diff_groups_mol2`: Number of difference groups in molecule 2
- `total_time`: Computation time in seconds
- `canonical1`, `canonical2`: Canonical SMILES representations
