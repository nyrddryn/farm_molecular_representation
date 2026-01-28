"""
FARM Model Inference - Generate embeddings and visualize molecular graphs from SMILES
"""

import torch
import torch.nn as nn
import pickle
import argparse
import ssl
import os
from rdkit import Chem
from rdkit.Chem import Draw, Fragments
from transformers import PreTrainedTokenizerFast, BertModel
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context


class ContrastiveBERT(nn.Module):
    """Contrastive BERT model for unified molecular embeddings"""
    def __init__(self, pretrained_model, tokenizer, projection_dim=128):
        super(ContrastiveBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.projection_head = nn.Linear(self.bert.config.hidden_size, projection_dim)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.activation = nn.GELU()
        self.projection_ln = nn.LayerNorm(projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        
        # CLS token embedding
        cls_output = last_hidden_state[:, 0]
        projected = self.projection_ln(self.projection_head(cls_output))
        
        return projected, cls_output


def get_functional_groups(mol):
    """Detect functional groups in molecule"""
    fg_functions = [
        ('Hydroxyl (-OH)', Fragments.fr_Al_OH),
        ('Aldehyde', Fragments.fr_aldehyde),
        ('Ketone', Fragments.fr_ketone),
        ('Carboxylic acid', Fragments.fr_COO),
        ('Ester', Fragments.fr_ester),
        ('Ether', Fragments.fr_ether),
        ('Amine (-NH2)', Fragments.fr_NH2),
        ('Amide', Fragments.fr_amide),
        ('Nitrile', Fragments.fr_nitrile),
        ('Nitro', Fragments.fr_nitro),
        ('Halide', Fragments.fr_halogen),
        ('Aromatic ring', Fragments.fr_benzene),
    ]
    
    detected = []
    for fg_name, fg_func in fg_functions:
        count = fg_func(mol)
        if count > 0:
            detected.append(f"{fg_name} ({count})")
    
    return detected if detected else ['None']


def get_functional_group_atoms(mol):
    """Get atom indices for each functional group with SMARTS patterns"""
    from rdkit.Chem import AllChem
    
    # SMARTS patterns for functional groups
    patterns = {
        'Carboxylic acid': '[CX3](=O)[OX2H1]',
        'Ester': '[#6][CX3](=O)[OX2H0][#6]',
        'Ketone': '[#6][CX3](=O)[#6]',
        'Aldehyde': '[CX3H1](=O)[#6]',
        'Hydroxyl': '[OX2H]',
        'Ether': '[OD2]([#6])[#6]',
        'Amine': '[NX3;H2,H1;!$(NC=O)]',
        'Amide': '[NX3][CX3](=[OX1])[#6]',
        'Nitrile': '[NX1]#[CX2]',
        'Nitro': '[NX3+](=O)[O-]',
        'Halide': '[F,Cl,Br,I]',
        'Aromatic ring': 'c1ccccc1',
    }
    
    fg_atoms = {}
    for fg_name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                # Flatten all matches to get unique atom indices
                atoms = set()
                for match in matches:
                    atoms.update(match)
                fg_atoms[fg_name] = atoms
    
    return fg_atoms


def draw_molecular_graph(smiles, output_path='molecular_graph.png'):
    """Draw molecular structure and graph representation with functional group highlights"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"\n⚠️  Warning: Cannot parse SMILES to molecular structure")
            print(f"   SMILES: {smiles}")
            print(f"   Possible reasons:")
            print(f"   - Invalid SMILES syntax")
            print(f"   - Unsupported molecular features")
            print(f"   - Chemical inconsistencies")
            print(f"\n   ℹ️  Graph visualization skipped, but embeddings are still available!")
            return None
    except Exception as e:
        print(f"\n⚠️  Error parsing SMILES: {e}")
        print(f"   ℹ️  Graph visualization skipped, but embeddings are still available!")
        return None
    
    # Get functional group atom indices
    fg_atoms = get_functional_group_atoms(mol)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(22, 7))
    
    # Left: 2D structure with FG highlights
    ax1 = plt.subplot(1, 3, 1)
    
    # Highlight functional groups in the molecule
    highlight_atoms = []
    highlight_colors = {}
    fg_colors = {
        'Carboxylic acid': (1.0, 0.7, 0.7),  # Light red
        'Ester': (0.7, 1.0, 0.7),            # Light green
        'Ketone': (0.7, 0.7, 1.0),           # Light blue
        'Aldehyde': (1.0, 1.0, 0.7),         # Light yellow
        'Hydroxyl': (1.0, 0.8, 0.8),         # Pink
        'Ether': (0.9, 0.9, 1.0),            # Light purple
        'Amine': (0.8, 1.0, 1.0),            # Light cyan
        'Amide': (1.0, 0.9, 0.7),            # Light orange
        'Nitrile': (0.8, 0.8, 0.8),          # Light gray
        'Nitro': (1.0, 0.6, 0.6),            # Red
        'Halide': (0.7, 1.0, 0.8),           # Light teal
        'Aromatic ring': (1.0, 1.0, 0.9),    # Very light yellow
    }
    
    for fg_name, atoms in fg_atoms.items():
        for atom_idx in atoms:
            highlight_atoms.append(atom_idx)
            if fg_name in fg_colors:
                highlight_colors[atom_idx] = fg_colors[fg_name]
    
    # Draw molecule with highlights
    if highlight_atoms:
        img = Draw.MolToImage(mol, size=(600, 600), 
                             highlightAtoms=highlight_atoms,
                             highlightAtomColors=highlight_colors)
    else:
        img = Draw.MolToImage(mol, size=(600, 600))
    
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Molecular Structure with FG Highlights', fontsize=14, fontweight='bold', pad=20)
    
    # Middle: Functional group legend and info
    ax2 = plt.subplot(1, 3, 2)
    ax2.axis('off')
    ax2.set_title('Functional Groups Detected', fontsize=14, fontweight='bold', pad=20)
    
    # Display legend for functional groups
    legend_y = 0.95
    if fg_atoms:
        for idx, (fg_name, atoms) in enumerate(fg_atoms.items()):
            color = fg_colors.get(fg_name, (0.8, 0.8, 0.8))
            # Draw color box
            rect = plt.Rectangle((0.1, legend_y - idx*0.08), 0.08, 0.06, 
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax2.add_patch(rect)
            # Add text
            atom_nums = ', '.join([str(a) for a in sorted(atoms)])
            ax2.text(0.22, legend_y - idx*0.08 + 0.03, 
                    f'{fg_name}\nAtoms: [{atom_nums}]',
                    fontsize=11, va='center', fontfamily='monospace')
    else:
        ax2.text(0.5, 0.5, 'No functional groups detected', 
                ha='center', va='center', fontsize=12)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Right: Graph network visualization
    ax3 = plt.subplot(1, 3, 3)
    
    # Build adjacency matrix for graph
    num_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((num_atoms, num_atoms))
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Create positions using spring layout
    np.random.seed(42)
    pos = {}
    
    # Try to use molecular coordinates if available
    try:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        for i in range(num_atoms):
            atom_pos = conf.GetAtomPosition(i)
            pos[i] = (atom_pos.x, atom_pos.y)
    except:
        # Fallback to circular layout
        angle = 2 * np.pi / num_atoms
        for i in range(num_atoms):
            pos[i] = (np.cos(i * angle), np.sin(i * angle))
    
    # Draw nodes with functional group highlighting
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        x, y = pos[i]
        
        # Check if atom is part of a functional group
        atom_fg = None
        for fg_name, atoms in fg_atoms.items():
            if i in atoms:
                atom_fg = fg_name
                break
        
        # Color by functional group if present, otherwise by atom type
        if atom_fg and atom_fg in fg_colors:
            color = fg_colors[atom_fg]
            edge_color = 'darkred'
            linewidth = 3
        elif symbol == 'C':
            color = 'lightgray'
            edge_color = 'black'
            linewidth = 2
        elif symbol == 'O':
            color = 'red'
            edge_color = 'black'
            linewidth = 2
        elif symbol == 'N':
            color = 'blue'
            edge_color = 'black'
            linewidth = 2
        elif symbol == 'S':
            color = 'yellow'
            edge_color = 'black'
            linewidth = 2
        elif symbol in ['F', 'Cl', 'Br', 'I']:
            color = 'green'
            edge_color = 'black'
            linewidth = 2
        else:
            color = 'pink'
            edge_color = 'black'
            linewidth = 2
        
        # Draw node
        circle = plt.Circle((x, y), 0.15, color=color, ec=edge_color, linewidth=linewidth, zorder=2)
        ax3.add_patch(circle)
        
        # Add atom label with index
        ax3.text(x, y, symbol, ha='center', va='center', 
                fontsize=11, fontweight='bold', zorder=3)
        # Add atom index below
        ax3.text(x, y-0.25, str(i), ha='center', va='top',
                fontsize=8, color='gray', zorder=3)
    
    # Draw edges (bonds)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        
        # Line width based on bond type
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            linewidth = 2
            linestyle = '-'
        elif bond_type == Chem.BondType.DOUBLE:
            linewidth = 3
            linestyle = '-'
        elif bond_type == Chem.BondType.TRIPLE:
            linewidth = 4
            linestyle = '-'
        elif bond_type == Chem.BondType.AROMATIC:
            linewidth = 2
            linestyle = '--'
        else:
            linewidth = 2
            linestyle = '-'
        
        ax3.plot([x1, x2], [y1, y2], 'k-', linewidth=linewidth, 
                linestyle=linestyle, alpha=0.6, zorder=1)
    
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Graph with Atom Indices', fontsize=14, fontweight='bold', pad=20)
    
    # Add information text
    num_bonds = mol.GetNumBonds()
    functional_groups = get_functional_groups(mol)
    
    info_text = f"SMILES: {smiles}\n"
    info_text += f"Atoms: {num_atoms} | Bonds: {num_bonds}\n"
    info_text += f"Functional Groups: {', '.join(functional_groups)}"
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph visualization saved to: {output_path}")
    plt.close()
    
    return {
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'functional_groups': functional_groups
    }


def inference_smiles(smiles, 
                    model_path='models/full_farm/checkpoint-252',
                    output_graph='molecular_graph.png'):
    """
    Run inference on SMILES string
    
    Args:
        smiles: SMILES string
        model_path: Path to trained model checkpoint
        output_graph: Path to save graph visualization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    model = ContrastiveBERT(model_path, tokenizer)
    model.to(device)
    model.eval()
    
    print("\n" + "="*80)
    print(f"Processing SMILES: {smiles}")
    print("="*80)
    
    # Tokenize
    encoded = tokenizer(
        [smiles],
        max_length=200,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get embeddings
    with torch.no_grad():
        unified_emb, bert_emb = model(input_ids, attention_mask)
    
    unified_emb = unified_emb.cpu().numpy()[0]
    bert_emb = bert_emb.cpu().numpy()[0]
    
    # Print embeddings
    print("\n📊 EMBEDDINGS:")
    print(f"\n  Unified Embedding (128-dim):")
    print(f"    Shape: (128,)")
    print(f"    L2 norm: {np.linalg.norm(unified_emb):.4f}")
    print(f"    First 10 values: {unified_emb[:10]}")
    
    print(f"\n  BERT CLS Embedding (768-dim):")
    print(f"    Shape: (768,)")
    print(f"    L2 norm: {np.linalg.norm(bert_emb):.4f}")
    print(f"    First 10 values: {bert_emb[:10]}")
    
    # Draw molecular graph
    print("\n🎨 VISUALIZATION:")
    graph_info = draw_molecular_graph(smiles, output_graph)
    
    if graph_info:
        print(f"\n📈 MOLECULAR PROPERTIES:")
        print(f"  • Number of atoms: {graph_info['num_atoms']}")
        print(f"  • Number of bonds: {graph_info['num_bonds']}")
        print(f"  • Functional groups: {', '.join(graph_info['functional_groups'])}")
    else:
        print(f"\n⚠️  Molecular graph could not be generated")
        print(f"   However, embeddings are still valid and can be used for:")
        print(f"   - Molecular similarity comparison")
        print(f"   - Property prediction")
        print(f"   - Drug discovery tasks")
    
    print("\n" + "="*80)
    print("✅ Inference completed successfully!")
    print("="*80 + "\n")
    
    return {
        'unified_embedding': unified_emb,
        'bert_embedding': bert_emb,
        'graph_info': graph_info
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FARM Model Inference')
    parser.add_argument('--smiles', type=str, required=False, default="CC(=O)OC1=CC=CC=C1C(=O)O",
                       help='SMILES string to process')
    parser.add_argument('--model', type=str, 
                       default='bert_model_output/checkpoint-1080',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, 
                       default='molecular_graph.png',
                       help='Output path for graph visualization')
    
    args = parser.parse_args()
    
    inference_smiles(args.smiles, args.model, args.output)
