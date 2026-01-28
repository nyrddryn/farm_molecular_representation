#!/usr/bin/env python3
"""
Visualize a single molecule at atom level with functional group coloring and atom numbering.
Each atom is numbered and colored according to its functional group.
"""

import argparse
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import sys
sys.path.append('src')
from helpers import get_structure, set_atom_map_num, preprocess_smiles
from torch_geometric.data import Data

# Color palette for functional groups
FG_COLORS = [
    (1.0, 0.2, 0.2),    # Red
    (0.2, 0.4, 1.0),    # Blue
    (0.2, 0.8, 0.2),    # Green
    (1.0, 0.6, 0.0),    # Orange
    (0.8, 0.2, 0.8),    # Purple
    (0.0, 0.8, 0.8),    # Cyan
    (1.0, 0.8, 0.0),    # Yellow
    (0.8, 0.4, 0.0),    # Brown
    (1.0, 0.4, 0.6),    # Pink
    (0.4, 0.4, 0.4),    # Gray
]

def load_farm_data():
    """Load FARM molecular graphs and feature dictionary."""
    print("📚 Loading FARM data...")
    
    # Load pre-computed molecular graphs
    with open('data/fg_molecular_graph.pkl', 'rb') as f:
        mol_graphs = pickle.load(f)
    print(f"✓ Loaded {len(mol_graphs)} molecular graphs")
    
    # Load functional group embeddings
    with open('data/feature_dict.pkl', 'rb') as f:
        feature_dict = pickle.load(f)
    print(f"✓ Loaded {len(feature_dict)} functional group embeddings")
    
    return mol_graphs, feature_dict

def create_farm_graph_from_smiles(smiles, feature_dict, dim=128):
    """
    Create FARM graph on-the-fly from SMILES string.
    Based on gen_FG_molecular_graph.py implementation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    set_atom_map_num(mol)
    structure, bonds = get_structure(mol)
    
    num_frags = len(structure)
    num_edges = len(bonds)
    
    atom_features = np.zeros((num_frags, dim))
    
    # Process structure and get features
    new_structure = dict()
    for idx, sm in enumerate(structure):
        new_sm = preprocess_smiles(sm)
        
        fg_feature = feature_dict.get(new_sm, np.zeros(dim))
        if torch.is_tensor(fg_feature):
            fg_feature = fg_feature.detach().cpu().numpy()
        
        new_structure[idx] = {
            'fg_feature': fg_feature,
            'fg_smiles': new_sm,  # Store original FG SMILES
            'atom': structure[sm]['atom']
        }
    
    # Process bonds
    new_bonds = []
    for bond in bonds:
        start_idx, end_idx = bond[:2]
        start_frag = None
        end_frag = None
        for key, value in new_structure.items():
            if start_idx in value['atom']:
                start_frag = key
            if end_idx in value['atom']:
                end_frag = key
        if start_frag is not None and end_frag is not None:
            new_bonds.append([start_frag, end_frag])
    
    # Assign features
    for idx, value in new_structure.items():
        atom_features[idx, :] = value['fg_feature']
    
    # Create edge index
    edge_index = torch.zeros((2, len(new_bonds)), dtype=torch.long)
    for bond_idx, bond in enumerate(new_bonds):
        edge_index[0, bond_idx] = bond[0]
        edge_index[1, bond_idx] = bond[1]
    
    # Create Data object with additional attributes
    graph = Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=edge_index,
        batch=torch.zeros(num_frags, dtype=torch.long),
        smiles=smiles
    )
    
    # Store node attributes for later use
    graph.node_attrs = [{'atoms': list(new_structure[i]['atom']), 
                         'fg_smiles': new_structure[i]['fg_smiles']} 
                        for i in range(num_frags)]
    
    return graph

def find_best_matching_fg(node_embedding, feature_dict, threshold=0.95):
    """Find the best matching functional group for a node embedding."""
    best_match = None
    best_similarity = threshold
    
    for fg_name, fg_embedding in feature_dict.items():
        # Convert to numpy if tensor
        if torch.is_tensor(fg_embedding):
            fg_embedding = fg_embedding.detach().cpu().numpy()
        
        # Compute cosine similarity
        node_norm = np.linalg.norm(node_embedding)
        fg_norm = np.linalg.norm(fg_embedding)
        
        if node_norm > 0 and fg_norm > 0:
            similarity = np.dot(node_embedding, fg_embedding) / (node_norm * fg_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = fg_name
    
    return best_match

def get_fg_display_name(fg_smiles):
    """Convert FG SMILES to readable chemical name"""
    patterns = {
        '*O': 'Hydroxyl (-OH)',
        '*C(=O)O': 'Carboxyl (-COOH)',
        '*C(=O)*': 'Carbonyl (C=O)',
        '*N': 'Amino (-NH2)',
        '*c1ccccc1': 'Phenyl (C6H5-)',
        '*c1': 'Aromatic',
        '*C': 'Alkyl (-R)',
        '*CC*': 'Alkyl (-R)',
        '*P': 'Phosphate (-PO4)',
        '*S': 'Thiol (-SH)',
        '*C=C': 'Alkene (C=C)',
        '*C#C': 'Alkyne (C≡C)',
        '*F': 'Fluoro (-F)',
        '*Cl': 'Chloro (-Cl)',
        '*Br': 'Bromo (-Br)',
        '*I': 'Iodo (-I)',
    }
    
    # Check for exact matches first
    if fg_smiles in patterns:
        return patterns[fg_smiles]
    
    # Check for pattern matches
    for pattern, name in patterns.items():
        if pattern in fg_smiles:
            return name
    
    # Clean up the SMILES for display if no match
    clean_name = fg_smiles.replace('*', '').replace('[', '').replace(']', '')
    if len(clean_name) > 20:
        clean_name = clean_name[:20] + '...'
    return clean_name if clean_name else 'Unknown FG'

def extract_fg_info_from_graph(graph_data, feature_dict):
    """Extract functional group information with atom mappings from FARM graph."""
    fg_info = []
    
    # Get node features (embeddings)
    node_features = graph_data.x.numpy()
    
    # For each node, find which FG it represents and which atoms it contains
    for node_idx in range(len(node_features)):
        node_embedding = node_features[node_idx]
        
        # Find best matching FG
        fg_smiles = find_best_matching_fg(node_embedding, feature_dict)
        
        if fg_smiles is None:
            fg_smiles = f"Unknown_FG_{node_idx}"
        
        # Convert SMILES to readable name
        fg_name = get_fg_display_name(fg_smiles)
        
        # Get atom indices for this node
        # In FARM graphs, node attributes contain atom mappings
        if hasattr(graph_data, 'node_attrs') and node_idx < len(graph_data.node_attrs):
            atom_indices = graph_data.node_attrs[node_idx].get('atoms', [node_idx])
        else:
            # Fallback: assume node index corresponds to atom index
            atom_indices = [node_idx]
        
        fg_info.append({
            'name': fg_name,
            'atoms': set(atom_indices)
        })
    
    return fg_info

def visualize_molecule_with_atom_numbers(smiles, fg_info, output_path):
    """
    Visualize a molecule with atoms colored by functional group and numbered.
    
    Args:
        smiles: SMILES string
        fg_info: List of dicts with 'name' and 'atoms' keys
        output_path: Path to save the image
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Invalid SMILES: {smiles}")
        return
    
    num_atoms = mol.GetNumAtoms()
    
    # Create atom to FG mapping and assign colors
    atom_to_fg = {}
    atom_colors = {}
    fg_color_map = {}
    
    for fg_idx, fg in enumerate(fg_info):
        fg_name = fg['name']
        color = FG_COLORS[fg_idx % len(FG_COLORS)]
        fg_color_map[fg_name] = color
        
        for atom_idx in fg['atoms']:
            if atom_idx < num_atoms:
                atom_to_fg[atom_idx] = fg_name
                atom_colors[atom_idx] = color
    
    # Use RDKit's Draw module with atom colors
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    drawer.drawOptions().addAtomIndices = True
    
    # Prepare highlight colors
    highlight_atoms = list(atom_colors.keys())
    highlight_atom_colors = {idx: atom_colors[idx] for idx in highlight_atoms}
    highlight_bonds = []
    highlight_bond_colors = {}
    
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors
    )
    drawer.FinishDrawing()
    
    # Get the image
    png_data = drawer.GetDrawingText()
    
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png_data))
    
    # Create figure with molecule on top and legend below (vertical layout)
    fig = plt.figure(figsize=(10, 12))
    
    # Top: Molecule structure
    ax_mol = plt.subplot(2, 1, 1)
    ax_mol.imshow(img)
    ax_mol.axis('off')
    ax_mol.set_title(f'{smiles}\nAtoms Colored by Functional Group', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Bottom: Legend
    ax_legend = plt.subplot(2, 1, 2)
    ax_legend.axis('off')
    
    # Group atoms by FG
    fg_to_atoms = {}
    for atom_idx, fg_name in atom_to_fg.items():
        if fg_name not in fg_to_atoms:
            fg_to_atoms[fg_name] = []
        fg_to_atoms[fg_name].append(atom_idx)
    
    # Sort FGs by first atom index
    sorted_fgs = sorted(fg_to_atoms.items(), key=lambda x: min(x[1]))
    
    # Draw legend box
    from matplotlib.patches import Rectangle, FancyBboxPatch
    
    # Background box for legend
    legend_box = FancyBboxPatch((0.05, 0.15), 0.9, 0.75,
                               boxstyle="round,pad=0.02",
                               edgecolor='black', facecolor='wheat',
                               linewidth=2, transform=ax_legend.transAxes)
    ax_legend.add_patch(legend_box)
    
    # Title
    ax_legend.text(0.5, 0.82, 'FUNCTIONAL GROUPS:', 
                  fontsize=13, fontweight='bold', ha='center',
                  transform=ax_legend.transAxes, family='monospace')
    
    # Draw each FG entry with colored square
    y_pos = 0.72
    x_start = 0.12
    
    for fg_name, atoms in sorted_fgs:
        color = fg_color_map[fg_name]
        atom_list = sorted(atoms)
        
        # Colored square (filled)
        square = Rectangle((x_start, y_pos-0.018), 0.025, 0.035, 
                          color=color, transform=ax_legend.transAxes,
                          edgecolor='black', linewidth=1.5)
        ax_legend.add_patch(square)
        
        # FG name and atoms in format: "■ FG_name\n    Atoms: {1, 2, 3}"
        atom_str = '{' + ', '.join(map(str, atom_list)) + '}'
        text = f"  {fg_name}\n      Atoms: {atom_str}"
        ax_legend.text(x_start + 0.04, y_pos, text, 
                      fontsize=11, va='center', 
                      transform=ax_legend.transAxes, family='monospace')
        
        y_pos -= 0.11
    
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()

def visualize_single_molecule(smiles, output_path, mol_graphs, feature_dict):
    """
    Main function to visualize a single molecule with FG coloring and atom numbering.
    
    Args:
        smiles: SMILES string of the molecule
        output_path: Path to save the visualization
        mol_graphs: Dictionary of pre-computed FARM graphs
        feature_dict: Dictionary of FG embeddings
    """
    print("\n" + "="*70)
    print(f"🔬 VISUALIZING MOLECULE: {smiles}")
    print("="*70)
    
    # Try to find pre-computed graph
    graph_data = None
    if smiles in mol_graphs:
        graph_data = mol_graphs[smiles]
        print(f"✓ Found pre-computed FARM graph")
    else:
        print(f"⚠ No pre-computed graph found for this SMILES")
        print(f"🔨 Creating FARM graph on-the-fly...")
        
        # Create graph on-the-fly
        graph_data = create_farm_graph_from_smiles(smiles, feature_dict)
        
        if graph_data is None:
            print(f"❌ Failed to create FARM graph")
            return
        
        print(f"✓ Successfully created FARM graph with {len(graph_data.x)} functional groups")
    
    # Extract FG information from graph
    fg_info = extract_fg_info_from_graph(graph_data, feature_dict)
    
    print(f"\n📊 Found {len(fg_info)} functional groups:")
    for i, fg in enumerate(fg_info, 1):
        atom_list = sorted(list(fg['atoms']))
        print(f"   {i}. {fg['name']} - Atoms: {atom_list}")
    
    # Visualize
    visualize_molecule_with_atom_numbers(smiles, fg_info, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize a molecule with atom-level FG coloring and numbering'
    )
    parser.add_argument('--smiles', type=str, required=True,
                       help='SMILES string of the molecule')
    parser.add_argument('--output', type=str, default='molecule_atoms_numbered.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    # Load FARM data
    mol_graphs, feature_dict = load_farm_data()
    
    # Visualize molecule
    visualize_single_molecule(args.smiles, args.output, mol_graphs, feature_dict)
    
    print("\n✅ Done!")

if __name__ == '__main__':
    main()
