"""
Compare Two SMILES using MCS with FARM Functional Group Detection
Includes performance measurement and cosine similarity calculation

Pipeline:
1. Find the Maximum Common Substructure (MCS) between two SMILES
2. Mark differences using FARM's functional group detection
3. Assign serial numbers based on topology/connection points
4. Calculate cosine similarity between molecule embeddings
5. Output format: atoms_functional_group:serial_number - MCS - other atoms_functional_group:serial_number
"""

import argparse
import sys
import os
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS, MolFromSmiles as s2m, MolToSmiles as m2s
from collections import defaultdict
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from helpers import detect_functional_group, get_new_smiles_rep


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class FARMMCSComparator:
    def __init__(self, model_path='bert_model_output/checkpoint-1080', use_embeddings=True):
        """
        Initialize FARM comparator
        
        Args:
            model_path: Path to FARM BERT model checkpoint
            use_embeddings: Whether to load FARM BERT for embeddings (optional)
        """
        self.use_embeddings = use_embeddings
        self.model = None
        self.tokenizer = None
        
        if use_embeddings:
            try:
                print(f"Loading FARM BERT model from {model_path}...")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
                self.model = BertForMaskedLM.from_pretrained(model_path)
                self.model.eval()
                print("✓ FARM BERT model loaded")
            except Exception as e:
                print(f"Warning: Could not load FARM BERT model: {e}")
                print("Continuing without embeddings...")
                self.use_embeddings = False
    
    def get_farm_embeddings(self, smiles):
        """Get FARM embeddings for a SMILES string"""
        if not self.use_embeddings or self.model is None:
            return None
        
        mol = s2m(smiles)
        if mol is None:
            return None
        
        farm_smiles = get_new_smiles_rep(mol)
        inputs = self.tokenizer(farm_smiles, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]  # (N, 768)
        
        # Average pooling for molecular representation
        mol_embedding = torch.mean(last_hidden_states, dim=0).numpy()
        return mol_embedding
    
    def get_atom_functional_group(self, mol, atom_idx):
        """Get functional group label for an atom using FARM detection"""
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        
        if atom.GetIsAromatic():
            symbol = symbol.lower()
        
        fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
        ring = atom.GetProp('RING') if atom.HasProp('RING') else ''
        
        # Use FARM functional group names as-is
        if fg != '' and ring != '':
            return f"{symbol}_{fg}_{ring}"
        elif fg != '':
            return f"{symbol}_{fg}"
        elif ring != '':
            return f"{symbol}_{ring}"
        else:
            return symbol
    
    def find_connected_diff_atoms(self, mol, diff_atoms):
        """Group connected difference atoms together"""
        if not diff_atoms:
            return []
        
        visited = set()
        groups = []
        
        for start_idx in diff_atoms:
            if start_idx in visited:
                continue
            
            # BFS to find connected component
            group = []
            queue = [start_idx]
            
            while queue:
                current = queue.pop(0)
                if current in visited or current not in diff_atoms:
                    continue
                
                visited.add(current)
                group.append(current)
                
                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx in diff_atoms and n_idx not in visited:
                        queue.append(n_idx)
            
            if group:
                groups.append(sorted(group))
        
        return groups
    
    def get_connection_point(self, mol, atom_idx, mcs_atoms):
        """Find which MCS atom this diff atom connects to"""
        atom = mol.GetAtomWithIdx(atom_idx)
        
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in mcs_atoms:
                return mcs_atoms.index(n_idx)
        
        return None
    
    def format_diff_group(self, mol, atom_group):
        """Format a group of diff atoms with functional group label"""
        if len(atom_group) == 1:
            fg_label = self.get_atom_functional_group(mol, atom_group[0])
            return fg_label
        
        # Multiple atoms - check if they form a recognizable pattern
        if len(atom_group) == 2:
            atom1 = mol.GetAtomWithIdx(atom_group[0])
            atom2 = mol.GetAtomWithIdx(atom_group[1])
            bond = mol.GetBondBetweenAtoms(atom_group[0], atom_group[1])
            
            if bond:
                fg1 = self.get_atom_functional_group(mol, atom_group[0])
                fg2 = self.get_atom_functional_group(mol, atom_group[1])
                
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    return f"({fg1}={fg2})"
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    return f"({fg1}#{fg2})"
        
        # General case
        labels = [self.get_atom_functional_group(mol, idx) for idx in atom_group]
        return f"({'_'.join(labels)})"
    
    def get_fragment_smiles(self, mol, atom_indices):
        """Get SMILES for a fragment"""
        if not atom_indices:
            return ""
        
        emol = Chem.EditableMol(Chem.Mol())
        atom_map = {}
        
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            new_idx = emol.AddAtom(atom)
            atom_map[idx] = new_idx
        
        # Add bonds - keep track of added bonds to avoid duplicates
        added_bonds = set()
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in atom_indices and end_idx in atom_indices:
                    # Create a canonical bond representation (smaller idx first)
                    bond_key = tuple(sorted([begin_idx, end_idx]))
                    if bond_key not in added_bonds:
                        added_bonds.add(bond_key)
                        emol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
        
        fragment_mol = emol.GetMol()
        try:
            smiles = m2s(fragment_mol)
            if '=' in smiles and '(' not in smiles and 'O' in smiles:
                smiles = smiles.replace('C=O', 'C(=O)')
            return smiles
        except:
            return "[fragment]"
    
    def compare_smiles(self, smiles1, smiles2, verbose=False):
        """
        Compare two SMILES using MCS and FARM functional groups
        Returns formatted comparison string and performance metrics
        """
        start_time = time.time()
        
        # Parse molecules and convert to canonical SMILES
        mol1 = s2m(smiles1)
        mol2 = s2m(smiles2)
        
        if mol1 is None or mol2 is None:
            return "Error: Invalid SMILES", None
        
        # Convert to canonical SMILES to avoid atom ordering issues
        smiles1_canonical = m2s(mol1)
        smiles2_canonical = m2s(mol2)
        mol1 = s2m(smiles1_canonical)
        mol2 = s2m(smiles2_canonical)
        
        # Apply FARM functional group detection
        detect_functional_group(mol1)
        detect_functional_group(mol2)
        
        # Find MCS
        mcs_start = time.time()
        mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                    timeout=5,
                                    bondCompare=rdFMCS.BondCompare.CompareAny,
                                    atomCompare=rdFMCS.AtomCompare.CompareElements)
        mcs_time = time.time() - mcs_start
        
        if mcs_result.numAtoms == 0:
            return f"{smiles1} and {smiles2} (No common structure)", {
                'total_time': time.time() - start_time,
                'mcs_time': mcs_time
            }
        
        # Get MCS matches - use GetSubstructMatches to get all possible matches
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        
        # Try to get the best match (prefer matches that minimize disconnected diff groups)
        matches1 = mol1.GetSubstructMatches(mcs_mol)
        matches2 = mol2.GetSubstructMatches(mcs_mol)
        
        # Select the match that results in fewest disconnected diff groups
        def score_match_by_diff_groups(mol, match):
            """Score based on number of disconnected diff groups (fewer is better)"""
            if not match:
                return (float('inf'), float('inf'), float('inf'))
            
            # Get diff atoms
            diff_atoms = [i for i in range(mol.GetNumAtoms()) if i not in match]
            
            # Count disconnected groups
            num_groups = len(self.find_connected_diff_atoms(mol, diff_atoms))
            
            # Secondary criteria: prefer consecutive indices and lower starting index
            sorted_match = sorted(match)
            gaps = sum(1 for i in range(len(sorted_match)-1) if sorted_match[i+1] - sorted_match[i] > 1)
            start_idx = min(match)
            
            return (num_groups, gaps, start_idx)
        
        if matches1:
            match1 = list(min(matches1, key=lambda m: score_match_by_diff_groups(mol1, m)))
        else:
            match1 = []
            
        if matches2:
            match2 = list(min(matches2, key=lambda m: score_match_by_diff_groups(mol2, m)))
        else:
            match2 = []
        
        # Find difference atoms
        diff_atoms1 = [i for i in range(mol1.GetNumAtoms()) if i not in match1]
        diff_atoms2 = [i for i in range(mol2.GetNumAtoms()) if i not in match2]
        
        # Group connected diff atoms
        diff_groups1 = self.find_connected_diff_atoms(mol1, diff_atoms1)
        diff_groups2 = self.find_connected_diff_atoms(mol2, diff_atoms2)
        
        # Sort groups by position first
        diff_groups1.sort(key=lambda g: min(g))
        diff_groups2.sort(key=lambda g: min(g))
        
        # Assign serial numbers sequentially (1, 2, 3...)
        serial_parts1 = []
        for i, group in enumerate(diff_groups1, 1):
            fg_label = self.format_diff_group(mol1, group)
            serial_parts1.append((min(group), fg_label, i))
        
        serial_parts2 = []
        for i, group in enumerate(diff_groups2, 1):
            fg_label = self.format_diff_group(mol2, group)
            serial_parts2.append((min(group), fg_label, i))
        
        # Get MCS SMILES
        mcs_smiles = self.get_fragment_smiles(mol1, match1)
        
        # Determine MCS atom positions
        mcs_set1 = set(match1)
        mcs_set2 = set(match2)
        
        # Build output for mol1 - insert diff groups in order
        output1_parts = []
        
        # Add diff groups before and after MCS in atomic order
        for pos, fg, serial in serial_parts1:
            output1_parts.append((pos, f"{fg}:{serial}"))
        
        # Add MCS at its minimum position
        mcs_pos1 = min(match1) if match1 else 0
        output1_parts.append((mcs_pos1, mcs_smiles))
        output1_parts.sort()
        
        # Build output for mol2
        output2_parts = []
        
        for pos, fg, serial in serial_parts2:
            output2_parts.append((pos, f"{fg}:{serial}"))
        
        mcs_pos2 = min(match2) if match2 else 0
        output2_parts.append((mcs_pos2, mcs_smiles))
        output2_parts.sort()
        
        # Format output
        output_str1 = " ".join([p[1] for p in output1_parts])
        output_str2 = " ".join([p[1] for p in output2_parts])
        result = f"{output_str1} and {output_str2}"
        
        # Calculate embeddings and cosine similarity
        embedding_start = time.time()
        cosine_sim = None
        if self.use_embeddings:
            emb1 = self.get_farm_embeddings(smiles1)
            emb2 = self.get_farm_embeddings(smiles2)
            if emb1 is not None and emb2 is not None:
                cosine_sim = cosine_similarity(emb1, emb2)
        embedding_time = time.time() - embedding_start
        
        # Performance metrics
        metrics = {
            'total_time': time.time() - start_time,
            'mcs_time': mcs_time,
            'embedding_time': embedding_time,
            'cosine_similarity': cosine_sim,
            'mcs_atoms': mcs_result.numAtoms,
            'mcs_bonds': mcs_result.numBonds,
            'diff_groups_mol1': len(serial_parts1),
            'diff_groups_mol2': len(serial_parts2),
            'canonical1': smiles1_canonical,
            'canonical2': smiles2_canonical
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"COMPARISON DETAILS")
            print(f"{'='*80}")
            print(f"Input 1: {smiles1}")
            print(f"Canonical 1: {smiles1_canonical}")
            print(f"  Atoms: {mol1.GetNumAtoms()}, Bonds: {mol1.GetNumBonds()}")
            print(f"  FARM representation: {get_new_smiles_rep(mol1)}")
            print(f"\nInput 2: {smiles2}")
            print(f"Canonical 2: {smiles2_canonical}")
            print(f"  Atoms: {mol2.GetNumAtoms()}, Bonds: {mol2.GetNumBonds()}")
            print(f"  FARM representation: {get_new_smiles_rep(mol2)}")
            print(f"\nMCS: {mcs_smiles}")
            print(f"  Atoms: {mcs_result.numAtoms}, Bonds: {mcs_result.numBonds}")
            print(f"\nPerformance:")
            print(f"  Total time: {metrics['total_time']*1000:.2f} ms")
            print(f"  MCS time: {metrics['mcs_time']*1000:.2f} ms")
            if self.use_embeddings:
                print(f"  Embedding time: {metrics['embedding_time']*1000:.2f} ms")
                print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}" if cosine_sim else "  Cosine similarity: N/A")
        
        return result, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compare SMILES using MCS and FARM with performance metrics'
    )
    parser.add_argument('--smiles1', help='First SMILES string')
    parser.add_argument('--smiles2', help='Second SMILES string')
    parser.add_argument('--model', default='bert_model_output/checkpoint-1080',
                       help='Path to FARM BERT model')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Skip loading FARM BERT (faster, no cosine similarity)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = FARMMCSComparator(
        model_path=args.model,
        use_embeddings=not args.no_embeddings
    )
    
    if args.smiles1 and args.smiles2:
        result, metrics = comparator.compare_smiles(args.smiles1, args.smiles2, verbose=args.verbose)
        print(f"\nInput 1: {args.smiles1}")
        print(f"Input 2: {args.smiles2}")
        if metrics:
            print(f"\nCanonical 1: {metrics['canonical1']}")
            print(f"Canonical 2: {metrics['canonical2']}")
        print(f"\nOutput: {result}")
        
        if not args.verbose and metrics:
            print(f"\nPerformance: {metrics['total_time']*1000:.2f} ms")
            if metrics['cosine_similarity'] is not None:
                print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Test examples
        print("="*80)
        print("FARM MCS COMPARISON - TEST CASES WITH PERFORMANCE")
        print("="*80)
        
        comparator = FARMMCSComparator(model_path='bert_model_output/checkpoint-1080')
        
        test_cases = [
            ("COC(=O)N", "COC(=O)C=C", "COC(=O) N_primary_amine:1 and COC(=O) (C_alkene=C_alkene):1"),
            ("NCOC(=O)N", "CCOC(=O)C=C", "N_primary_amine:1 COC(=O) N_primary_amine:2 and C_alkyl:1 COC(=O) (C_alkene=C_alkene):2"),
            ("ClCCCCCBr", "FCCCCCCl", "Expected: High similarity"),
            ("CCOC(=O)CCC(=O)O", "CCNC(=O)CCC(=O)O", "Expected: Symmetric diff groups")
        ]
        
        for i, (smiles1, smiles2, expected) in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}")
            print(f"{'='*80}")
            result, metrics = comparator.compare_smiles(smiles1, smiles2, verbose=True)
            print(f"\nInput (original): {smiles1} and {smiles2}")
            if metrics:
                print(f"Input (canonical): {metrics['canonical1']} and {metrics['canonical2']}")
            print(f"Output: {result}")
            print(f"Expected: {expected}")
            print(f"Match: {'✓' if result == expected else '✗'}")
        
        print("\n" + "="*80)
        print("USAGE:")
        print("  python compare_mcs_with_performance.py --smiles1 'SMILES1' --smiles2 'SMILES2'")
        print("  python compare_mcs_with_performance.py --smiles1 'SMILES1' --smiles2 'SMILES2' --verbose")
        print("  python compare_mcs_with_performance.py --smiles1 'SMILES1' --smiles2 'SMILES2' --no-embeddings")
        print("="*80)
