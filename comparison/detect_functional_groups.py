"""
Detect and count functional groups in a SMILES string using FARM
Shows how many functional groups are present and what they are
"""

import argparse
import sys
import os
from collections import Counter

# Add src to path to import helpers
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from helpers import detect_functional_group, get_new_smiles_rep
from rdkit.Chem import MolFromSmiles as s2m


def detect_groups_in_smiles(smiles):
    """
    Detect all functional groups in a SMILES string using FARM
    Returns: list of functional groups found
    """
    mol = s2m(smiles)
    if mol is None:
        return None, "Error: Invalid SMILES"
    
    # Apply FARM's functional group detection
    detect_functional_group(mol)
    
    # Collect all functional groups
    functional_groups = []
    ring_structures = []
    atom_details = []
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if atom.GetIsAromatic():
            symbol = symbol.lower()
        
        # Get functional group and ring properties
        fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
        ring = atom.GetProp('RING') if atom.HasProp('RING') else ''
        
        # Build the full annotation
        if fg != '' and ring != '':
            feature = f"{symbol}_{fg}_{ring}"
            functional_groups.append(fg)
            ring_structures.append(ring)
        elif fg != '':
            feature = f"{symbol}_{fg}"
            functional_groups.append(fg)
        elif ring != '':
            feature = f"{symbol}_{ring}"
            ring_structures.append(ring)
        else:
            feature = symbol
        
        atom_details.append({
            'idx': atom_idx,
            'symbol': symbol,
            'fg': fg,
            'ring': ring,
            'full_annotation': feature
        })
    
    return mol, functional_groups, ring_structures, atom_details


def main():
    parser = argparse.ArgumentParser(
        description='Detect functional groups in a SMILES string using FARM'
    )
    parser.add_argument('smiles', help='SMILES string to analyze')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed atom-by-atom breakdown')
    parser.add_argument('--show-farm', action='store_true',
                       help='Show FARM FG-enhanced SMILES representation')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"FUNCTIONAL GROUP DETECTION FOR: {args.smiles}")
    print("="*80)
    
    result = detect_groups_in_smiles(args.smiles)
    
    if result[0] is None:
        print(result[1])
        return
    
    mol, functional_groups, ring_structures, atom_details = result
    
    # Count unique functional groups
    fg_counts = Counter(functional_groups)
    ring_counts = Counter(ring_structures)
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total atoms: {mol.GetNumAtoms()}")
    print(f"  Total bonds: {mol.GetNumBonds()}")
    print(f"  Unique functional groups detected: {len(fg_counts)}")
    print(f"  Unique ring structures detected: {len(ring_counts)}")
    
    if fg_counts:
        print(f"\n🔬 FUNCTIONAL GROUPS FOUND:")
        for fg, count in sorted(fg_counts.items()):
            print(f"  - {fg}: {count} occurrence(s)")
    else:
        print(f"\n🔬 FUNCTIONAL GROUPS FOUND: None")
    
    if ring_counts:
        print(f"\n💍 RING STRUCTURES FOUND:")
        for ring, count in sorted(ring_counts.items()):
            print(f"  - {ring}: {count} occurrence(s)")
    
    if args.verbose:
        print(f"\n📋 DETAILED ATOM-BY-ATOM BREAKDOWN:")
        print(f"{'Atom':<6} {'Symbol':<8} {'Functional Group':<25} {'Ring':<15} {'Full Annotation':<30}")
        print("-"*90)
        for atom_info in atom_details:
            print(f"{atom_info['idx']:<6} {atom_info['symbol']:<8} {atom_info['fg']:<25} {atom_info['ring']:<15} {atom_info['full_annotation']:<30}")
    
    if args.show_farm:
        farm_rep = get_new_smiles_rep(mol)
        print(f"\n🌾 FARM FG-ENHANCED SMILES REPRESENTATION:")
        print(f"  {farm_rep}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Test examples
        print("="*80)
        print("FARM FUNCTIONAL GROUP DETECTION - TEST EXAMPLES")
        print("="*80)
        
        test_smiles = [
            "COC(=O)N",
            "NCOC(=O)N",
            "CCO",
            "c1ccccc1",
            "CC(=O)Nc1ccccc1",
            "NNc1nncc2ccccc12"
        ]
        
        for smiles in test_smiles:
            print(f"\n{'='*80}")
            print(f"SMILES: {smiles}")
            print(f"{'='*80}")
            
            result = detect_groups_in_smiles(smiles)
            if result[0] is None:
                print(result[1])
                continue
            
            mol, functional_groups, ring_structures, atom_details = result
            fg_counts = Counter(functional_groups)
            ring_counts = Counter(ring_structures)
            
            print(f"Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")
            print(f"Unique functional groups: {len(fg_counts)}")
            
            if fg_counts:
                print(f"Functional groups found:")
                for fg, count in sorted(fg_counts.items()):
                    print(f"  - {fg}: {count}")
            
            if ring_counts:
                print(f"Ring structures found:")
                for ring, count in sorted(ring_counts.items()):
                    print(f"  - {ring}: {count}")
            
            # Show FARM representation
            farm_rep = get_new_smiles_rep(mol)
            print(f"FARM representation: {farm_rep}")
        
        print("\n" + "="*80)
        print("USAGE:")
        print("  python detect_functional_groups.py 'SMILES'")
        print("  python detect_functional_groups.py 'SMILES' --verbose")
        print("  python detect_functional_groups.py 'SMILES' --show-farm")
        print("="*80)
