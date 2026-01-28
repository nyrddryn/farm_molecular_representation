"""
Compare Two SMILES using MCS with Functional Group Marking
Pipeline:
1. Find the Maximum Common Substructure (MCS) between two SMILES
2. Mark differences using functional group annotations
3. Assign serial numbers to track positions
4. Output format: atoms_functional_group:serial_number - MCS - other atoms_functional_group:serial_number
"""

import argparse
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import defaultdict


def classify_atom_functional_group(mol, atom_idx, atom):
    """
    Classify an atom into its functional group type
    Returns a readable functional group name
    """
    symbol = atom.GetSymbol()
    hybridization = atom.GetHybridization()
    total_h = atom.GetTotalNumHs()
    neighbors = atom.GetNeighbors()
    num_neighbors = len(neighbors)
    is_aromatic = atom.GetIsAromatic()
    
    # Special handling for common functional groups
    if symbol == 'N':
        # Check for amine types
        if not any(mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()).GetBondType() != Chem.BondType.SINGLE 
                   for n in neighbors):
            if total_h == 2:
                return 'N_primary'
            elif total_h == 1:
                return 'N_secondary'
            elif total_h == 0 and num_neighbors == 3:
                return 'N_tertiary'
        # Check for nitro, nitrile, etc.
        for neighbor in neighbors:
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            if neighbor.GetSymbol() == 'C' and bond.GetBondType() == Chem.BondType.TRIPLE:
                return 'N_nitrile'
        if is_aromatic:
            return 'N_aromatic'
        return 'N'
    
    elif symbol == 'O':
        # Check for various oxygen-containing groups
        if total_h == 1 and num_neighbors == 1:
            return 'O_hydroxyl'
        elif num_neighbors == 2:
            c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
            if len(c_neighbors) == 2:
                return 'O_ether'
        # Check for carbonyl
        for neighbor in neighbors:
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE and neighbor.GetSymbol() == 'C':
                return 'O_carbonyl'
        return 'O'
    
    elif symbol == 'C':
        # Check for alkene, alkyne
        double_bonds = sum(1 for n in neighbors 
                          if mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)
        triple_bonds = sum(1 for n in neighbors 
                          if mol.GetBondBetweenAtoms(atom_idx, n.GetIdx()).GetBondType() == Chem.BondType.TRIPLE)
        
        if triple_bonds > 0:
            return 'C_alkyne'
        elif double_bonds > 0:
            # Check if it's part of carbonyl or alkene
            for neighbor in neighbors:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    if neighbor.GetSymbol() == 'O':
                        return 'C_carbonyl'
                    elif neighbor.GetSymbol() == 'C':
                        return 'C_alkene'
            return 'C_alkene'
        elif is_aromatic:
            return 'C_aromatic'
        else:
            return 'C_alkyl'
    
    elif symbol == 'S':
        if total_h == 1:
            return 'S_thiol'
        return 'S_sulfide'
    
    elif symbol in ['F', 'Cl', 'Br', 'I']:
        return f'{symbol}_halide'
    
    elif symbol == 'P':
        return 'P_phosphate'
    
    # Default: return just the element symbol
    return symbol


def get_atom_group_with_neighbors(mol, atom_idx, direction='left'):
    """
    Get a group of connected atoms (for display purposes)
    Returns a list of atom indices that form a connected group
    """
    visited = set()
    to_visit = [atom_idx]
    group = []
    
    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        group.append(current)
        
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited:
                to_visit.append(n_idx)
    
    return group


def find_mcs_and_differences(smiles1, smiles2):
    """
    Find Maximum Common Substructure and identify differences
    Returns: (mcs_smiles, diff_atoms_mol1, diff_atoms_mol2, mol1, mol2, mcs_atoms1, mcs_atoms2)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return None, None, None, None, None, None, None
    
    # Find MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                timeout=5,
                                bondCompare=rdFMCS.BondCompare.CompareAny,
                                atomCompare=rdFMCS.AtomCompare.CompareElements)
    
    if mcs_result.numAtoms == 0:
        return "", list(range(mol1.GetNumAtoms())), list(range(mol2.GetNumAtoms())), mol1, mol2, [], []
    
    # Get MCS as a molecule
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    if mcs_mol is None:
        return mcs_result.smartsString, [], [], mol1, mol2, [], []
    
    # Find matching atoms in both molecules
    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)
    
    # Convert to lists
    mcs_atoms1 = list(match1)
    mcs_atoms2 = list(match2)
    
    # Find different atoms
    all_atoms1 = set(range(mol1.GetNumAtoms()))
    all_atoms2 = set(range(mol2.GetNumAtoms()))
    
    diff_atoms1 = sorted(list(all_atoms1 - set(mcs_atoms1)))
    diff_atoms2 = sorted(list(all_atoms2 - set(mcs_atoms2)))
    
    # Get MCS SMILES
    mcs_smiles = mcs_result.smartsString
    try:
        # Try to convert to readable SMILES
        mcs_canonical = Chem.MolToSmiles(mcs_mol)
        if mcs_canonical:
            mcs_smiles = mcs_canonical
    except:
        pass
    
    return mcs_smiles, diff_atoms1, diff_atoms2, mol1, mol2, mcs_atoms1, mcs_atoms2


def group_consecutive_atoms(mol, atom_indices):
    """
    Group consecutive connected atoms into fragments
    Returns list of groups where each group is a list of connected atom indices
    """
    if not atom_indices:
        return []
    
    visited = set()
    groups = []
    
    for start_idx in atom_indices:
        if start_idx in visited:
            continue
        
        # BFS to find connected component
        group = []
        queue = [start_idx]
        
        while queue:
            current = queue.pop(0)
            if current in visited or current not in atom_indices:
                continue
            
            visited.add(current)
            group.append(current)
            
            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in atom_indices and n_idx not in visited:
                    queue.append(n_idx)
        
        if group:
            groups.append(sorted(group))
    
    return groups


def get_fragment_smiles(mol, atom_indices):
    """Get SMILES representation of a fragment"""
    if not atom_indices:
        return ""
    
    # Create a new molecule with only these atoms
    emol = Chem.EditableMol(Chem.Mol())
    atom_map = {}
    
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_idx = emol.AddAtom(atom)
        atom_map[idx] = new_idx
    
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in atom_indices and end_idx in atom_indices:
                if begin_idx < idx:  # Avoid duplicate bonds
                    continue
                emol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
    
    fragment_mol = emol.GetMol()
    try:
        return Chem.MolToSmiles(fragment_mol)
    except:
        return "[fragment]"


def format_difference_group(mol, atom_group):
    """
    Format a group of different atoms with functional group annotation
    Returns: string in format "(atoms_functional_group)"
    """
    if not atom_group:
        return ""
    
    # Classify atoms
    atom_classifications = []
    for atom_idx in atom_group:
        atom = mol.GetAtomWithIdx(atom_idx)
        fg_type = classify_atom_functional_group(mol, atom_idx, atom)
        atom_classifications.append((atom_idx, atom.GetSymbol(), fg_type))
    
    # Check if all atoms form a cohesive functional group
    # For simple cases, try to get fragment SMILES
    fragment_smiles = get_fragment_smiles(mol, atom_group)
    
    # Build output string
    if len(atom_group) == 1:
        atom_idx, symbol, fg_type = atom_classifications[0]
        return f"{fg_type}"
    else:
        # Multiple atoms - try to represent as a cohesive group
        # Check for common patterns like C=C (alkene)
        if len(atom_group) == 2:
            atom1 = mol.GetAtomWithIdx(atom_group[0])
            atom2 = mol.GetAtomWithIdx(atom_group[1])
            bond = mol.GetBondBetweenAtoms(atom_group[0], atom_group[1])
            
            if bond:
                if (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C' and 
                    bond.GetBondType() == Chem.BondType.DOUBLE):
                    return "(C_alkene=C_alkene)"
                elif (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C' and 
                      bond.GetBondType() == Chem.BondType.TRIPLE):
                    return "(C_alkyne#C_alkyne)"
        
        # General case: list all atoms
        parts = []
        for atom_idx, symbol, fg_type in atom_classifications:
            parts.append(fg_type)
        return f"({'_'.join(parts)})"


def compare_smiles_with_mcs(smiles1, smiles2):
    """
    Main function to compare two SMILES using MCS and functional group marking
    Returns formatted comparison string
    """
    # Step 1: Find MCS
    result = find_mcs_and_differences(smiles1, smiles2)
    if result[0] is None:
        return "Error: Invalid SMILES input"
    
    mcs_smiles, diff_atoms1, diff_atoms2, mol1, mol2, mcs_atoms1, mcs_atoms2 = result
    
    # Step 2: Group different atoms into connected fragments
    groups1 = group_consecutive_atoms(mol1, diff_atoms1)
    groups2 = group_consecutive_atoms(mol2, diff_atoms2)
    
    # Step 3: Format differences with functional group marking
    parts1 = []
    for i, group in enumerate(groups1, 1):
        fg_str = format_difference_group(mol1, group)
        parts1.append(f"{fg_str}:{i}")
    
    parts2 = []
    for i, group in enumerate(groups2, 1):
        fg_str = format_difference_group(mol2, group)
        parts2.append(f"{fg_str}:{i}")
    
    # Step 4: Get MCS string (try to convert to readable format)
    if mcs_smiles.startswith('['):
        # It's a SMARTS pattern, try to get the actual SMILES from molecule
        mcs_display = mcs_smiles
        # Try to extract a cleaner representation
        try:
            # Get the actual SMILES of MCS from one of the molecules
            if mcs_atoms1:
                mcs_part = get_fragment_smiles(mol1, mcs_atoms1)
                if mcs_part and mcs_part != "[fragment]":
                    mcs_display = mcs_part
        except:
            pass
    else:
        mcs_display = mcs_smiles
    
    # Step 5: Build output format
    diff_str1 = " ".join(parts1) if parts1 else ""
    diff_str2 = " ".join(parts2) if parts2 else ""
    
    # Format: diff1 MCS diff2
    output_parts = []
    if diff_str1:
        output_parts.append(diff_str1)
    if mcs_display:
        output_parts.append(mcs_display)
    if diff_str2:
        output_parts.append(diff_str2)
    
    return " ".join(output_parts)


def compare_smiles_with_mcs_detailed(smiles1, smiles2):
    """
    Enhanced version that provides more detailed output matching the expected format
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return "Error: Invalid SMILES input"
    
    # Find MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                timeout=5,
                                bondCompare=rdFMCS.BondCompare.CompareAny,
                                atomCompare=rdFMCS.AtomCompare.CompareElements)
    
    if mcs_result.numAtoms == 0:
        # No common structure
        return f"{smiles1} and {smiles2}"
    
    # Get MCS matches
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    if mcs_mol is None:
        return f"{smiles1} and {smiles2}"
    
    match1 = list(mol1.GetSubstructMatch(mcs_mol))
    match2 = list(mol2.GetSubstructMatch(mcs_mol))
    
    # Find difference atoms
    diff_atoms1 = [i for i in range(mol1.GetNumAtoms()) if i not in match1]
    diff_atoms2 = [i for i in range(mol2.GetNumAtoms()) if i not in match2]
    
    # Extract MCS SMILES from original molecules (preserving stereochemistry and explicit structure)
    # Try to get the canonical SMILES that preserves brackets
    mcs_frag1 = get_fragment_smiles(mol1, match1)
    mcs_frag2 = get_fragment_smiles(mol2, match2)
    
    # Use the original SMILES representation where possible
    # For carbonyl groups, ensure we keep C(=O) format
    mcs_smiles = mcs_frag1
    if '=' in mcs_smiles and '(' not in mcs_smiles:
        # Try to add explicit parentheses for clarity
        mcs_smiles = mcs_smiles.replace('C=O', 'C(=O)')
    
    # Group and format differences - need to maintain position order
    groups1 = group_consecutive_atoms(mol1, diff_atoms1)
    groups2 = group_consecutive_atoms(mol2, diff_atoms2)
    
    # Sort groups by their minimum atom index to maintain order
    groups1_with_pos = [(min(g), g) for g in groups1]
    groups1_with_pos.sort()
    
    groups2_with_pos = [(min(g), g) for g in groups2]
    groups2_with_pos.sort()
    
    parts1 = []
    for i, (pos, group) in enumerate(groups1_with_pos, 1):
        fg_str = format_difference_group(mol1, group)
        parts1.append((pos, f"{fg_str}:{i}"))
    
    parts2 = []
    for i, (pos, group) in enumerate(groups2_with_pos, 1):
        fg_str = format_difference_group(mol2, group)
        parts2.append((pos, f"{fg_str}:{i}"))
    
    # Interleave parts with MCS based on atom positions
    # Get position of MCS in molecule
    if match1:
        mcs_start1 = min(match1)
        mcs_end1 = max(match1)
    else:
        mcs_start1 = mcs_end1 = 0
    
    if match2:
        mcs_start2 = min(match2)
        mcs_end2 = max(match2)
    else:
        mcs_start2 = mcs_end2 = 0
    
    # Build output for mol1 side
    output1_parts = []
    for pos, part in parts1:
        if pos < mcs_start1:
            output1_parts.append((pos, part))
    output1_parts.append((mcs_start1, mcs_smiles))
    for pos, part in parts1:
        if pos > mcs_end1:
            output1_parts.append((pos, part))
    output1_parts.sort()
    
    # Build output for mol2 side
    output2_parts = []
    for pos, part in parts2:
        if pos < mcs_start2:
            output2_parts.append((pos, part))
    output2_parts.append((mcs_start2, mcs_smiles))
    for pos, part in parts2:
        if pos > mcs_end2:
            output2_parts.append((pos, part))
    output2_parts.sort()
    
    # Format final strings
    output_str1 = " ".join([p[1] for p in output1_parts])
    output_str2 = " ".join([p[1] for p in output2_parts])
    
    return f"{output_str1} and {output_str2}"


def main():
    parser = argparse.ArgumentParser(
        description='Compare two SMILES using MCS with functional group marking'
    )
    parser.add_argument('smiles1', help='First SMILES string')
    parser.add_argument('smiles2', help='Second SMILES string')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    print(f"Input: {args.smiles1} and {args.smiles2}")
    result = compare_smiles_with_mcs_detailed(args.smiles1, args.smiles2)
    print(f"Output: {result}")
    
    if args.verbose:
        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)
        
        mol1 = Chem.MolFromSmiles(args.smiles1)
        mol2 = Chem.MolFromSmiles(args.smiles2)
        
        if mol1 and mol2:
            print(f"\nMolecule 1: {args.smiles1}")
            print(f"  Atoms: {mol1.GetNumAtoms()}")
            print(f"  Bonds: {mol1.GetNumBonds()}")
            
            print(f"\nMolecule 2: {args.smiles2}")
            print(f"  Atoms: {mol2.GetNumAtoms()}")
            print(f"  Bonds: {mol2.GetNumBonds()}")
            
            mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                       timeout=5,
                                       bondCompare=rdFMCS.BondCompare.CompareAny,
                                       atomCompare=rdFMCS.AtomCompare.CompareElements)
            
            print(f"\nMCS:")
            print(f"  Atoms: {mcs_result.numAtoms}")
            print(f"  Bonds: {mcs_result.numBonds}")
            print(f"  SMARTS: {mcs_result.smartsString}")


if __name__ == '__main__':
    import sys
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Run with command line arguments
        main()
    else:
        # Run test cases
        print("="*80)
        print("TEST CASES")
        print("="*80)
        
        # Test 1
        smiles1_1 = "COC(=O)N"
        smiles2_1 = "COC(=O)C=C"
        print(f"\nTest 1:")
        print(f"input: {smiles1_1} and {smiles2_1}")
        result1 = compare_smiles_with_mcs_detailed(smiles1_1, smiles2_1)
        print(f"output: {result1}")
        print(f"expected: COC(=O) N_primary:1 and COC(=O) (C_alkene=C_alkene):1")
        
        # Test 2
        smiles1_2 = "NCOC(=O)N"
        smiles2_2 = "CCOC(=O)C=C"
        print(f"\nTest 2:")
        print(f"input: {smiles1_2} and {smiles2_2}")
        result2 = compare_smiles_with_mcs_detailed(smiles1_2, smiles2_2)
        print(f"output: {result2}")
        print(f"expected: N_primary:1 COC(=O) N_primary:2 and C_alkyl:1 COC(=O) (C_alkene=C_alkene):2")
        
        print("\n" + "="*80)
        print("To run with custom SMILES:")
        print("  python compare_mcs_with_fg_marking.py 'SMILES1' 'SMILES2'")
        print("  python compare_mcs_with_fg_marking.py 'SMILES1' 'SMILES2' --verbose")
        print("="*80)
