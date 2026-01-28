#!/usr/bin/env python3
"""Debug script to understand MCS matching issue"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
import sys
sys.path.append('src')
from helpers import detect_functional_group

def analyze_molecule(smiles, name):
    """Analyze molecule structure and MCS matching"""
    print(f"\n{'='*60}")
    print(f"{name}: {smiles}")
    print(f"{'='*60}")
    
    mol = Chem.MolFromSmiles(smiles)
    detect_functional_group(mol)
    
    print(f"Atoms ({mol.GetNumAtoms()}):")
    for i, atom in enumerate(mol.GetAtoms()):
        fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
        symbol = atom.GetSymbol()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        print(f"  {i}: {symbol} (FG={fg}) - neighbors: {neighbors}")
    
    print(f"\nBonds ({mol.GetNumBonds()}):")
    for bond in mol.GetBonds():
        print(f"  {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: {bond.GetBondType()}")
    
    return mol

def analyze_mcs(mol1, mol2, smiles1, smiles2):
    """Analyze MCS matching"""
    print(f"\n{'='*60}")
    print("MCS ANALYSIS")
    print(f"{'='*60}")
    
    mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                timeout=5,
                                bondCompare=rdFMCS.BondCompare.CompareAny,
                                atomCompare=rdFMCS.AtomCompare.CompareElements)
    
    print(f"MCS SMARTS: {mcs_result.smartsString}")
    print(f"MCS Atoms: {mcs_result.numAtoms}, Bonds: {mcs_result.numBonds}")
    
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    matches1 = mol1.GetSubstructMatches(mcs_mol)
    matches2 = mol2.GetSubstructMatches(mcs_mol)
    
    print(f"\nMol1 matches ({len(matches1)}):")
    for i, match in enumerate(matches1):
        print(f"  Match {i}: {list(match)}")
    
    print(f"\nMol2 matches ({len(matches2)}):")
    for i, match in enumerate(matches2):
        print(f"  Match {i}: {list(match)}")
    
    # Select best match
    def score_match(match):
        if not match:
            return float('inf')
        sorted_match = sorted(match)
        gaps = sum(1 for i in range(len(sorted_match)-1) if sorted_match[i+1] - sorted_match[i] > 1)
        return (min(match), gaps)
    
    match1 = list(min(matches1, key=score_match)) if matches1 else []
    match2 = list(min(matches2, key=score_match)) if matches2 else []
    
    print(f"\nSelected matches:")
    print(f"  Mol1: {match1}")
    print(f"  Mol2: {match2}")
    
    # Find diff atoms
    diff1 = [i for i in range(mol1.GetNumAtoms()) if i not in match1]
    diff2 = [i for i in range(mol2.GetNumAtoms()) if i not in match2]
    
    print(f"\nDiff atoms:")
    print(f"  Mol1: {diff1}")
    print(f"  Mol2: {diff2}")
    
    # Check connectivity of diff atoms
    print(f"\nDiff atom connectivity (Mol1):")
    for idx in diff1:
        atom = mol1.GetAtomWithIdx(idx)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        neighbors_in_diff = [n for n in neighbors if n in diff1]
        neighbors_in_mcs = [n for n in neighbors if n in match1]
        fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
        print(f"  Atom {idx} ({atom.GetSymbol()}, FG={fg}):")
        print(f"    All neighbors: {neighbors}")
        print(f"    Diff neighbors: {neighbors_in_diff}")
        print(f"    MCS neighbors: {neighbors_in_mcs}")
    
    print(f"\nDiff atom connectivity (Mol2):")
    for idx in diff2:
        atom = mol2.GetAtomWithIdx(idx)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        neighbors_in_diff = [n for n in neighbors if n in diff2]
        neighbors_in_mcs = [n for n in neighbors if n in match2]
        fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
        print(f"  Atom {idx} ({atom.GetSymbol()}, FG={fg}):")
        print(f"    All neighbors: {neighbors}")
        print(f"    Diff neighbors: {neighbors_in_diff}")
        print(f"    MCS neighbors: {neighbors_in_mcs}")

if __name__ == '__main__':
    smiles1 = "CCOC(=O)CCC(=O)O"
    smiles2 = "CCNC(=O)CCC(=O)O"
    
    mol1 = analyze_molecule(smiles1, "Molecule 1")
    mol2 = analyze_molecule(smiles2, "Molecule 2")
    
    analyze_mcs(mol1, mol2, smiles1, smiles2)
