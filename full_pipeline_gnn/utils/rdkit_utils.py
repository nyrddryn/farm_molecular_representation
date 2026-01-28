"""
RDKit utility functions

Minimal RDKit wrapper for SMILES handling
"""

from rdkit import Chem

# Aliases for cleaner code
s2m = Chem.MolFromSmiles
m2s = Chem.MolToSmiles
