import sys
sys.path.append('src')
from rdkit import Chem
from rdkit.Chem import rdFMCS
from helpers import detect_functional_group

smiles1 = "COc1ccc(CC(=O)O)cc1"
smiles2 = "CNc1ccc(CC(=O)Cl)cc1"

mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)
detect_functional_group(mol1)
detect_functional_group(mol2)

print("Mol1 atoms:")
for i, atom in enumerate(mol1.GetAtoms()):
    fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    print(f"  {i}: {atom.GetSymbol()} (FG={fg}) - neighbors: {neighbors}")

print("\nMol2 atoms:")
for i, atom in enumerate(mol2.GetAtoms()):
    fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    print(f"  {i}: {atom.GetSymbol()} (FG={fg}) - neighbors: {neighbors}")

mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=5,
                            bondCompare=rdFMCS.BondCompare.CompareAny,
                            atomCompare=rdFMCS.AtomCompare.CompareElements)

print(f"\nMCS SMARTS: {mcs_result.smartsString}")
print(f"MCS: {mcs_result.numAtoms} atoms, {mcs_result.numBonds} bonds")

mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
match1 = list(mol1.GetSubstructMatches(mcs_mol)[0])
match2 = list(mol2.GetSubstructMatches(mcs_mol)[0])

print(f"\nMol1 MCS match: {match1}")
print(f"Mol2 MCS match: {match2}")

diff1 = [i for i in range(mol1.GetNumAtoms()) if i not in match1]
diff2 = [i for i in range(mol2.GetNumAtoms()) if i not in match2]

print(f"\nMol1 diff atoms: {diff1}")
for idx in diff1:
    atom = mol1.GetAtomWithIdx(idx)
    fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    print(f"  {idx}: {atom.GetSymbol()} (FG={fg}) - neighbors: {neighbors}")

print(f"\nMol2 diff atoms: {diff2}")
for idx in diff2:
    atom = mol2.GetAtomWithIdx(idx)
    fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    print(f"  {idx}: {atom.GetSymbol()} (FG={fg}) - neighbors: {neighbors}")
