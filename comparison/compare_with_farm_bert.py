"""
Compare Two SMILES using FARM BERT Model Embeddings and MCS
Pipeline:
1. Convert SMILES to FARM FG-enhanced SMILES representation
2. Use FARM BERT model to get embeddings for each molecule
3. Find the Maximum Common Substructure (MCS) between two SMILES
4. Mark differences using FARM's functional group annotations
5. Output format with serial numbers tracking positions
"""

import argparse
import sys
import os
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from rdkit import Chem
from rdkit.Chem import rdFMCS, MolFromSmiles as s2m
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from helpers import get_new_smiles_rep, detect_functional_group


class FARMComparator:
    def __init__(self, model_path='bert_model_output/checkpoint-1080', tokenizer_path=None):
        """
        Initialize FARM model and tokenizer
        
        Args:
            model_path: Path to FARM BERT model checkpoint
            tokenizer_path: Path to tokenizer (if None, uses model_path)
        """
        print(f"Loading FARM model from {model_path}...")
        
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            self.model = BertForMaskedLM.from_pretrained(model_path)
            self.model.eval()
            print("✓ FARM model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model path is correct")
            raise
    
    def smiles_to_farm_representation(self, smiles):
        """Convert SMILES to FARM FG-enhanced representation"""
        mol = s2m(smiles)
        if mol is None:
            return None
        
        farm_smiles = get_new_smiles_rep(mol)
        return farm_smiles
    
    def get_farm_embeddings(self, smiles):
        """
        Get FARM embeddings for a SMILES string
        
        Returns:
            - farm_smiles: FG-enhanced SMILES representation
            - embeddings: Atom-level embeddings from FARM BERT
            - tokens: Token list
        """
        farm_smiles = self.smiles_to_farm_representation(smiles)
        if farm_smiles is None:
            return None, None, None
        
        # Tokenize
        inputs = self.tokenizer(farm_smiles, return_tensors='pt')
        
        # Get embeddings from FARM BERT
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden states (atom-level embeddings)
            last_hidden_states = outputs.hidden_states[-1][0]  # Shape: (N, 768)
        
        # Get tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return farm_smiles, last_hidden_states.numpy(), tokens
    
    def compare_embeddings(self, smiles1, smiles2):
        """
        Compare two SMILES using FARM embeddings
        """
        print(f"\n{'='*80}")
        print(f"COMPARING SMILES USING FARM BERT MODEL")
        print(f"{'='*80}")
        
        print(f"\nMolecule 1: {smiles1}")
        farm1, emb1, tokens1 = self.get_farm_embeddings(smiles1)
        if farm1 is None:
            print("Error: Invalid SMILES 1")
            return
        print(f"  FARM representation: {farm1}")
        print(f"  Tokens: {len(tokens1)}, Embedding shape: {emb1.shape}")
        
        print(f"\nMolecule 2: {smiles2}")
        farm2, emb2, tokens2 = self.get_farm_embeddings(smiles2)
        if farm2 is None:
            print("Error: Invalid SMILES 2")
            return
        print(f"  FARM representation: {farm2}")
        print(f"  Tokens: {len(tokens2)}, Embedding shape: {emb2.shape}")
        
        # Calculate embedding similarity
        # Average pooling for molecular-level representation
        mol_emb1 = np.mean(emb1, axis=0)
        mol_emb2 = np.mean(emb2, axis=0)
        
        # Cosine similarity
        similarity = np.dot(mol_emb1, mol_emb2) / (np.linalg.norm(mol_emb1) * np.linalg.norm(mol_emb2))
        
        print(f"\n📊 FARM EMBEDDING ANALYSIS:")
        print(f"  Molecular embedding similarity (cosine): {similarity:.4f}")
        
        # Find MCS
        mol1 = s2m(smiles1)
        mol2 = s2m(smiles2)
        
        mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                    timeout=5,
                                    bondCompare=rdFMCS.BondCompare.CompareAny,
                                    atomCompare=rdFMCS.AtomCompare.CompareElements)
        
        print(f"\n🔍 MAXIMUM COMMON SUBSTRUCTURE (MCS):")
        print(f"  Common atoms: {mcs_result.numAtoms}")
        print(f"  Common bonds: {mcs_result.numBonds}")
        print(f"  SMARTS: {mcs_result.smartsString}")
        
        return {
            'smiles1': smiles1,
            'smiles2': smiles2,
            'farm1': farm1,
            'farm2': farm2,
            'embeddings1': emb1,
            'embeddings2': emb2,
            'tokens1': tokens1,
            'tokens2': tokens2,
            'similarity': similarity,
            'mcs': mcs_result
        }
    
    def detect_functional_groups(self, smiles):
        """Detect functional groups in a SMILES using FARM"""
        print(f"\n{'='*80}")
        print(f"FUNCTIONAL GROUP DETECTION: {smiles}")
        print(f"{'='*80}")
        
        mol = s2m(smiles)
        if mol is None:
            print("Error: Invalid SMILES")
            return
        
        # Apply FARM functional group detection
        detect_functional_group(mol)
        
        # Get FARM representation
        farm_smiles = get_new_smiles_rep(mol)
        print(f"\nFARM FG-enhanced representation:")
        print(f"  {farm_smiles}")
        
        # Get embeddings
        _, embeddings, tokens = self.get_farm_embeddings(smiles)
        
        print(f"\nFARM BERT embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Tokens: {tokens}")
        
        # Collect functional groups
        from collections import Counter
        functional_groups = []
        
        for atom in mol.GetAtoms():
            fg = atom.GetProp('FG') if atom.HasProp('FG') else ''
            if fg:
                functional_groups.append(fg)
        
        fg_counts = Counter(functional_groups)
        
        print(f"\n🔬 Functional Groups Detected:")
        if fg_counts:
            for fg, count in sorted(fg_counts.items()):
                print(f"  - {fg}: {count} occurrence(s)")
        else:
            print(f"  No specific functional groups detected")


def main():
    parser = argparse.ArgumentParser(
        description='Compare SMILES using FARM BERT model'
    )
    parser.add_argument('--smiles1', help='First SMILES string')
    parser.add_argument('--smiles2', help='Second SMILES string')
    parser.add_argument('--detect', help='Detect functional groups in a single SMILES')
    parser.add_argument('--model', default='bert_model_output/checkpoint-1080',
                       help='Path to FARM model checkpoint')
    parser.add_argument('--tokenizer', default=None,
                       help='Path to tokenizer (default: same as model)')
    
    args = parser.parse_args()
    
    # Initialize FARM comparator
    try:
        comparator = FARMComparator(model_path=args.model, tokenizer_path=args.tokenizer)
    except Exception as e:
        print(f"\nFailed to load FARM model. Please check the model path.")
        print(f"Available checkpoints in bert_model_output/:")
        import os
        if os.path.exists('bert_model_output'):
            for item in os.listdir('bert_model_output'):
                print(f"  - {item}")
        return
    
    if args.detect:
        # Detect functional groups in single SMILES
        comparator.detect_functional_groups(args.detect)
    
    elif args.smiles1 and args.smiles2:
        # Compare two SMILES
        comparator.compare_embeddings(args.smiles1, args.smiles2)
    
    else:
        print("Please provide either:")
        print("  --smiles1 and --smiles2 for comparison")
        print("  --detect SMILES for functional group detection")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Run test examples
        print("="*80)
        print("FARM BERT MODEL - TEST EXAMPLES")
        print("="*80)
        
        try:
            comparator = FARMComparator(model_path='bert_model_output/checkpoint-1080')
            
            print("\n" + "="*80)
            print("TEST 1: Functional Group Detection")
            print("="*80)
            comparator.detect_functional_groups("COC(=O)N")
            
            print("\n" + "="*80)
            print("TEST 2: SMILES Comparison")
            print("="*80)
            comparator.compare_embeddings("COC(=O)N", "COC(=O)C=C")
            
            print("\n" + "="*80)
            print("TEST 3: Complex Molecule")
            print("="*80)
            comparator.detect_functional_groups("NNc1nncc2ccccc12")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("\nTo run with custom SMILES:")
            print("  python compare_with_farm_bert.py --detect 'SMILES'")
            print("  python compare_with_farm_bert.py --smiles1 'SMILES1' --smiles2 'SMILES2'")
            print("  python compare_with_farm_bert.py --smiles1 'SMILES1' --smiles2 'SMILES2' --model bert_model_output/checkpoint-1080")
