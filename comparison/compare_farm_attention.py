#!/usr/bin/env python3
"""
FARM-based Molecular Comparison using Attention Alignment
Minimal dependency on RDKit Mol objects - only for FG-enhanced representation
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import time
import sys
sys.path.append('src')

from rdkit import Chem
from helpers import get_new_smiles_rep

# Aliases
s2m = Chem.MolFromSmiles
m2s = Chem.MolToSmiles


class FARMAttentionComparator:
    """
    Compare molecules using FARM embeddings and attention-based alignment
    Minimal Mol dependency - only for FG-enhanced SMILES generation
    """
    
    def __init__(self, model_path='bert_model_output/checkpoint-1080'):
        print(f"Loading FARM BERT model from {model_path}...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        print("✓ FARM BERT model loaded")
    
    def smiles_to_farm_tokens(self, smiles):
        """
        Convert SMILES to FG-enhanced tokens
        Only Mol usage: get FG-enhanced representation
        """
        # Step 1: Convert to canonical SMILES
        mol = s2m(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        canonical_smiles = m2s(mol)
        
        # Step 2: Get FG-enhanced representation (only Mol usage)
        farm_smiles = get_new_smiles_rep(mol)
        
        # Step 3: Parse tokens (pure string processing)
        tokens = farm_smiles.split()
        
        return tokens, canonical_smiles
    
    def get_atom_embeddings(self, smiles):
        """
        Get atom-level embeddings from FARM (not average pooling)
        Better handling of tokenization mismatch
        """
        # Tokenize
        tokens, canonical = self.smiles_to_farm_tokens(smiles)
        
        # Convert to string for tokenizer
        farm_smiles_str = ' '.join(tokens)
        
        # Tokenize for BERT - get token to word mapping
        inputs = self.tokenizer(
            farm_smiles_str,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512,
            return_offsets_mapping=False
        )
        
        # Get word ids to map subword tokens to original tokens
        word_ids = inputs.word_ids()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state
            hidden_states = outputs.hidden_states[-1]  # (1, seq_len, 768)
            subword_embeddings = hidden_states[0]  # (seq_len, 768)
        
        # Map subword embeddings back to original tokens
        atom_embeddings = []
        for i in range(len(tokens)):
            # Find all subword tokens that belong to this original token
            subword_indices = [j for j, wid in enumerate(word_ids) if wid == i]
            
            if subword_indices:
                # Average the embeddings of subword tokens
                token_emb = subword_embeddings[subword_indices].mean(dim=0)
                atom_embeddings.append(token_emb)
            else:
                # Fallback: use mean of all embeddings
                token_emb = subword_embeddings.mean(dim=0)
                atom_embeddings.append(token_emb)
        
        atom_embeddings = torch.stack(atom_embeddings)  # (n_tokens, 768)
        
        return atom_embeddings, tokens, canonical
    
    def align_atoms_by_attention(self, emb1, emb2, tokens1, tokens2, threshold=0.65):
        """
        Align atoms using attention mechanism with improved matching
        Only align atoms of the same element type
        """
        # Compute attention matrix: (N1, N2)
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        
        # Cosine similarity matrix
        attention = torch.matmul(emb1_norm, emb2_norm.T)  # (N1, N2)
        
        # Define FG atoms (not special tokens/structure markers)
        structure_tokens = {'(', ')', '=', '#', '/', '\\', '[', ']', ':', '-', '+', 
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                           'c', 'n', 'o', 's', 'p'}  # Include lowercase aromatic atoms
        
        # Extract element type from token (first char before underscore)
        def get_element(token):
            if '_' not in token:
                return None
            base = token.split('_')[0]
            return base[0] if base else None
        
        # Use greedy bidirectional matching for FG atoms only
        aligned_pairs = []
        matched1 = set()
        matched2 = set()
        
        # First pass: Strong matches (high confidence) with same element
        for i in range(len(tokens1)):
            token1 = tokens1[i]
            # Only match FG-labeled atoms (contains underscore)
            if '_' not in token1:
                continue
            
            elem1 = get_element(token1)
            
            for j in range(len(tokens2)):
                token2 = tokens2[j]
                if '_' not in token2 or j in matched2:
                    continue
                
                elem2 = get_element(token2)
                
                # Only align same element types
                if elem1 != elem2:
                    continue
                
                score = attention[i, j].item()
                
                # Strong match: very similar tokens of same element
                if score > 0.85:  # High threshold for strong matches
                    aligned_pairs.append((i, j, score))
                    matched1.add(i)
                    matched2.add(j)
                    break
        
        # Second pass: Weaker matches for remaining FG atoms (same element only)
        for i in range(len(tokens1)):
            token1 = tokens1[i]
            if '_' not in token1 or i in matched1:
                continue
            
            elem1 = get_element(token1)
            
            # Find best unmatched partner of same element
            best_j = -1
            best_score = threshold
            
            for j in range(len(tokens2)):
                token2 = tokens2[j]
                if '_' not in token2 or j in matched2:
                    continue
                
                elem2 = get_element(token2)
                
                # Only align same element types
                if elem1 != elem2:
                    continue
                
                score = attention[i, j].item()
                if score > best_score:
                    best_score = score
                    best_j = j
            
            if best_j >= 0:
                aligned_pairs.append((i, best_j, best_score))
                matched1.add(i)
                matched2.add(best_j)
        
        # Find unmatched FG atoms (only atoms with FG labels)
        unmatched1 = [
            i for i in range(len(tokens1)) 
            if i not in matched1 and '_' in tokens1[i]
        ]
        
        unmatched2 = [
            j for j in range(len(tokens2))
            if j not in matched2 and '_' in tokens2[j]
        ]
        
        return aligned_pairs, unmatched1, unmatched2, attention
    
    def group_consecutive_atoms(self, unmatched, tokens):
        """
        Group consecutive unmatched atoms into functional groups
        Include special tokens (=, #) that are between unmatched atoms
        """
        if not unmatched:
            return []
        
        # Sort unmatched indices
        unmatched_set = set(unmatched)
        unmatched = sorted(unmatched)
        
        groups = []
        current_group = [unmatched[0]]
        
        for idx in unmatched[1:]:
            # Check if should group together
            # Include atoms that are close AND any special tokens between them
            gap = idx - current_group[-1]
            
            if gap <= 1:
                # Directly consecutive
                current_group.append(idx)
            elif gap <= 3:
                # Check if there are bond symbols (=, #) in between
                has_bond_symbol = False
                for i in range(current_group[-1] + 1, idx):
                    if i < len(tokens) and tokens[i] in {'=', '#'}:
                        has_bond_symbol = True
                        # Also add the bond symbol index
                        current_group.append(i)
                
                if has_bond_symbol or gap <= 2:
                    current_group.append(idx)
                else:
                    # Start new group
                    if current_group:
                        groups.append(sorted(current_group))
                    current_group = [idx]
            else:
                # Too far, start new group
                if current_group:
                    groups.append(sorted(current_group))
                current_group = [idx]
        
        if current_group:
            groups.append(sorted(current_group))
        
        return groups
    
    def format_diff_group(self, tokens, group_indices):
        """
        Format a group of diff atoms with FG labels
        Preserve special tokens like = for double bonds
        """
        # Get all tokens including special ones for bonds
        group_tokens = []
        for idx in group_indices:
            token = tokens[idx]
            # Include bond symbols (=, #) but skip parentheses
            if token not in {'(', ')', '[', ']', '/', '\\', ':', '-', '+'}:
                group_tokens.append(token)
        
        if not group_tokens:
            return ""
        
        # Check if it's a double/triple bond pattern (e.g., C_alkene = C_alkene)
        # Look for = or # in the group
        if '=' in group_tokens or '#' in group_tokens:
            # Format with bond symbol
            formatted_parts = []
            for token in group_tokens:
                if token in {'=', '#'}:
                    formatted_parts.append(token)
                else:
                    formatted_parts.append(token)
            result = ''.join(formatted_parts)
            # Wrap in parentheses
            return f"({result})"
        else:
            # Regular functional group
            # Remove consecutive duplicates for cleaner output
            unique_labels = []
            for token in group_tokens:
                if not unique_labels or token != unique_labels[-1]:
                    unique_labels.append(token)
            
            # Format
            if len(unique_labels) == 1:
                return unique_labels[0]
            else:
                return f"({'_'.join(unique_labels)})"
    
    def extract_mcs_string(self, tokens1, tokens2, aligned_pairs, unmatched1, unmatched2):
        """
        Extract MCS representation from aligned tokens
        Skip substituent groups that contain unmatched atoms
        """
        if not tokens1 or not tokens2:
            return ""
        
        # Create sets
        unmatched1_set = set(unmatched1)
        unmatched2_set = set(unmatched2)
        align_map = {pair[0]: pair[1] for pair in aligned_pairs}
        
        # Build MCS by walking through tokens
        # Skip entire ( ) groups that contain unmatched atoms
        mcs_parts = []
        i = 0
        skip_until_close = 0  # Depth of parentheses to skip
        
        while i < len(tokens1):
            token = tokens1[i]
            
            # Handle parentheses
            if token == '(':
                # Look ahead to see if this group contains unmatched atoms
                paren_depth = 1
                j = i + 1
                has_unmatched = False
                
                while j < len(tokens1) and paren_depth > 0:
                    if tokens1[j] == '(':
                        paren_depth += 1
                    elif tokens1[j] == ')':
                        paren_depth -= 1
                    elif '_' in tokens1[j] and j in unmatched1_set:
                        has_unmatched = True
                        break
                    j += 1
                
                if has_unmatched:
                    # Skip this entire group
                    skip_until_close = 1
                    i += 1
                    continue
                else:
                    # Include this group
                    mcs_parts.append(token)
            
            elif token == ')':
                if skip_until_close > 0:
                    skip_until_close -= 1
                    i += 1
                    continue
                else:
                    mcs_parts.append(token)
            
            elif skip_until_close > 0:
                # Inside skipped group
                if token == '(':
                    skip_until_close += 1
                elif token == ')':
                    skip_until_close -= 1
                i += 1
                continue
            
            elif '_' in token:
                # FG atom
                if i in unmatched1_set:
                    # Skip unmatched atom
                    i += 1
                    continue
                else:
                    # Include matched atom (without FG suffix)
                    base_token = token.split('_')[0]
                    mcs_parts.append(base_token)
            
            else:
                # Structure token
                mcs_parts.append(token)
            
            i += 1
        
        # Build MCS string
        mcs_string = ''.join(mcs_parts)
        
        return mcs_string
    
    def compare_smiles(self, smiles1, smiles2, threshold=0.7, verbose=False):
        """
        Compare two SMILES using FARM attention-based alignment
        """
        start_time = time.time()
        
        # Step 1: Get atom embeddings
        embedding_start = time.time()
        emb1, tokens1, canonical1 = self.get_atom_embeddings(smiles1)
        emb2, tokens2, canonical2 = self.get_atom_embeddings(smiles2)
        embedding_time = time.time() - embedding_start
        
        # Step 2: Align atoms using attention
        alignment_start = time.time()
        aligned_pairs, unmatched1, unmatched2, attention_matrix = self.align_atoms_by_attention(
            emb1, emb2, tokens1, tokens2, threshold=threshold
        )
        alignment_time = time.time() - alignment_start
        
        # Step 3: Group unmatched atoms
        diff_groups1 = self.group_consecutive_atoms(unmatched1, tokens1)
        diff_groups2 = self.group_consecutive_atoms(unmatched2, tokens2)
        
        # Step 4: Extract MCS
        mcs_string = self.extract_mcs_string(tokens1, tokens2, aligned_pairs, unmatched1, unmatched2)
        
        # Step 5: Format diff groups with serial numbers
        serial_parts1 = []
        for i, group in enumerate(diff_groups1, 1):
            fg_label = self.format_diff_group(tokens1, group)
            if fg_label:
                serial_parts1.append((min(group), fg_label, i))
        
        serial_parts2 = []
        for i, group in enumerate(diff_groups2, 1):
            fg_label = self.format_diff_group(tokens2, group)
            if fg_label:
                serial_parts2.append((min(group), fg_label, i))
        
        # Step 6: Build output string
        # Determine MCS position
        mcs_pos1 = min([pair[0] for pair in aligned_pairs]) if aligned_pairs else 0
        mcs_pos2 = min([pair[1] for pair in aligned_pairs]) if aligned_pairs else 0
        
        # Build mol1 output
        output1_parts = []
        for pos, fg, serial in serial_parts1:
            output1_parts.append((pos, f"{fg}:{serial}"))
        output1_parts.append((mcs_pos1, mcs_string))
        output1_parts.sort()
        
        # Build mol2 output
        output2_parts = []
        for pos, fg, serial in serial_parts2:
            output2_parts.append((pos, f"{fg}:{serial}"))
        output2_parts.append((mcs_pos2, mcs_string))
        output2_parts.sort()
        
        # Format final output
        output_str1 = " ".join([p[1] for p in output1_parts])
        output_str2 = " ".join([p[1] for p in output2_parts])
        result = f"{output_str1} and {output_str2}"
        
        # Step 7: Calculate similarity
        mol_emb1 = emb1.mean(dim=0, keepdim=True)
        mol_emb2 = emb2.mean(dim=0, keepdim=True)
        similarity = cosine_similarity(
            mol_emb1.cpu().numpy(),
            mol_emb2.cpu().numpy()
        )[0][0]
        
        # Metrics
        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'embedding_time': embedding_time,
            'alignment_time': alignment_time,
            'cosine_similarity': similarity,
            'aligned_atoms': len(aligned_pairs),
            'diff_groups_mol1': len(serial_parts1),
            'diff_groups_mol2': len(serial_parts2),
            'attention_threshold': threshold,
            'canonical1': canonical1,
            'canonical2': canonical2
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FARM ATTENTION-BASED COMPARISON")
            print(f"{'='*80}")
            print(f"Input 1: {smiles1}")
            print(f"Canonical 1: {canonical1}")
            print(f"  Tokens: {len(tokens1)}")
            print(f"  FARM representation: {' '.join(tokens1)}")
            print(f"\nInput 2: {smiles2}")
            print(f"Canonical 2: {canonical2}")
            print(f"  Tokens: {len(tokens2)}")
            print(f"  FARM representation: {' '.join(tokens2)}")
            print(f"\nAlignment:")
            print(f"  Aligned atoms: {len(aligned_pairs)}")
            print(f"  Unmatched mol1: {len(unmatched1)}")
            print(f"  Unmatched mol2: {len(unmatched2)}")
            print(f"  Threshold: {threshold}")
            print(f"\nMCS (from alignment): {mcs_string}")
            print(f"\nPerformance:")
            print(f"  Total time: {total_time*1000:.2f} ms")
            print(f"  Embedding time: {embedding_time*1000:.2f} ms")
            print(f"  Alignment time: {alignment_time*1000:.2f} ms")
            print(f"  Cosine similarity: {similarity:.4f}")
        
        return result, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compare SMILES using FARM attention-based alignment'
    )
    parser.add_argument('--smiles1', help='First SMILES string')
    parser.add_argument('--smiles2', help='Second SMILES string')
    parser.add_argument('--model', default='bert_model_output/checkpoint-1080',
                       help='Path to FARM BERT model')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Attention threshold for alignment (default: 0.7)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = FARMAttentionComparator(model_path=args.model)
    
    if args.smiles1 and args.smiles2:
        result, metrics = comparator.compare_smiles(
            args.smiles1, args.smiles2, 
            threshold=args.threshold,
            verbose=args.verbose
        )
        
        print(f"\nInput 1: {args.smiles1}")
        print(f"Input 2: {args.smiles2}")
        if metrics:
            print(f"\nCanonical 1: {metrics['canonical1']}")
            print(f"Canonical 2: {metrics['canonical2']}")
        print(f"\nOutput: {result}")
        
        if not args.verbose and metrics:
            print(f"\nMetrics:")
            print(f"  Aligned atoms: {metrics['aligned_atoms']}")
            print(f"  Diff groups: Mol1={metrics['diff_groups_mol1']}, Mol2={metrics['diff_groups_mol2']}")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  Time: {metrics['total_time']*1000:.2f} ms")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Test examples
        print("="*80)
        print("FARM ATTENTION-BASED COMPARISON - TEST CASES")
        print("="*80)
        
        comparator = FARMAttentionComparator(model_path='bert_model_output/checkpoint-1080')
        
        test_cases = [
            ("COC(=O)N", "COC(=O)C=C", "Ester vs Ester with alkene"),
            ("CCOC(=O)CCC(=O)O", "CCNC(=O)CCC(=O)O", "Ester vs Amide"),
            ("ClCCCCCBr", "FCCCCCCl", "Halogen substitution"),
            ("COc1ccc(CC(=O)O)cc1", "CNc1ccc(CC(=O)Cl)cc1", "Aromatic complex")
        ]
        
        for i, (smiles1, smiles2, description) in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}: {description}")
            print(f"{'='*80}")
            
            try:
                result, metrics = comparator.compare_smiles(smiles1, smiles2, verbose=True)
                print(f"\nInput (original): {smiles1} and {smiles2}")
                if metrics:
                    print(f"Input (canonical): {metrics['canonical1']} and {metrics['canonical2']}")
                print(f"Output: {result}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\n{'='*80}")
        print("USAGE:")
        print("  python compare_farm_attention.py --smiles1 'SMILES1' --smiles2 'SMILES2'")
        print("  python compare_farm_attention.py --smiles1 'SMILES1' --smiles2 'SMILES2' --verbose")
        print("  python compare_farm_attention.py --smiles1 'SMILES1' --smiles2 'SMILES2' --threshold 0.8")
        print("="*80)
