import argparse
import pickle
import os
import tempfile
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

def main(corpus_path, save_path, vocab_size=30000, min_frequency=5):
    """
    Main function to create and save a tokenizer from the provided corpus.

    Args:
        corpus_path (str): Path to the text corpus for training the tokenizer (.txt or .pkl).
        save_path (str): Directory to save the trained tokenizer.
        vocab_size (int): Size of the vocabulary to be created.
        min_frequency (int): Minimum frequency for tokens to be included in the vocabulary.
    """
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))  # Initialize WordLevel tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(' ')  # Use character delimiter for splitting
    
    # Define special tokens
    special_tokens = ["[UNK]", "[PAD]", "[MASK]"]
    
    # Train the tokenizer
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens
    )
    
    # Check if corpus is .pkl or .txt
    if corpus_path.endswith('.pkl'):
        print(f"Loading pickle corpus from {corpus_path}...")
        with open(corpus_path, 'rb') as f:
            corpus_list = pickle.load(f)
        print(f"Loaded {len(corpus_list)} FG-enhanced SMILES")
        
        # Create a temporary text file for training
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp_path = temp_file.name
        
        print(f"Writing corpus to temporary text file...")
        for smiles in corpus_list:
            temp_file.write(smiles + '\n')
        temp_file.close()
        
        print(f"Training tokenizer...")
        tokenizer.train([temp_path], trainer=trainer)
        
        # Clean up temporary file
        os.unlink(temp_path)
    else:
        # Assume it's a text file
        print(f"Training tokenizer from text file: {corpus_path}")
        tokenizer.train([corpus_path], trainer=trainer)
    
    # Save the tokenizer to a file
    tokenizer.save("tokenizer.json")
    
    # Load the tokenizer from the saved file
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # Wrap the tokenizer for use with the Transformers library
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        tokenizer_file="tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=True
    )

    # Save the wrapped tokenizer
    wrapped_tokenizer.save_pretrained(save_path)
    
    # Print the vocabulary size
    print('Vocabulary size:', tokenizer.get_vocab_size())

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and save a tokenizer from a text corpus.')
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the text corpus for tokenizer training.')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the trained tokenizer.')
    parser.add_argument('--vocab_size', type=int, default=30000, help='Size of the vocabulary to create.')
    parser.add_argument('--min_frequency', type=int, default=5, help='Minimum frequency for tokens to be included in the vocabulary.')

    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.corpus_path, args.save_path, args.vocab_size, args.min_frequency)
