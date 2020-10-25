import sys
import argparse

from tokenizers import BertWordPieceTokenizer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='train wordpiece tokenizer')
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--vocab_size", type=int, default="40000")
    parser.add_argument("--limit_alphabet", type=int, default="1000") # 30000 for chinese and similar languages
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=False)

    tokenizer.train(
        files=[args.input_file], 
        vocab_size=args.vocab_size, 
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=args.limit_alphabet,
        wordpieces_prefix="##",
    )

    # Save files to disk
    tokenizer.save_model(args.output_path)