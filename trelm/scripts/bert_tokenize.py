import argparse
from tokenizers import BertWordPieceTokenizer

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--model_path", type=str, default="", help="Pretrained tokenizer path")
    parser.add_argument("--do_lower_case", type=bool_flag, default=False, help="do_lower_case")
    parser.add_argument("--input_file", type=str, default="", help="Input file to be tokenized")
    parser.add_argument("--output_file", type=str, default="", help="Output tokenized file")
    args = parser.parse_args()

    tokenizer = BertWordPieceTokenizer(args.model_path, lowercase=args.do_lower_case)
    count = 0
    with open(args.input_file, 'r', encoding='utf-8') as fin:
        with open(args.output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                if len(line.strip())>0:
                    output = tokenizer.encode(line.strip())
                    fout.write(' '.join(output.tokens)+'\n')
                else:
                    fout.write('\n')
                count += 1
                if count % 1000 == 0:
                    fout.flush()
                    print('%d sentences tokenized!'%count)