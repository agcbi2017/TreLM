import os
import sys
import json
import argparse


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
    parser.add_argument("--input_para_file", type=str, default="", help="Input Parallel File Path")
    parser.add_argument("--input_align_file", type=str, default="", help="Input Alignment File Path")
    parser.add_argument("--left_lang", type=str, default="en", help="Left Language for parallel file")
    parser.add_argument("--right_lang", type=str, default="zh", help="Right Language for parallel file")
    parser.add_argument("--jsonl_out_path", type=str, default="", help="Bert Embedding Output Path")
    parser.add_argument("--reverse", type=bool_flag, default=False, help="do_lower_case")
    args = parser.parse_args()

    with open(args.input_para_file, 'r', encoding='utf-8') as fpara:
        with open(args.input_align_file, 'r', encoding='utf-8') as falign:
            with open(args.jsonl_out_path, 'w', encoding='utf-8') as fjsonl:
                for para_line in fpara:
                    para_line = para_line.strip()
                    align_line = falign.readline().strip()
                    if len(para_line) > 0:
                        para_line = para_line.split(' ||| ')
                        assert len(para_line) == 2
                        src_sent = para_line[0].split(' ')
                        tgt_sent = para_line[1].split(' ')
                        aligns = [item.split('-') for item in align_line.split(' ')]
                        if args.reverse:
                            tgt_sent += ['[PAD]'] # to make it compatible with the one not aligned
                            align_index = [len(tgt_sent)-1 for _ in range(len(src_sent))] # set the default index is the last one special token
                            for item in aligns:
                                align_index[int(item[0])] = int(item[1])
                            jsonl_obj = {'src': tgt_sent, 'src_lang': args.right_lang, 'tgt': src_sent, 'tgt_lang': args.left_lang, 'align_index': align_index}
                        else:
                            src_sent += ['[PAD]'] # to make it compatible with the one not aligned
                            align_index = [len(src_sent)-1 for _ in range(len(tgt_sent))] # set the default index is the last one special token
                            for item in aligns:
                                align_index[int(item[1])] = int(item[0])
                            jsonl_obj = {'src': src_sent, 'src_lang': args.left_lang, 'tgt': tgt_sent, 'tgt_lang': args.right_lang, 'align_index': align_index}
                        fjsonl.write(json.dumps(jsonl_obj, ensure_ascii=False)+'\n')