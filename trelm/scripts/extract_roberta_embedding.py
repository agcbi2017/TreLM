
import torch
import argparse
from transformers import RobertaForMaskedLM, RobertaTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--model_path", type=str, default="", help="Pretrained Roberta Model Path")
    parser.add_argument("--emb_out_path", type=str, default="", help="Roberta Embedding Output Path")
    args = parser.parse_args()

    roberta_model = RobertaForMaskedLM.from_pretrained(args.model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    vocab_size, emb_dim = roberta_model.roberta.embeddings.word_embeddings.weight.size()

    assert vocab_size == tokenizer.vocab_size

    vectors = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)

    dico = []
    for idx in range(vocab_size):
        if idx == tokenizer.bos_token_id or idx == tokenizer.cls_token_id:
            dico.append('<s>')
        elif idx == tokenizer.eos_token_id or idx == tokenizer.sep_token_id:
            dico.append('</s>')
        elif idx == tokenizer.unk_token_id:
            dico.append('<unk>')
        elif idx == tokenizer.pad_token_id:
            dico.append('<pad>')
        elif idx == tokenizer.mask_token_id:
            dico.append('<mask>')
        else:
            dico.append('ROBERTAVOCAB-'+str(idx))
        vectors[idx] = roberta_model.roberta.embeddings.word_embeddings.weight[idx]

    torch.save({'dico': dico, 'vectors': vectors}, args.emb_out_path)