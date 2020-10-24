
import torch
import argparse
from transformers import ElectraForMaskedLM, ElectraTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--model_path", type=str, default="", help="Pretrained Electra Model Path")
    parser.add_argument("--emb_out_path", type=str, default="", help="Electra Embedding Output Path")
    args = parser.parse_args()

    electra_model = ElectraForMaskedLM.from_pretrained(args.model_path)
    tokenizer = ElectraTokenizer.from_pretrained(args.model_path)

    vocab_size, emb_dim = electra_model.electra.embeddings.word_embeddings.weight.size()

    assert vocab_size == tokenizer.vocab_size

    vectors = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)

    dico = []
    for idx in range(vocab_size):
        token = tokenizer.ids_to_tokens.get(idx)
        assert token is not None
        dico.append(token)
        vectors[idx] = electra_model.electra.embeddings.word_embeddings.weight[idx]

    torch.save({'dico': dico, 'vectors': vectors}, args.emb_out_path)