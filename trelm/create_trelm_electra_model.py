import logging
import argparse
import collections
import os
import shutil

import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from trelm_electra import TrelmElectraForMaskedLM

from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fix_electra_model_for_trelm(electra_model_path, trelm_model_path):
    electra_model = torch.load(electra_model_path)
    trelm_model = collections.OrderedDict()
    for k, v in electra_model.items():
        trelm_model[k.replace('electra.', 'trelm_electra.')] = v
    torch.save(trelm_model, trelm_model_path)


def create_trelm_electra_model(pretrained_model_path, vocab_path, do_lower_case, vocab_emb_path, vocab_emb_type, save_model_to, langid_list):
    
    tokenizer = BertTokenizerFast(vocab_path, do_lower_case=do_lower_case)

    vocab_emb_weights = None
    if vocab_emb_type == 'pth':
        vocab_emb_data = torch.load(vocab_emb_path)
        vocab_emb_weights = vocab_emb_data['vectors']
        assert tokenizer.vocab_size == vocab_emb_weights.size(0)
    elif vocab_emb_type == 'word2vec':
        wv_model = KeyedVectors.load_word2vec_format(vocab_emb_path)
        vocab_emb_weights = torch.FloatTensor(wv_model.vectors)
        assert tokenizer.vocab_size == vocab_emb_weights.size(0)

    model = TrelmElectraForMaskedLM.from_pretrained(pretrained_model_path)
    
    if vocab_emb_weights is not None:
        assert model.config.embedding_size == vocab_emb_weights.size(1)

    # set the hyperparameters
    model.config.vocab_size = tokenizer.vocab_size
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.model_type = 'trelm_electra'
    model.config.architectures = ['TrelmElectraForMaskedLM']
    model.config.n_langs = 2
    model.config.langs_to_id = {langid: idx for idx, langid in enumerate(langid_list)}

    # initial the word embeddings
    model.trelm_electra.embeddings.word_embeddings = nn.Embedding(tokenizer.vocab_size, model.config.embedding_size, padding_idx=model.config.pad_token_id)
    if vocab_emb_weights is not None:
        model.trelm_electra.embeddings.word_embeddings.weight.data.copy_(vocab_emb_weights)
    else:
        logger.info('word_embeddings random initialized!')
        model.trelm_electra.embeddings.word_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    # reset generator_predictions generator_lm_head
    delattr(model, "generator_predictions")
    delattr(model, "generator_lm_head")

    # initial lang embeddings?

    # initial the tlayer
    layer = model.trelm_electra.encoder.layer[int(model.config.num_hidden_layers / 2)]

    model.trelm_electra.encoder.tlayer.attention.self.query.weight = layer.attention.self.query.weight
    model.trelm_electra.encoder.tlayer.attention.self.query.bias = layer.attention.self.query.bias
    model.trelm_electra.encoder.tlayer.attention.self.key.weight = layer.attention.self.key.weight
    model.trelm_electra.encoder.tlayer.attention.self.key.bias = layer.attention.self.key.bias
    model.trelm_electra.encoder.tlayer.attention.self.value.weight = layer.attention.self.value.weight
    model.trelm_electra.encoder.tlayer.attention.self.value.bias = layer.attention.self.value.bias
    model.trelm_electra.encoder.tlayer.attention.output.dense.weight = layer.attention.output.dense.weight
    model.trelm_electra.encoder.tlayer.attention.output.dense.bias = layer.attention.output.dense.bias
    model.trelm_electra.encoder.tlayer.attention.output.LayerNorm.weight = layer.attention.output.LayerNorm.weight
    model.trelm_electra.encoder.tlayer.attention.output.LayerNorm.bias = layer.attention.output.LayerNorm.bias

    model.trelm_electra.encoder.tlayer.intermediate.dense.weight = layer.intermediate.dense.weight
    model.trelm_electra.encoder.tlayer.intermediate.dense.bias = layer.intermediate.dense.bias

    model.trelm_electra.encoder.tlayer.output.dense.weight = layer.output.dense.weight
    model.trelm_electra.encoder.tlayer.output.dense.bias = layer.output.dense.bias
    model.trelm_electra.encoder.tlayer.output.LayerNorm.weight = layer.output.LayerNorm.weight
    model.trelm_electra.encoder.tlayer.output.LayerNorm.bias = layer.output.LayerNorm.bias
    
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)


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
    parser.add_argument("--electra_model_ckpt_path", type=str, default="")
    parser.add_argument("--electra_model_config_path", type=str, default="")
    parser.add_argument("--trelm_model_path", type=str, default="")
    parser.add_argument("--trelm_vocab_path", type=str, default="")
    parser.add_argument("--trelm_emb_path", type=str, default="")
    parser.add_argument("--trelm_emb_type", type=str, default="pth")
    parser.add_argument("--do_lower_case", type=bool_flag, default=False)
    parser.add_argument("--langids", type=str, default="en,zh")
    args = parser.parse_args()

    fix_electra_model_for_trelm(
        args.electra_model_ckpt_path, 
        os.path.join(args.trelm_model_path, 'pytorch_model.bin')
    )

    shutil.copyfile(args.electra_model_config_path,  os.path.join(args.trelm_model_path, 'config.json'))

    create_trelm_electra_model(
        args.trelm_model_path, 
        args.trelm_vocab_path, 
        args.do_lower_case, 
        args.trelm_emb_path,
        args.trelm_emb_type,
        args.trelm_model_path,
        args.langids.split(',')
    )