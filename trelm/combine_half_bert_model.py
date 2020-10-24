import logging
import argparse
import collections
import os
import shutil
import json

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def combine_half_model(ahalf_model_path, bhalf_model_path, config_path, model_output_path):

    with open(config_path, 'r', encoding='utf-8') as fin:
        config = json.load(fin)

    assert config['num_hidden_layers'] % 2 == 0, "num_hidden_layers must be even in Trelm!"
    tlayer_position = int(config['num_hidden_layers'] / 2)

    ahalf_state = torch.load(
        ahalf_model_path,
        map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ),
    )

    bhalf_state = torch.load(
        bhalf_model_path,
        map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ),
    )

    ahalf_model_params_keys = list(ahalf_state.keys())

    bhalf_model_params_keys = list(bhalf_state.keys())

    assert ahalf_model_params_keys == bhalf_model_params_keys

    new_params_dict = collections.OrderedDict()

    for k in ahalf_model_params_keys:
        ahalf_p = ahalf_state[k]
        if isinstance(ahalf_p, torch.HalfTensor):
            ahalf_p = ahalf_p.float()

        bhalf_p = bhalf_state[k]
        if isinstance(bhalf_p, torch.HalfTensor):
            bhalf_p = bhalf_p.float()

        if k.startswith('trelm_bert.embeddings') or \
            k.startswith('trelm_bert.encoder.tlayer') or \
            k.startswith('trelm_bert.pooler') or \
            k.startswith('cls'):
            new_params_dict[k] = ahalf_p.clone() + bhalf_p.clone()
            if new_params_dict[k].is_floating_point():
                new_params_dict[k].div_(2)
            else:
                new_params_dict[k] //= 2
        elif k.startswith('trelm_bert.encoder.layer'):
            if int(k.split('.')[3]) < tlayer_position:
                new_params_dict[k] = bhalf_p.clone()
            else:
                new_params_dict[k] = ahalf_p.clone()
        else:
            raise ValueError("key {} is not supported!".format(k))

    torch.save(new_params_dict, model_output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--trelm_ahalf_model_path", type=str, default="")
    parser.add_argument("--trelm_bhalf_model_path", type=str, default="")
    parser.add_argument("--trelm_config_path", type=str, default="")
    parser.add_argument("--trelm_vocab_path", type=str, default="")
    parser.add_argument("--trelm_output_path", type=str, default="")
    args = parser.parse_args()

    combine_half_model(args.trelm_ahalf_model_path, args.trelm_bhalf_model_path, args.trelm_config_path, os.path.join(args.trelm_output_path, 'pytorch_model.bin'))

    shutil.copyfile(args.trelm_config_path,  os.path.join(args.trelm_output_path, 'config.json'))
    shutil.copyfile(args.trelm_vocab_path,  os.path.join(args.trelm_output_path, 'vocab.txt'))
