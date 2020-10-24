import logging
import collections
import os
import time
import pickle
import random
import json

from typing import Dict, Optional, List, Union, Tuple

from filelock import FileLock

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast
from trelm_bert import TrelmBertForMaskedLM

import math
from dataclasses import dataclass, field
from transformers import Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers import PreTrainedTokenizer

from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class JsonAlignTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        langs_to_id: dict, 
        file_path: str, 
        block_size: int,
        fix_ahalf: bool,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_transfer_lm_{}_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
                'fix_ahalf' if fix_ahalf else 'fix_bhalf',
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:

                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    cache_data = pickle.load(handle)
                self.src = cache_data['src']
                self.src_langids = cache_data['src_langids']
                self.tgt = cache_data['tgt']
                self.tgt_langids = cache_data['tgt_langids']
                self.align = cache_data['align']
                self.for_mlm = cache_data['for_mlm']
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.src = []
                self.src_langids = []
                self.tgt = []
                self.tgt_langids = []
                self.align = []
                self.for_mlm = []

                cls_token_id = tokenizer.cls_token_id
                sep_token_id = tokenizer.sep_token_id

                with open(file_path, encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if len(line) > 0:
                            line = json.loads(line)

                            src_ids = tokenizer.convert_tokens_to_ids(line['src'])
                            src_langid = langs_to_id[line['src_lang']]

                            assert len(src_ids) <= block_size and src_ids[0] == cls_token_id and src_ids[-2] == sep_token_id and src_ids[-1] == tokenizer.pad_token_id

                            tgt_ids = tokenizer.convert_tokens_to_ids(line['tgt'])
                            tgt_langid = langs_to_id[line['tgt_lang']]

                            assert len(tgt_ids) <= block_size and tgt_ids[0] == cls_token_id and tgt_ids[-1] == sep_token_id

                            align_index = line['align_index']

                            assert len(tgt_ids) == len(align_index)

                            pad_align_index = [len(src_ids)-1] *  block_size
                            pad_align_index[:len(tgt_ids)] = align_index

                            pad_src_ids = [tokenizer.pad_token_id] * block_size
                            pad_src_ids[:len(src_ids)] = src_ids

                            pad_tgt_ids = [tokenizer.pad_token_id] * block_size
                            pad_tgt_ids[:len(tgt_ids)] = tgt_ids

                            self.src.append(pad_src_ids)
                            self.src_langids.append(src_langid)
                            self.tgt.append(pad_tgt_ids)
                            self.tgt_langids.append(tgt_langid)
                            self.align.append(pad_align_index)
                            self.for_mlm.append(False)

                            # for mlm
                            if fix_ahalf:
                                self.src.append(pad_src_ids)
                                self.src_langids.append(src_langid)
                                self.tgt.append(pad_src_ids)
                                self.tgt_langids.append(src_langid)
                                self.align.append(list(range(block_size)))
                                self.for_mlm.append(True)
                            else:
                                self.src.append(pad_tgt_ids)
                                self.src_langids.append(tgt_langid)
                                self.tgt.append(pad_tgt_ids)
                                self.tgt_langids.append(tgt_langid)
                                self.align.append(list(range(block_size)))
                                self.for_mlm.append(True)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    cache_data = {'src': self.src, 'src_langids': self.src_langids, 'tgt': self.tgt, 'tgt_langids': self.tgt_langids, 'align': self.align, 'for_mlm': self.for_mlm}
                    pickle.dump(cache_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return {
            "src_ids": torch.tensor(self.src[i], dtype=torch.long), 
            "src_langids": torch.tensor([self.src_langids[i]], dtype=torch.long),
            "tgt_ids": torch.tensor(self.tgt[i], dtype=torch.long), 
            "tgt_langids": torch.tensor([self.tgt_langids[i]], dtype=torch.long),
            "align_index": torch.tensor(self.align[i], dtype=torch.long),
            "for_mlm": torch.tensor(self.for_mlm[i]),
        }
            


@dataclass
class AlignDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    # wwm: bool = False

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        assert isinstance(examples[0], dict)
        
        input_ids = []
        lang_ids = []
        tlayer_lang_ids = []
        align_index = []
        labels = []

        for e in examples:
            if e['for_mlm']:
                mlm_input_ids, mlm_label_ids = self.mask_tokens(e['src_ids'].unsqueeze(0))
                input_ids.append(mlm_input_ids.squeeze(0))
                labels.append(mlm_label_ids.squeeze(0))
            else:
                input_ids.append(e['src_ids'])
                labels.append(e['tgt_ids'])
            lang_ids.append(e['src_langids'])
            tlayer_lang_ids.append(e['tgt_langids'])
            align_index.append(e['align_index'])

        batch = self._tensorize_batch(input_ids)
        lang_ids = torch.stack(lang_ids, dim=0)
        tlayer_lang_ids = torch.stack(tlayer_lang_ids, dim=0)
        align_index = torch.stack(align_index, dim=0)
        labels = self._tensorize_batch(labels)

        assert lang_ids.size(0) == batch.size(0)
        
        assert self.tokenizer.pad_token_id is not None
        
        labels[labels == self.tokenizer.pad_token_id] = -100

        ordering_attention_mask = torch.ones(labels.size(), device=batch.device) 
        ordering_attention_mask[labels == self.tokenizer.pad_token_id] == 0

        attention_mask = torch.ones(batch.size(), device=batch.device)
        attention_mask[batch == self.tokenizer.pad_token_id] = 0

        return {"input_ids": batch, "lang_ids": lang_ids, "tlayer_lang_ids": tlayer_lang_ids, "ordering_alignment": align_index, "ordering_attention_mask": ordering_attention_mask, "attention_mask": attention_mask, "labels": labels}
            

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        assert self.tokenizer._pad_token is not None
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if self.wwm:
        #     # word_indices = torch.LongTensor([0, 0, 1, 2, 2, 2, 3])
        #     masked_word_indices = word_indices.masked_select(masked_indices)
        #     masked_size = masked_word_indices.shape[0]
        #     masked_word_indices = masked_word_indices.unsqueeze(dim=-1).expand((masked_size, masked_indices.shape[1]))
        #     wwm_masked_indices = torch.sum(masked_word_indices == word_indices, dim=0)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



def pretrain_and_evaluate(args, data_args, model, tokenizer, eval_only, model_path):
    val_dataset = JsonAlignTextDataset(tokenizer=tokenizer,
                              langs_to_id=model.config.langs_to_id,
                              file_path=data_args.val_datapath,
                              block_size=data_args.block_size,
                              fix_ahalf=data_args.fix_ahalf)
    
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {data_args.train_datapath}')
        if ',' in data_args.train_datapath:
            train_datasets = []
            for train_datapath in data_args.train_datapath.split(','):
                train_datasets.append(
                    JsonAlignTextDataset(tokenizer=tokenizer,
                        langs_to_id=model.config.langs_to_id,
                        file_path=train_datapath,
                        block_size=data_args.block_size,
                        fix_ahalf=data_args.fix_ahalf
                    )
                )
                train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = JsonAlignTextDataset(tokenizer=tokenizer,
                langs_to_id=model.config.langs_to_id,
                file_path=data_args.train_datapath,
                block_size=data_args.block_size,
                fix_ahalf=data_args.fix_ahalf
            )

    data_collator = AlignDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')


@dataclass
class DataArgs:
    train_datapath: str = field(default='', metadata={"help": "training dataset path"})
    val_datapath: str = field(default='', metadata={"help": "validation dataset path"})
    init_model_path: str = field(default='', metadata={"help": "initial model path"})
    block_size: int = field(default=512, metadata={"help": "block size"})
    fix_ahalf: bool = field(default=False, metadata={"help": "fix the above half attention layer"})

if __name__ == '__main__':

    parser = HfArgumentParser((TrainingArguments, DataArgs, ))

    training_args, data_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    trelm_bert_model = TrelmBertForMaskedLM.from_pretrained(data_args.init_model_path)
    trelm_bert_tokenizer = BertTokenizerFast.from_pretrained(data_args.init_model_path)

    if not data_args.fix_ahalf:
        # fix the below half self-attention parameters
        for layer_idx in range(trelm_bert_model.trelm_bert.encoder.tlayer_position):
            for param in trelm_bert_model.trelm_bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
    else:
        # fix the above half self-attention parameters
        for layer_idx in range(trelm_bert_model.trelm_bert.encoder.tlayer_position, len(trelm_bert_model.trelm_bert.encoder.layer)):
            for param in trelm_bert_model.trelm_bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    logger.info(trelm_bert_model)

    logger.info('Evaluating trelm-bert for refernece ...')
    pretrain_and_evaluate(training_args, data_args, trelm_bert_model, trelm_bert_tokenizer, eval_only=True, model_path=None)

    logger.info(f'Pretraining trelm-bert ... ')
    pretrain_and_evaluate(training_args, data_args, trelm_bert_model, trelm_bert_tokenizer, eval_only=False, model_path=training_args.output_dir)

    model_path = training_args.output_dir
    
    logger.info(f'Saving model to {model_path}')
    trelm_bert_model.save_pretrained(model_path)
    trelm_bert_tokenizer.save_pretrained(model_path)