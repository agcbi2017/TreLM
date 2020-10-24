import logging
import collections
import os
import time
import pickle
import random

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


class LangidLineByLineTextDataset(Dataset):
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
        window_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        full_block_size = block_size
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        assert window_size <= block_size

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_joint_lm_{}_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(full_block_size),
                str(window_size),
                filename,
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
                self.examples = cache_data['examples']
                self.langids = cache_data['langids']
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                self.langids = []
                sent_count = 0
                with open(file_path, encoding="utf-8") as fin:
                    for line in fin:
                        if (len(line) > 0 and not line.isspace()):
                            line = line.split('\t')
                            assert len(line) >= 2, line
                            langid = langs_to_id[line[0]]
                            text = '\t'.join(line[1:])

                            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                            # offset = 0
                            # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                            #     self.examples.append(
                            #         tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                            #     )
                            #     self.langids.append(langid)
                            #     offset += block_size
                            
                            # if offset < len(tokenized_text):
                            #     input_ids = tokenizer.build_inputs_with_special_tokens(tokenized_text[offset : len(tokenized_text)])
                            #     pad_input_ids = [tokenizer.pad_token_id] * full_block_size
                            #     pad_input_ids[:len(input_ids)] = input_ids
                            #     self.examples.append(pad_input_ids)
                            #     self.langids.append(langid)

                            offset = 0
                            while offset < len(tokenized_text):
                                pad_input_ids = [tokenizer.pad_token_id] * full_block_size
                                if offset + block_size > len(tokenized_text):
                                    end = len(tokenized_text)
                                else:
                                    end = offset + block_size
                                input_ids = tokenizer.build_inputs_with_special_tokens(tokenized_text[offset : end])
                                pad_input_ids[:len(input_ids)] = input_ids
                                self.examples.append(pad_input_ids)
                                self.langids.append(langid)
                                offset += window_size
                                if end >= len(tokenized_text):
                                    break
                            
                            sent_count += 1
                            if sent_count % 1000 == 0:
                                logger.info("processed sents: {}".format(sent_count))

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    cache_data = {'examples': self.examples, 'langids': self.langids}
                    pickle.dump(cache_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
            
            assert len(self.examples) == len(self.langids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.examples[i], dtype=torch.long), "lang_ids": torch.tensor([self.langids[i]], dtype=torch.long)}


@dataclass
class LangidDataCollatorForLanguageModeling:
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
        
        input_ids = [e["input_ids"] for e in examples]
        lang_ids = [e["lang_ids"] for e in examples]

        batch = self._tensorize_batch(input_ids)

        lang_ids = torch.stack(lang_ids, dim=0)

        assert lang_ids.size(0) == batch.size(0)
        
        assert self.tokenizer.pad_token_id is not None
        attention_mask = torch.ones(batch.size(), device=batch.device)
        attention_mask[batch == self.tokenizer.pad_token_id] = 0
        
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "lang_ids": lang_ids, "attention_mask": attention_mask, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "lang_ids": lang_ids, "attention_mask": attention_mask, "labels": labels}

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
    val_dataset = LangidLineByLineTextDataset(tokenizer=tokenizer,
                              langs_to_id=model.config.langs_to_id,
                              file_path=data_args.val_datapath,
                              block_size=data_args.block_size,
                              window_size=data_args.window_size)
    
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {data_args.train_datapath}')
        if ',' in data_args.train_datapath:
            train_datasets = []
            for train_datapath in data_args.train_datapath.split(','):
                train_datasets.append(
                    LangidLineByLineTextDataset(tokenizer=tokenizer,
                        langs_to_id=model.config.langs_to_id,
                        file_path=train_datapath,
                        block_size=data_args.block_size,
                        window_size=data_args.window_size
                    )
                )
                train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = LangidLineByLineTextDataset(tokenizer=tokenizer,
                langs_to_id=model.config.langs_to_id,
                file_path=data_args.train_datapath,
                block_size=data_args.block_size,
                window_size=data_args.window_size
            )

    data_collator = LangidDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

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
    window_size: int = field(default=510, metadata={"help": "window size"})
    finetune_self_attn: bool = field(default=False, metadata={"help": "finetune the self attention layer"})

if __name__ == '__main__':

    parser = HfArgumentParser((TrainingArguments, DataArgs, ))

    training_args, data_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    trelm_bert_model = TrelmBertForMaskedLM.from_pretrained(data_args.init_model_path)
    trelm_bert_model_tokenizer = BertTokenizerFast.from_pretrained(data_args.init_model_path)

    if not data_args.finetune_self_attn:
        # fix the self-attention parameters
        for param in trelm_bert_model.trelm_bert.encoder.layer.parameters():
            param.requires_grad = False

    logger.info(trelm_bert_model)

    logger.info('Evaluating trelm-bert for refernece ...')
    pretrain_and_evaluate(training_args, data_args, trelm_bert_model, trelm_bert_model_tokenizer, eval_only=True, model_path=None)

    logger.info(f'Pretraining trelm-bert ... ')
    pretrain_and_evaluate(training_args, data_args, trelm_bert_model, trelm_bert_model_tokenizer, eval_only=False, model_path=training_args.output_dir)

    model_path = training_args.output_dir

    logger.info(f'Saving model to {model_path}')
    trelm_bert_model.save_pretrained(model_path)
    trelm_bert_model_tokenizer.save_pretrained(model_path)