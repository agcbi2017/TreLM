# Introduction
This repository contains the code for TreLM: a general cross-lingual transfer learning framework for pre-trained language models, which is introduced in the ICLR-21 submission.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.6.0
* [transformers](https://github.com/huggingface/transformers) version >= 3.4.0
* [tokenizers](https://github.com/huggingface/tokenizers)
* [fastText](https://github.com/facebookresearch/fastText)
* [faiss](https://github.com/facebookresearch/faiss)
* [fast_align](https://github.com/clab/fast_align)
* Python version >= 3.6

**Installing from source**

To install fairseq from source and develop locally:
```
git clone https://github.com/agcbi2017/TreLM
cd TreLM
```

# Getting Started

### Workspace and Data Download
```shell
mkdir /path/to/trelm_en_zh_workspace

# download the monolingual and parallel data to the workspace

```
### Vocabulary Preparation
```shell
# vocabulary size: 80K alphabet size: 30K
mkdir /path/to/trelm_en_id_workspace/tokenizer_en_zh_v80k_a30k

python ./trelm/scripts/train_wordpiece_tokenizer.py \
    --input_file /path/to/trelm_en_zh_workspace/all_en_zh.text \
    --vocab_size 80000 --limit_alphabet 30000 \
    --output_path /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k
```

# Data Preparation
```shell

# tokenize the data like:
python ./trelm/scripts/bert_tokenize.py \
    --model_path /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/vocab.txt \
    --do_lower_case false \
    --input_file /path/to/trelm_en_zh_workspace/all_en_zh.text \
    --output_file /path/to/trelm_en_zh_workspace/all_en_zh.tok.text

# prepare the text with langid like:
cat /path/to/trelm_en_zh_workspace/all_zh.text | shuf | grep -Ev '^\s*$' | tr -d '\r' | head -n 1000000 | awk '{print "zh\t" $0}' > /path/to/trelm_en_zh_workspace/joint_training_data_1m/train_1m_zh.txt

# prepare the parallel data in jsonl format like:

# obtain the alignment information
/path/to/fast_align -i /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh -d -o -v > /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh.forward.align

/path/to/fast_align -i /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh -d -o -v -r > /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh.reverse.align

# convert the alignment information to jsonl format
python ./trelm/scripts/read_alignment_to_jsonl.py \
    --input_para_file  /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh 
    --input_align_file  /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh.forward.align \
    --left_lang en --right_lang zh \
    --jsonl_out_path  /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh.jsonl

python ./trelm/scripts/read_alignment_to_jsonl.py \
    --input_para_file /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh \
    --input_align_file /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.en-zh.reverse.align \
    --left_lang en --right_lang zh --reverse true \
    --jsonl_out_path /path/to/trelm_en_zh_workspace/transfer_training_data/para.tok.clean.zh-en.jsonl
```

### FastText Pretraining
```shell


# fastText pretraining
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epoch
DIM=768

FASTTEXT=/path/to/fasttext 

NAME=trelm_en_zh.fasttext.${DIM}d
INPUT=/path/to/trelm_en_zh_workspace/all_en_zh.tok.text
OUTPUT=/path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/$NAME
LOG=/path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/${NAME}.log

$FASTTEXT skipgram -epoch $N_EPOCHS \
    -minCount 0 -dim $DIM -thread $N_THREADS -ws 5 -neg 10 \
    -input $INPUT -output $OUTPUT 1>$LOG 2>&1

# extract the embedding for joint vocabulary
python ./trelm/scripts/extact_vocab_fasttext_embedding.py \
    /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/trelm_en_zh.fasttext.768d.bin \
    /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/tokenizer_en_zh_v80k_a30k/vocab.txt \
    /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/vocab.fastext.768d.vec \
    /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/vocab.fastext.768d.w2vec

```

### BERT's Raw Embedding Extraction
```shell

python ./trelm/scripts/extract_bert_embedding.py \
    --model_path /path/to/pretrain/bert/bert-base-cased/ \
    --emb_out_path /path/to/trelm_en_zh_workspace/init_emb/trelm-bert-base-emb.pth

```

### Embedding Adversarial Alignment
```shell

CUDA_VISIBLE_DEVICES=0 python trelm/MUSE/unsupervised.py \
    --exp_path /path/to/trelm_en_zh_workspace/emb_align/ \
    --exp_name bert_base_emb_align \
    --exp_id e50_r5_mv-1_dmf0 \
    --src_lang trelm --tgt_lang bert \
    --src_emb /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/vocab.fastext.768d.w2vec \
    --tgt_emb /path/to/trelm_en_zh_workspace/init_emb/trelm-bert-base-emb.pth \
    --n_epochs 50 \
    --n_refinement 5 \
    --fixed_tgt_emb true \
    --emb_dim 768 \
    --max_vocab -1 \
    --dis_most_frequent 0 \
    --export pth > /path/to/trelm_en_zh_workspace/emb_align/bert_base_emb_align_e50_r5_mv-1_dmf0.log 2>&1

```

### Create the Initial Checkpoint
```shell
mkdir /path/to/trelm_en_zh_workspace/init_model/trelm_bert_base
python ./trelm/create_trelm_bert_model.py \
    --bert_model_ckpt_path ./pretrain/bert/bert-base-cased/pytorch_model.bin \
    --bert_model_config_path ./pretrain/bert/bert-base-cased/config.json \
    --trelm_model_path /path/to/trelm_en_zh_workspace/init_model/trelm_bert_base \
    --trelm_vocab_path /path/to/trelm_en_zh_workspace/tokenizer_en_zh_v80k_a30k/vocab.txt \
    --trelm_emb_path /path/to/trelm_en_zh_workspace/emb_align/bert_base_emb_align/e50_r5_mv-1_dmf0/vectors-trelm.pth
```

### Commonality Training (embedding + TRILayer)
```shell
EXP_NAME=trelm_bert
UPDATES=20000
LR=0.00003
NGPUS=8
BATCH_SIZE=8
UPDATE_FREQ=2
EXP_ID=trelm_bert_base_joint_up${UPDATES}_lr${LR}_ngpus${NGPUS}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}
SAVE_PATH=./outputs/$EXP_NAME/${EXP_ID}
LOG_PATH=./logs/$EXP_NAME/${EXP_ID}.log

if [ ! -d ./outputs/$EXP_NAME/ ];
then 
    mkdir -p ./outputs/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python ./trelm/train_trelm_bert_model.py \
    --output_dir ${SAVE_PATH} \
    --init_model_path /path/to/trelm_en_zh_workspace/init_model/trelm_bert_base \
    --train_datapath /path/to/trelm_en_zh_workspace/joint_training_data_1m/train_en_zh_shuf_1m.txt \
    --val_datapath /path/to/trelm_en_zh_workspace/joint_training_data_1m/eval_en_zh_shuf.txt \
    --warmup_steps 500 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps ${UPDATES} \
    --logging_steps 500 \
    --save_steps 500 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${UPDATE_FREQ} \
    --evaluate_during_training \
    --do_train \
    --do_eval > ${LOG_PATH} 2>&1
```

### Transfer Training (Above-Half / Below-Half)
```shell
# en->zh CdLM pretraining for Above-Half
EXP_NAME=trelm_bert
UPDATES=20000
LR=0.00003
NGPUS=8
BATCH_SIZE=8
UPDATE_FREQ=2
EXP_ID=trelm_bert_base_transfer_ahalf_up${UPDATES}_lr${LR}_ngpus${NGPUS}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}
SAVE_PATH=./outputs/$EXP_NAME/${EXP_ID}
LOG_PATH=./logs/$EXP_NAME/${EXP_ID}.log

if [ ! -d ./outputs/$EXP_NAME/ ];
then 
    mkdir -p ./outputs/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python ./trelm/transfer_train_trelm_bert_model.py \
    --output_dir ${SAVE_PATH} \
    --init_model_path ./outputs/trelm_bert/trelm_bert_base_joint_up20000_lr0.00003_ngpus8_bs8_uf2/ \
    --train_datapath /path/to/trelm_en_zh_workspace/transfer_training_data/wmt2020.tok.clean.en-zh.1m.shuf.jsonl \
    --val_datapath /path/to/trelm_en_zh_workspace/transfer_training_data/wmt2020.tok.clean.en-zh.eval.shuf.jsonl \
    --warmup_steps 500 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps ${UPDATES} \
    --logging_steps 500 \
    --save_steps 500 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${UPDATE_FREQ} \
    --evaluate_during_training \
    --do_train \
    --do_eval > ${LOG_PATH} 2>&1

# zh->en CdLM pretraining for Below-Half
EXP_NAME=trelm_bert
UPDATES=20000
LR=0.00003
NGPUS=8
BATCH_SIZE=8
UPDATE_FREQ=2
EXP_ID=trelm_bert_base_transfer_bhalf_up${UPDATES}_lr${LR}_ngpus${NGPUS}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}
SAVE_PATH=./outputs/$EXP_NAME/${EXP_ID}
LOG_PATH=./logs/$EXP_NAME/${EXP_ID}.log

if [ ! -d ./outputs/$EXP_NAME/ ];
then 
    mkdir -p ./outputs/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python ./trelm/transfer_train_trelm_bert_model.py \
    --output_dir ${SAVE_PATH} \
    --init_model_path ./outputs/trelm_bert/trelm_bert_base_joint_up20000_lr0.00003_ngpus8_bs8_uf2/ \
    --train_datapath /path/to/trelm_en_zh_workspace/transfer_training_data/wmt2020.tok.clean.zh-en.1m.shuf.jsonl \
    --val_datapath /path/to/trelm_en_zh_workspace/transfer_training_data/wmt2020.tok.clean.zh-en.eval.shuf.jsonl \
    --fix_ahalf \
    --warmup_steps 500 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps ${UPDATES} \
    --logging_steps 500 \
    --save_steps 500 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${UPDATE_FREQ} \
    --evaluate_during_training \
    --do_train \
    --do_eval > ${LOG_PATH} 2>&1

# combine the two halves
mkdir ./outputs/trelm_bert/trelm_bert_base_transfer_combine_up20000_lr0.00003_ngpus8_bs8_uf2/
python ./trelm/combine_half_bert_model.py \
    --trelm_ahalf_model_path ./outputs/trelm_bert/trelm_bert_base_transfer_ahalf_up20000_lr0.00003_ngpus8_bs8_uf2/pytorch_model.bin \
    --trelm_bhalf_model_path ./outputs/trelm_bert/trelm_bert_base_transfer_bhalf_up20000_lr0.00003_ngpus8_bs8_uf2/pytorch_model.bin \
    --trelm_config_path ./outputs/trelm_bert/trelm_bert_base_transfer_ahalf_up20000_lr0.00003_ngpus8_bs8_uf2/config.json \
    --trelm_vocab_path ./outputs/trelm_bert/trelm_bert_base_transfer_ahalf_up20000_lr0.00003_ngpus8_bs8_uf2/vocab.txt \
    --trelm_output_path ./outputs/trelm_bert/trelm_bert_base_transfer_combine_up20000_lr0.00003_ngpus8_bs8_uf2/
```

### Language-specific Training
```shell

EXP_NAME=trelm_bert
UPDATES=80000
LR=0.00003
NGPUS=8
BATCH_SIZE=8
UPDATE_FREQ=2
EXP_ID=trelm_bert_base_indiv_up${UPDATES}_lr${LR}_ngpus${NGPUS}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}
SAVE_PATH=./outputs/$EXP_NAME/${EXP_ID}
LOG_PATH=./logs/$EXP_NAME/${EXP_ID}.log

if [ ! -d ./outputs/$EXP_NAME/ ];
then 
    mkdir -p ./outputs/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python ./trelm/train_trelm_bert_model.py \
    --output_dir ${SAVE_PATH} \
    --init_model_path ./outputs/trelm_bert/trelm_bert_base_transfer_combine_up20000_lr0.00003_ngpus8_bs8_uf2/ \
    --train_datapath /path/to/trelm_en_zh_workspace/indiv_training_data_2m/train_2m_zh.txt \
    --val_datapath /path/to/trelm_en_zh_workspace/indiv_training_data_2m/eval_zh.txt \
    --finetune_self_attn \
    --block_size 512 \
    --window_size 510 \
    --warmup_steps 500 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps ${UPDATES} \
    --logging_steps 500 \
    --save_steps 500 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${UPDATE_FREQ} \
    --evaluate_during_training \
    --do_train \
    --do_eval > ${LOG_PATH} 2>&1

```

# Transfered Models
* Coming soon...
