# PyTorch Reimplementation of REALM and ORQA

This is PyTorch reimplementation of REALM ([paper](https://arxiv.org/abs/2002.08909), [codebase](https://github.com/google-research/language/tree/master/language/realm)) and ORQA ([paper](https://arxiv.org/abs/1906.00300), [codebase](https://github.com/google-research/language/tree/master/language/orqa)). 


*The term `Scorer` is actually the pretraining `Retriever` in REALM paper, we change it to `Scorer` to prevent a conflict with finetuning `Retriever`.*


## Prerequisite

```bash
pip install -U transformers apache_beam
```

## Data

To download TensorFlow checkpoints and preprocessed data, please follow the instructions below:

```bash
cd data
pip install -U -r requirements.txt
sh download.sh
```

To convert pretrained TensorFlow checkpoints like **CC-News** to PyTorch checkpoints:

```bash
python checkpoint_converter.py \
    --block_records_path "data/enwiki-20181220/blocks.tfr" \
    --block_emb_path "./data/cc_news_pretrained/embedder/encoded/encoded.ckpt" \
    --embedder_path "./data/cc_news_pretrained/embedder/variables/variables" \
    --bert_path "./data/cc_news_pretrained/bert/variables/variables" \
    --output_path path_to_save_converted_model \
    --from_pretrained
```

To convert finetuned TensorFlow checkpoints like **Natural Questions (NQ)** to PyTorch checkpoints:

```bash
python checkpoint_converter.py \
    --block_records_path "data/enwiki-20181220/blocks.tfr" \
    --block_emb_path "./data/cc_news_pretrained/embedder/encoded/encoded.ckpt" \
    --checkpoint_path "./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000" \
    --output_path path_to_save_converted_model
```

## Predict

The default checkpoint is `qqaatw/realm-orqa-nq-openqa`. To change it, kindly specify `--checkpoint_pretrained_name`, which can be a local path or a model name on the huggingface model hub.

```bash
python predictor.py --question "Who is the pioneer in modern computer science?"

Output: alan mathison turing
```

## Finetune (Experimental)

The default finetuning dataset is **Natural Questions (NQ)**. To load your custom dataset, please change the loading function in `data.py`.

Training:

```bash
python run_finetune.py --is_train \
    --checkpoint_pretrained_name "qqaatw/realm-cc-news-pretrained-openqa" \
    --checkpoint_name "checkpoint" \
    --dataset_name_path "natural_questions" \
    --model_dir "./" \
    --num_epochs 2 \
    --device cuda
```

The output model will be stored in `./checkpoint-x`, where x is the training step when saving.

Evaluation:

```bash
python run_finetune.py \
    --checkpoint_pretrained_name "qqaatw/realm-cc-news-pretrained-openqa" \
    --checkpoint_name "checkpoint" \
    --checkpoint_step 50000 \
    --dataset_name_path "natural_questions" \
    --model_dir "./" \
    --device cuda
```

## Benchmark

### Natural Questions (NQ)

Using brute-force matrix multiplication searcher:

```bash
python benchmark.py \
    --dataset_name_path natural_questions \
    --checkpoint_pretrained_name qqaatw/realm-orqa-nq-openqa \
    --device cuda
```

Outputs with brute-force matrix multiplication searcher:

```
exact_match:0.410526305437088
official_exact_match:0.4041551351547241 # value in the paper: ~0.404
reader_oracle:0.7193905711174011
top_5_match:0.7218836545944214
top_10_match:0.7218836545944214
top_50_match:0.7218836545944214
top_100_match:0.7218836545944214
top_500_match:0.7218836545944214
top_1000_match:0.7218836545944214
top_5000_match:0.7218836545944214
```

~Using ScaNN searcher~(currently not available):

```bash
python run_finetune.py --benchmark --use_scann
```

Outputs with ScaNN searcher:

```
exact_match:0.4019390642642975
official_exact_match:0.3972299098968506 # value in the paper: ~0.404
reader_oracle:0.7041551470756531
top_5_match:0.7058171629905701
top_10_match:0.7058171629905701
top_50_match:0.7058171629905701
top_100_match:0.7058171629905701
top_500_match:0.7058171629905701
top_1000_match:0.7058171629905701
top_5000_match:0.7058171629905701
```

### Web Questions (WQ)

Using brute-force matrix multiplication searcher:

```bash
python benchmark.py \
    --dataset_name_path web_questions \
    --checkpoint_pretrained_name qqaatw/realm-orqa-wq-openqa \
    --device cuda
```

Outputs with brute-force matrix multiplication searcher:

```
exact_match:0.4345472455024719
official_exact_match:0.41683071851730347 # value in the paper: ~0.407
reader_oracle:0.6929134130477905
top_5_match:0.6934055089950562
top_10_match:0.6934055089950562
top_50_match:0.6934055089950562
top_100_match:0.6934055089950562
top_500_match:0.6934055089950562
top_1000_match:0.6934055089950562
top_5000_match:0.6934055089950562
```

~Using ScaNN searcher~(currently not available):

```bash
python run_finetune.py \
    --benchmark \
    --use_scann \
    --dataset_name_path web_questions \
    --retriever_path ./data/orqa_wq_model_from_realm/export/best_default/checkpoint/model.ckpt-205020 \
    --checkpoint_path ./data/orqa_wq_model_from_realm/export/best_default/checkpoint/model.ckpt-205020
```

Outputs with ScaNN searcher:

```
exact_match:0.42814961075782776
official_exact_match:0.4114173352718353 # value in the paper: ~0.407
reader_oracle:0.6840550899505615
top_5_match:0.6840550899505615
top_10_match:0.6840550899505615
top_50_match:0.6840550899505615
top_100_match:0.6840550899505615
top_500_match:0.6840550899505615
top_1000_match:0.6840550899505615
top_5000_match:0.6840550899505615
```

## License

Apache License 2.0