# PyTorch Reimplementation of REALM and ORQA

This is PyTorch reimplementation of REALM ([paper](https://arxiv.org/abs/2002.08909), [codebase](https://github.com/google-research/language/tree/master/language/realm)) and ORQA ([paper](https://arxiv.org/abs/1906.00300), [codebase](https://github.com/google-research/language/tree/master/language/orqa)). 

Some features have not been implemented yet, currently the predictor and finetuning script are available.

*The term retriever and searcher in the code are basically interchangeable, their difference is that retriever is for REALM pretraining, and searcher is for ORQA finetuning.*


## Prerequisite

```bash
cd transformers && pip install -U -e ".[dev]"
pip install -U scann, apache_beam
```

## Data

To download pretrained checkpoints and preprocessed data, please follow the instructions below:

```bash
cd data
pip install -U -r requirements.txt
sh download.sh
```

## Finetune (Experimental)

The default finetuning dataset is **NaturalQuestions(NQ)**. To load your custom dataset, please change the loading function in `data.py`.

Training:

```bash
python run_finetune.py --is_train \
    --model_dir "./" \
    --num_epochs 2 \
    --device cuda
```

Evaluation:

```bash
python run_finetune.py \
    --retriever_pretrained_name "retriever" \
    --checkpoint_pretrained_name "reader" \
    --model_dir "./" \
    --device cuda
```

## Predict

The default checkpoints of retriever and reader are `orqa_nq_model_from_realm`. To change them, kindly specify `--retriever_path` and `--checkpoint_path`.

```bash
python predictor.py --question "Who is the pioneer in modern computer science?"

Output: alan mathison turing
```

## Benchmark

Benchmarking on NaturalQuestions(NQ).

To run benchmark, please ensure that `./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000` exists.

Using ScaNN searcher:

```bash
python run_finetune.py --benchmark --use_scann
```

Outputs with ScaNN searcher:

```
INFO:root:exact_match:0.40166205167770386 # value in the paper: ~0.404
official_exact_match:0.3972299098968506
reader_oracle:0.7047091126441956
top_5_match:0.7063711881637573
top_10_match:0.7063711881637573
top_50_match:0.7063711881637573
top_100_match:0.7063711881637573
top_500_match:0.7063711881637573
top_1000_match:0.7063711881637573
top_5000_match:0.7063711881637573
```

Using brute-force matrix multiplication searcher:

```bash
python run_finetune.py --benchmark
```

Outputs with brute-force matrix multiplication searcher:

```
INFO:root:exact_match:0.4102492928504944  # value in the paper: ~0.404
official_exact_match:0.4041551351547241
reader_oracle:0.7193905711174011
top_5_match:0.7218836545944214
top_10_match:0.7218836545944214
top_50_match:0.7218836545944214
top_100_match:0.7218836545944214
top_500_match:0.7218836545944214
top_1000_match:0.7218836545944214
top_5000_match:0.7218836545944214
```

## License

Apache License 2.0