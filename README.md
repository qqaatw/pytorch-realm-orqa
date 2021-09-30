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

The default finetuning dataset is **Natural Question(NQ)**. To laod your custom dataset, please change the loading function in `data.py`.

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

The default checkpoints of retriever and reader are `cc_news_pretrained` and `orqa_nq_model_from_realm`, respectively. To change them, kindly specify `--retriever_path` and `--checkpoint_path`.

```bash
python predictor.py --question "Who is the pioneer in modern computer science?"
```

## License

Apache License 2.0