# PyTorch Reimplementation of REALM and ORQA

This is PyTorch reimplementation of REALM ([paper](https://arxiv.org/abs/2002.08909), [codebase](https://github.com/google-research/language/tree/master/language/realm)) and ORQA ([paper](https://arxiv.org/abs/1906.00300), [codebase](https://github.com/google-research/language/tree/master/language/orqa)). 

Some features have not been implemented yet, currently only predictor is available.


## Prerequisite

```bash
cd transformers && pip install -U -e ".[dev]"
pip install -U scann
```

## Data

To download pretrained checkpoints and preprocessed data, please follow the instructions below:

```bash
cd data
pip install -U -r requirements.txt
sh download.sh
```

## Predict

The default checkpoints of retriever and reader are `cc_news_pretrained` and `orqa_nq_model_from_realm`, respectively. To change them, kindly specify `--retriever_path` and `--checkpoint_path`.

```bash
python predictor.py --question "Who is the pioneer in modern computer science?"
```

## License

Apache License 2.0