from transformers import (
    RealmConfig,
    RealmReader,
    RealmRetriever,
    RealmScorer,
    RealmForOpenQA,
    RealmTokenizer,
    load_tf_weights_in_realm,
)


def get_openqa_tf_finetuned(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    retriever = RealmRetriever(config, tokenizer, args.block_records_path)

    openqa = RealmForOpenQA(config, retriever)

    openqa = load_tf_weights_in_realm(
        openqa,
        config,
        args.checkpoint_path,
    )

    openqa = load_tf_weights_in_realm(
        openqa,
        config,
        args.block_emb_path,
    )

    openqa.eval()

    return openqa

def get_openqa_tf_pretrained(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    retriever = RealmRetriever(config, tokenizer, args.block_records_path)

    openqa = RealmForOpenQA(config, retriever)

    openqa = load_tf_weights_in_realm(
        openqa,
        config,
        args.bert_path,
    )

    openqa = load_tf_weights_in_realm(
        openqa,
        config,
        args.embedder,
    )

    openqa = load_tf_weights_in_realm(
        openqa,
        config,
        args.block_emb_path,
    )

    openqa.eval()

    return openqa

def get_openqa(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    retriever = RealmRetriever(config, tokenizer, args.block_records_path)

    openqa = RealmForOpenQA.from_pretrained(
        args.checkpoint_pretrained_name,
        retriever=retriever,
        config=config,
    )
    openqa.eval()

    return openqa

def get_scorer_reader_tokenizer_tf(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer(config, args.block_records_path)

    # Load retriever weights
    scorer = load_tf_weights_in_realm(
        scorer,
        config,
        args.retriever_path,
    )

    # Load block_emb weights
    scorer = load_tf_weights_in_realm(
        scorer,
        config,
        args.block_emb_path,
    )
    scorer.eval()

    reader = RealmReader.from_pretrained(
        args.checkpoint_path,
        config=config,
        from_tf=True,
    )
    reader.eval()

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return scorer, reader, tokenizer

def get_scorer_reader_tokenizer_pt_pretrained(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer.from_pretrained(args.retriever_pretrained_name, args.block_records_path, config=config)
    
    # Load block_emb weights
    scorer = load_tf_weights_in_realm(
        scorer,
        config,
        args.block_emb_path,
    )
    scorer.eval()
    
    reader = RealmReader.from_pretrained(args.checkpoint_pretrained_name, config=config)
    reader.eval()

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return scorer, reader, tokenizer

def get_scorer_reader_tokenizer_pt_finetuned(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer.from_pretrained(args.retriever_pretrained_name, args.block_records_path, config=config)
    scorer.eval()
    
    reader = RealmReader.from_pretrained(args.checkpoint_pretrained_name, config=config)
    reader.eval()

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return scorer, reader, tokenizer

def get_scorer_reader_tokenizer(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")

    scorer = RealmScorer(config, args.block_records_path)
    reader = RealmReader(config)
    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return scorer, reader, tokenizer