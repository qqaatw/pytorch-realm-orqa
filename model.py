from transformers import RealmTokenizer
from transformers.models.realm.modeling_realm import (
    load_tf_weights_in_realm,
    RealmConfig,
    RealmReader,
    RealmSearcher,
)

def get_searcher_reader_tokenizer(args, config=None):
    if config is None: 
        config = RealmConfig(hidden_act="gelu_new")
    searcher = RealmSearcher(config, args.block_records_path)
    
    # Load retriever weights
    searcher = load_tf_weights_in_realm(
        searcher,
        config,
        args.retriever_path,
    )

    # Load block_emb weights
    searcher = load_tf_weights_in_realm(
        searcher,
        config,
        args.block_emb_path,
    )
    searcher.eval()

    reader = RealmReader.from_pretrained(
        args.checkpoint_path,
        config=config,
        from_tf=True,
    )
    reader.eval()

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return searcher, reader, tokenizer