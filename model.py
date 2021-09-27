from transformers import RealmTokenizer
from transformers.models.realm.modeling_realm import (
    load_tf_weights_in_realm,
    RealmConfig,
    RealmReader,
    RealmSearcher,
)

def get_searcher_reader_tokenizer(args):
    searcher_config = RealmConfig(hidden_act="gelu_new")
    searcher = RealmSearcher(searcher_config, args.block_records_path)
    
    # Load retriever weights
    searcher = load_tf_weights_in_realm(
        searcher,
        searcher_config,
        args.retriever_path,
    )

    # Load block_emb weights
    searcher = load_tf_weights_in_realm(
        searcher,
        searcher_config,
        args.block_emb_path,
    )
    searcher.eval()

    reader_config = RealmConfig(hidden_act="gelu_new")
    reader = RealmReader.from_pretrained(
        args.checkpoint_path,
        config=reader_config,
        from_tf=True,
    )
    reader.eval()

    tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder", do_lower_case=True)

    return searcher, reader, tokenizer