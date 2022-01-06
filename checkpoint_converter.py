import logging
from argparse import ArgumentParser
from transformers import RealmConfig
from transformers.models.realm.modeling_realm import logger

from model import get_openqa_tf_finetuned, get_openqa_tf_pretrained

logger.setLevel(logging.INFO)

def get_arg_parser():
    parser = ArgumentParser()

    # ./data/enwiki-20181220/blocks.tfr
    parser.add_argument("--block_records_path", type=str, required=True,
        help="Block records path.")
    # ./data/cc_news_pretrained/embedder/encoded/encoded.ckpt
    parser.add_argument("--block_emb_path", type=str, required=True,
        help="Block embeddings path.")
    
    pretrained_group = parser.add_argument_group("pretrained conversion")
    
    pretrained_group.add_argument("--embedder_path", type=str, default=r"./data/cc_news_pretrained/embedder/variables/variables",
        help="Pretrained embedder path.")
    pretrained_group.add_argument("--bert_path", type=str, default=r"./data/cc_news_pretrained/bert/variables/variables",
        help="Pretrained bert path.")

    finetuned_group = parser.add_argument_group("finetuned conversion")

    finetuned_group.add_argument("--checkpoint_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000",
        help="Finetuned checkpoint path.")
    
    parser.add_argument("--output_path", type=str, default=r"./converted_model/",
        help="Converted checkpoint path.")
    parser.add_argument("--from_pretrained", action="store_true",
        help="Whether to convert from a pretrained checkpoint or a finetuned checkpoint.")

    return parser

def main(args):
    config = RealmConfig()
    
    if args.from_pretrained:
        model = get_openqa_tf_pretrained(args, config)
    else:
        model = get_openqa_tf_finetuned(args, config)
    
    model.save_pretrained(args.output_path)
    model.retriever.save_pretrained(args.output_path)

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)