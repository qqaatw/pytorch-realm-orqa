from argparse import ArgumentParser
import os

import torch

from model import get_searcher_reader_tokenizer_pt_finetuned, get_searcher_reader_tokenizer_pt_pretrained, get_searcher_reader_tokenizer_tf
from transformers.models.realm.modeling_realm import logger
from transformers.utils import logging

logger.setLevel(logging.INFO)
torch.set_printoptions(precision=8)


def get_arg_parser():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--from_pt_pretrained", action="store_true", 
        help="Load weights from PyTorch pretrained checkpoints.")
    group.add_argument("--from_pt_finetuned", action="store_true",
        help="Load weights from PyTorch finetuned checkpoints.")
    parser.add_argument("--saved_path", type=str, default=None,
        help="If specified, the PyTorch weights will be stored in this path; otherwise, weights will not be saved.")

    # Question
    parser.add_argument("--question", type=str, required=True,
        help="Input question.")
    # Retriever
    parser.add_argument("--block_emb_path", type=str, default=r"./data/cc_news_pretrained/embedder/encoded/encoded.ckpt")
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--retriever_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000")
    # Reader
    parser.add_argument("--checkpoint_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000")
    # from_pt path
    parser.add_argument("--retriever_pretrained_name", type=str, default=r"qqaatw/realm-orqa-nq-searcher")
    parser.add_argument("--checkpoint_pretrained_name", type=str, default=r"qqaatw/realm-orqa-nq-reader")

    return parser

def retrieve(args, searcher, tokenizer):
    with torch.no_grad():
        question = args.question
        question_ids = tokenizer([question], return_tensors='pt')

        output = searcher(**question_ids, return_dict=True)

    print(output)

    return output

def read(args, reader, tokenizer, searcher_output):
    with torch.no_grad():
        text = []
        text_pair = []
        for retrieved_block in searcher_output.retrieved_blocks:
            text.append(args.question)
            text_pair.append(retrieved_block.decode())

        concat_inputs = tokenizer(text, text_pair, return_tensors='pt', padding=True, truncation=True)

        output = reader(
            **concat_inputs,
            relevance_score=searcher_output.retrieved_logits,
            return_dict=True,
        )

    answer = tokenizer.decode(concat_inputs.input_ids[output.block_idx][output.start_pos: output.end_pos + 1])

    print(output)

    return output, answer

def main(args):
    if args.from_pt_pretrained:
        searcher, reader, tokenizer = get_searcher_reader_tokenizer_pt_pretrained(args)
    elif args.from_pt_finetuned:
        searcher, reader, tokenizer = get_searcher_reader_tokenizer_pt_finetuned(args)
    else:
        searcher, reader, tokenizer = get_searcher_reader_tokenizer_tf(args)

    retriever_output = retrieve(args, searcher, tokenizer)
    reader_output, answer = read(args, reader, tokenizer, retriever_output)

    print(f"Question: {args.question}\nAnswer: {answer}")

    if args.saved_path is not None:
        searcher.save_pretrained(os.path.join(args.saved_path, "searcher/"))
        reader.save_pretrained(os.path.join(args.saved_path, "reader/"))

    return answer

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)