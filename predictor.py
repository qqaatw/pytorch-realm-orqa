from argparse import ArgumentParser
import os

import torch

from model import get_openqa
from transformers.models.realm.modeling_realm import logger
from transformers.utils import logging

logger.setLevel(logging.INFO)
torch.set_printoptions(precision=8)


def get_arg_parser():
    parser = ArgumentParser()

    # Question
    parser.add_argument("--question", type=str, required=True,
        help="Input question.")
    # Retriever
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--checkpoint_pretrained_name", type=str, default=r"qqaatw/realm-orqa-nq-openqa")

    return parser

def main(args):
    openqa = get_openqa(args)
    tokenizer = openqa.retriever.tokenizer

    question_ids = tokenizer(args.question, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = openqa(
            input_ids=question_ids,
            return_dict=True,
        )

    predicted_answer = tokenizer.decode(outputs.predicted_answer_ids)

    print(f"Question: {args.question}\nAnswer: {predicted_answer}")

    return predicted_answer

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)