from argparse import ArgumentParser

import torch
from transformers.models.realm.modeling_realm import logger
from transformers.utils import logging

from model import get_searcher_reader_tokenizer_tf

logger.setLevel(logging.INFO)
torch.set_printoptions(precision=8)

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
    searcher, reader, tokenizer = get_searcher_reader_tokenizer_tf(args)
    
    retriever_output = retrieve(args, searcher, tokenizer)
    reader_output, answer = read(args, reader, tokenizer, retriever_output)

    print(f"Question: {args.question}\nAnswer: {answer}")

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Question
    parser.add_argument("--question", type=str, required=True)
    # Retriever
    parser.add_argument("--block_emb_path", type=str, default=r"./data/cc_news_pretrained/embedder/encoded/encoded.ckpt")
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--retriever_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000")
    # Reader
    parser.add_argument("--checkpoint_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000")

    args = parser.parse_args()
    
    main(args)