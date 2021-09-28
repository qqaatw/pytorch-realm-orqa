from argparse import ArgumentParser
import os
import logging


import torch
import itertools
from datasets import load_dataset
from datasets.utils import DownloadConfig
from transformers import RealmConfig, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from transformers.models.realm.modeling_realm import logger as model_logger

from model import get_searcher_reader_tokenizer
from data import load_nq

model_logger.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.set_printoptions(precision=8)


class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        example = batch[0]
        question = example["question.text"]
        answer_text = example["annotations.short_answers"][0]["text"]
        if len(answer_text) != 0:
            answers = self.tokenizer(answer_text, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False).input_ids
        else:
            answers = [[-1]]
        return question, answers
    

def retrieve(args, searcher, tokenizer, question):
    question_ids = tokenizer([question], return_tensors='pt')
    output = searcher(**question_ids.to(args.device), return_dict=True)

    return output

def read(args, reader, tokenizer, searcher_output, question, answers):
    def block_has_answer(concat_inputs, answers):
        """check if retrieved_blocks has answers."""
        has_answers = []
        start_pos = []
        end_pos = []
        max_answers = 0

        for input_id in concat_inputs.input_ids:
            pass_sep = False
            answer_pos = 0
            start=-1
            start_pos.append([])
            end_pos.append([])
            for answer in answers:
                for idx, id in enumerate(input_id):
                    if id == tokenizer.sep_token_id:
                        pass_sep = True
                    if not pass_sep:
                        continue
                    if answer[answer_pos] == id:
                        if start == -1:
                            start = idx
                        if answer_pos == len(answer) - 1:
                            start_pos[-1].append(start)
                            end_pos[-1].append(idx)
                            answer_pos = 0
                            start = -1
                            break
                        else:
                            answer_pos += 1
                    else:
                        answer_pos = 0
                        start = -1
            
            if len(start_pos[-1]) == 0:
                start_pos[-1].append(-1)
                end_pos[-1].append(-1)
                has_answers.append(False)
            else:
                has_answers.append(True)
            if len(start_pos[-1]) > max_answers:
                max_answers = len(start_pos[-1])

        for start_pos_, end_pos_ in zip(start_pos, end_pos):
            while len(start_pos_) < max_answers:
                start_pos_.append(-1)
            while len(end_pos_) < max_answers:
                end_pos_.append(-1)

        assert len(has_answers) == len(start_pos) == len(end_pos)

        return (
            torch.tensor(has_answers, dtype=torch.bool),
            torch.tensor(start_pos, dtype=torch.int64),
            torch.tensor(end_pos, dtype=torch.int64),
        )

    text = []
    text_pair = []
    for retrieved_block in searcher_output.retrieved_blocks:
        text.append(question)
        text_pair.append(retrieved_block.decode())

    concat_inputs = tokenizer(text, text_pair, return_tensors='pt', padding=True, truncation=True, max_length=reader.config.reader_seq_len)

    has_answers, start_positions, end_positions = block_has_answer(concat_inputs, answers)

    output = reader(
        input_ids=concat_inputs.input_ids[0: reader.config.reader_beam_size].to(args.device),
        attention_mask=concat_inputs.attention_mask[0: reader.config.reader_beam_size].to(args.device),
        token_type_ids=concat_inputs.token_type_ids[0: reader.config.reader_beam_size].to(args.device),
        relevance_score=searcher_output.retrieved_logits.to(args.device),
        has_answers=has_answers.to(args.device),
        start_positions=start_positions.to(args.device),
        end_positions=end_positions.to(args.device),
        return_dict=True,
    )

    answer = tokenizer.decode(concat_inputs.input_ids[output.block_idx][output.start_pos: output.end_pos + 1])

    return output, answer


def main(args):
    config = RealmConfig(searcher_beam_size=10)
    training_dataset, eval_dataset = load_nq(args)

    searcher, reader, tokenizer = get_searcher_reader_tokenizer(args, config)

    if args.is_train:
        # Setup training mode
        searcher.train()
        searcher.to(args.device)
        reader.train()
        reader.to(args.device)


        # Setup data
        print(training_dataset)
        data_collector = DataCollator(tokenizer)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=data_collector
        )

        optimizer = AdamW(
            itertools.chain(searcher.parameters(), reader.parameters()),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(10000, max(100,
                                        int(args.epochs * len(train_dataloader) / 10))),
            num_training_steps=args.epochs * len(train_dataloader),
        )

        global_step = 0

        for epoch in range(1, args.epochs + 1):
            for batch in train_dataloader:
                question, answers = batch
                retriever_output = retrieve(args, searcher, tokenizer, question)
                reader_output, predicted_answer = read(args, reader, tokenizer, retriever_output, question, answers)

                reader_output.loss.backward()
                optimizer.step()
                lr_scheduler.step()

                logging.info(f"Step: {optimizer._step_count} Retriever Loss: {reader_output.retriever_loss.mean()} Reader Loss: {reader_output.reader_loss.mean()}\nQuestion: {question} Gold Answer: {tokenizer.batch_decode(answers) if answers != [[-1]] else None} Predicted Answer: {predicted_answer}")
        
        if epoch % args.chpt_interval == 0:
            torch.save(
                {
                    'searcher_state_dict': searcher.state_dict(),
                    'reader_state_dict': reader.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                os.path.join(args.model_dir, "checkpoint.bin")
            )

    else:
        searcher.eval()
        reader.eval()
        data_collector = DataCollator(tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=data_collector
        )
        


if __name__ == "__main__":
    parser = ArgumentParser()
    
    logging.info("Test logging")

    # Training dir
    parser.add_argument("--dataset_name_path", type=str, default=r"natural_questions")
    parser.add_argument("--dataset_cache_dir", type=str, default=r"./data/dataset_cache_dir/")
    parser.add_argument("--model_dir", type=str, default=r"./")

    # Training hparams
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)

    # Retriever
    parser.add_argument("--block_emb_path", type=str, default=r"./data/cc_news_pretrained/embedder/encoded/encoded.ckpt")
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--retriever_path", type=str, default=r"./data/cc_news_pretrained/embedder/variables/variables")
    # Reader
    parser.add_argument("--checkpoint_path", type=str, default=r"./data/orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000")

    args = parser.parse_args()
    
    main(args)