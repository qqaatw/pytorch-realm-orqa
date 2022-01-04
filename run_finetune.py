import itertools
import logging
import os
from argparse import ArgumentParser

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data import DataCollator, normalize_answer
from data import load as load_dataset
from model import get_openqa
from transformers import RealmConfig, get_linear_schedule_with_warmup
from transformers.models.realm.modeling_realm import logger as model_logger
from transformers.optimization import AdamW

model_logger.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.set_printoptions(precision=8)

MAX_EPOCHS = 100


def get_arg_parser():
    parser = ArgumentParser()

    parser.add_argument("--benchmark", action="store_true")

    # Data processing
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--max_answer_tokens", type=int, default=5)

    # Training dir
    parser.add_argument("--dataset_name_path", type=str, default=r"natural_questions")
    parser.add_argument("--dataset_cache_dir", type=str, default=r"./data/dataset_cache_dir/")
    parser.add_argument("--model_dir", type=str, default=r"./")

    # Training hparams
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--searcher_beam_size", type=int, default=5000)
    parser.add_argument("--reader_beam_size", type=int, default=5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num_training_steps", type=int, default=100)
    group.add_argument("--num_epochs", type=int, default=0)

    # Evaluation hparams
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pt")

    # Model path
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--checkpoint_pretrained_name", type=str, default=r"qqaatw/realm-cc-news-pretrained-openqa")

    return parser

def compute_eval_metrics(labels, predicted_answer, reader_output):
    """Compute eval metrics."""
    # []
    exact_match = torch.index_select(
        torch.index_select(
            reader_output.reader_correct,
            dim=0,
            index=reader_output.block_idx
        ),
        dim=1,
        index=reader_output.candidate
    )

    def _official_exact_match(predicted_answer, references):
        return torch.tensor(max(
            [normalize_answer(predicted_answer) == normalize_answer(reference) for reference in references]
        ))

    official_exact_match = _official_exact_match(predicted_answer, labels)

    eval_metric = dict(
        exact_match=exact_match[0][0],
        official_exact_match=official_exact_match,
        reader_oracle=torch.any(reader_output.reader_correct)
    )
    
    for k in (5, 10, 50, 100, 500, 1000, 5000):
        eval_metric["top_{}_match".format(k)] = torch.any(reader_output.retriever_correct[:k])
    return eval_metric

def main(args):
    config = RealmConfig(searcher_beam_size=10)
    if args.resume:
        searcher, reader, tokenizer = get_openqa(args, config)
    else:
        openqa = get_openqa(args, config)

    training_dataset, dev_dataset, eval_dataset = load_dataset(args)

    optimizer = AdamW(
        openqa.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(10000, max(100,
                                    int(args.num_training_steps / 10))),
        num_training_steps=args.num_training_steps,
    )

    if args.is_train:
        if args.resume:
            checkpoint = torch.load(os.path.join(args.model_dir, "checkpoint.pt"), map_location='cpu')
            openqa.load_state_dict(checkpoint["openqa_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            global_step = checkpoint["training_step"]
            starting_epoch = checkpoint["training_epoch"]
        else:
            global_step = 1
            starting_epoch = 1

        openqa.to(args.device)

        # Setup data
        print(training_dataset)
        tokenizer = openqa.retriever.tokenizer
        data_collector = DataCollator(args, tokenizer)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=data_collector
        )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=data_collector
        )

        if args.num_epochs == 0:
            args.num_epochs = MAX_EPOCHS
        else:
            args.num_training_steps = args.num_epochs * len(train_dataloader)

        for epoch in range(starting_epoch, args.num_epochs + 1):

            # Setup training mode
            openqa.train()

            for batch in train_dataloader:
                optimizer.zero_grad()
                question, answer_texts, answer_ids = batch
                
                question_ids = tokenizer(question, return_tensors="pt").input_ids
                reader_output, predicted_answer_ids = openqa(
                    input_ids=question_ids.to(args.device),
                    answer_ids=answer_ids,
                    return_dict=False,
                )

                predicted_answer = tokenizer.decode(predicted_answer_ids)

                reader_output.loss.backward()
                clip_grad_norm_(openqa.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=False)

                optimizer.step()
                lr_scheduler.step()

                logging.info(
                    f"Epoch: {epoch}, Step: {global_step}, Retriever Loss: {reader_output.retriever_loss.mean()}, Reader Loss: {reader_output.reader_loss.mean()}\nQuestion: {question}, Gold Answer: {tokenizer.batch_decode(answer_ids) if answer_ids != [[-1]] else None}, Predicted Answer: {predicted_answer}"
                )

                if global_step % args.ckpt_interval == 0:
                    logging.info(f"Saving checkpint at step {global_step}")
                    torch.save(
                        {
                            'state_dict': openqa.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'training_step': global_step,
                            'training_epoch': epoch,
                        }, 
                        os.path.join(args.model_dir, "checkpoint.pt")
                    )

                global_step += 1
                if global_step >= args.num_training_steps:
                    break
            
            # Setup eval mode
            openqa.eval()
            all_metrics = []

            for batch in tqdm(eval_dataloader):
                question, answer_texts, answer_ids = batch
                
                question_ids = tokenizer(question, return_tensors="pt").input_ids
                with torch.no_grad():
                    outputs = openqa(
                        input_ids=question_ids.to(args.device),
                        answer_ids=answer_ids,
                        return_dict=True,
                    )

                predicted_answer = tokenizer.decode(outputs.predicted_answer_ids)
                all_metrics.append(compute_eval_metrics(answer_texts, predicted_answer, outputs.reader_output))

            stacked_metrics = { 
                metric_key : torch.stack((*map(lambda metrics: metrics[metric_key], all_metrics),)) for metric_key in all_metrics[0].keys()
            }
            logging.info(f"Step: {global_step}, Epoch: {epoch}")
            logging.info('\n'.join(map(lambda metric: f"{metric[0]}:{metric[1].type(torch.float32).mean()}", stacked_metrics.items())))
            
            if global_step >= args.num_training_steps:
                break

        openqa.save_pretrained(os.path.join(args.model_dir, "openqa/"))

    else:
        # Setup eval mode
        openqa.eval()
        openqa.to(args.device)

        # Setup data
        print(eval_dataset)
        data_collector = DataCollator(args, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=data_collector
        )

        all_metrics = []
        for batch in tqdm(eval_dataloader):
            question, answer_texts, answer_ids = batch
            question_ids = tokenizer(question, return_tensors="pt").input_ids

            with torch.no_grad():
                outputs = openqa(
                    input_ids=question_ids.to(args.device),
                    answer_ids=answer_ids,
                    return_dict=True,
                )

            predicted_answer = tokenizer.decode(outputs.predicted_answer_ids)
            all_metrics.append(compute_eval_metrics(answer_texts, predicted_answer, outputs.reader_output))


        stacked_metrics = { 
            metric_key : torch.stack((*map(lambda metrics: metrics[metric_key], all_metrics),)) for metric_key in all_metrics[0].keys()
        }

        logging.info('\n'.join(map(lambda metric: f"{metric[0]}:{metric[1].type(torch.float32).mean()}", stacked_metrics.items())))
        


if __name__ == "__main__":
    logging.info("Test logging")

    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)