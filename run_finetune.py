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

model_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler('fine-tuning_test.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

torch.set_printoptions(precision=8)

MAX_EPOCHS = 2


def get_arg_parser():
    parser = ArgumentParser()

    # Data processing
    parser.add_argument("--dev_ratio", type=float, default=0.1,
        help="The ratio of development set which will be splitted from training set.")
    parser.add_argument("--max_answer_tokens", type=int, default=5,
        help="Answers below max_answer_tokens will be used for training and evaluation.")

    # Training dir
    parser.add_argument("--dataset_name_path", type=str, default=r"natural_questions", 
        help="Dataset name or path. Currently available datasets: natural_questions and web_questions. See data.py for more details.")
    parser.add_argument("--dataset_cache_dir", type=str, default=r"./data/dataset_cache_dir/",
        help="Directory storing dataset caches.")
    parser.add_argument("--model_dir", type=str, default=r"./",
        help="Directory storing resulting models. ")

    # Training hparams
    parser.add_argument("--ckpt_interval", type=int, default=5000,
        help="Number of steps the checkpoint will be saved.")
    parser.add_argument("--device", type=str, default='cpu',
        help="Device used for training and evaluation.")
    parser.add_argument("--is_train", action="store_true",
        help="If specified, training mode is set; otherwise, evaluation mode is set.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
        help="Learning rate.")
    parser.add_argument("--searcher_beam_size", type=int, default=5000,
        help="Searcher (Retriever) beam size.")
    parser.add_argument("--reader_beam_size", type=int, default=5,
        help="Reader beam size.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num_training_steps", type=int, default=100,
        help="Number of training steps.")
    group.add_argument("--num_epochs", type=int, default=0,
        help="Number of training epochs.")

    # Evaluation hparams
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint",
        help="Checkpoint name for evalutaion.")
    parser.add_argument("--checkpoint_step", type=int, default=5000,
        help="Checkpoint step for evalutaion.")

    # Model path
    parser.add_argument("--checkpoint_pretrained_name", type=str, default=r"qqaatw/realm-cc-news-pretrained-openqa",
        help="Pretrained checkpoint for fine-tuning.")

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
    config = RealmConfig(
        searcher_beam_size=args.searcher_beam_size,
        reader_beam_size=args.reader_beam_size,
    )

    openqa = get_openqa(args, config)
    retriever = openqa.retriever
    tokenizer = openqa.retriever.tokenizer

    training_dataset, dev_dataset, eval_dataset = load_dataset(args)

    # Optimizer
    # See: https://github.com/huggingface/transformers/blob/e239fc3b0baf1171079a5e0177a69254350a063b/examples/pytorch/language-modeling/run_mlm_no_trainer.py#L456-L468
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in openqa.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in openqa.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.is_train:
        global_step = 1
        starting_epoch = 1

        openqa.to(args.device)

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=0.01,
            eps=1e-6,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(10000, max(100,
                                        int(args.num_training_steps / 10))),
            num_training_steps=args.num_training_steps,
        )

        # Setup data
        logging.info(training_dataset)
        logging.info(dev_dataset)

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
                    openqa.save_pretrained(os.path.join(args.model_dir, f"{args.checkpoint_name}-{global_step}"))

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

        logging.info(f"Saving final checkpint at step {global_step}")
        openqa.save_pretrained(os.path.join(args.model_dir, f"{args.checkpoint_name}-{global_step}"))
    else:
        openqa = openqa.from_pretrained(os.path.join(args.model_dir, f"{args.checkpoint_name}-{args.checkpoint_step}"), retriever)
        
        # Setup eval mode
        openqa.eval()
        openqa.to(args.device)

        # Setup data
        logging.info(eval_dataset)
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