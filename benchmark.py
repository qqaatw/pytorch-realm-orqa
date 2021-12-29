import torch
from argparse import ArgumentParser
from transformers import RealmConfig
from tqdm import tqdm

from data import load as load_dataset
from data import DataCollator
from run_finetune import compute_eval_metrics
from model import get_openqa

def get_arg_parser():
    parser = ArgumentParser()

    # Data
    parser.add_argument("--dataset_name_path", type=str, default=r"natural_questions",
        choices=["natural_questions", "web_questions"])
    parser.add_argument("--dataset_cache_dir", type=str, default=r"./data/dataset_cache_dir/")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--max_answer_tokens", type=int, default=5)

    # Model
    parser.add_argument("--block_records_path", type=str, default=r"./data/enwiki-20181220/blocks.tfr")
    parser.add_argument("--retriever_pretrained_name", type=str, default=r"qqaatw/realm-orqa-nq-searcher")
    parser.add_argument("--checkpoint_pretrained_name", type=str, default=r"qqaatw/realm-orqa-nq-reader")

    # Config
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_scann", action="store_true")

    return parser

def main(args):
    config = RealmConfig(searcher_beam_size=10, use_scann=args.use_scann)
    
    openqa = get_openqa(args, config=config)
    openqa.to(args.device)

    # Setup data
    _, _, eval_dataset = load_dataset(args)
    data_collector = DataCollator(args, openqa.tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collector
    )
    print(eval_dataset)

    all_metrics = []
    for batch in tqdm(eval_dataloader):
        question, answer_texts, answer_ids = batch
        with torch.no_grad():
            openqa_output = openqa(question, answer_ids, return_dict=True)
            all_metrics.append(compute_eval_metrics(answer_texts, openqa_output.predicted_answer, openqa_output.reader_output))


    stacked_metrics = { 
        metric_key : torch.stack((*map(lambda metrics: metrics[metric_key], all_metrics),)) for metric_key in all_metrics[0].keys()
    }

    print('\n'.join(map(lambda metric: f"{metric[0]}:{metric[1].type(torch.float32).mean()}", stacked_metrics.items())))

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)