import os
import re
import string
import unicodedata

from datasets import load_dataset


def normalize_answer(s):
    """Normalize answer. (Directly copied from ORQA codebase)"""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load(args):
    """Load dataset"""
    if os.path.isdir(args.dataset_name_path):
        raise ValueError("Dataset path currently not supported.")

    if args.dataset_name_path == "natural_questions":
        return load_nq(args)
    elif args.dataset_name_path == "web_questions":
        return load_wq(args)
    else:
        raise ValueError("Invalid dataset name or path")
    
def load_nq(args):
    """Load NaturalQuestions."""
    def filter_fn(example):
        """Remove answers having length more than 5."""
        for short_answer in example['annotations.short_answers']:
            if len(short_answer) != 0:
                for i in range(len(short_answer['text'])):
                    if short_answer['end_token'][i] - short_answer['start_token'][i] <= args.max_answer_tokens:
                        return True
        return False

    def map_fn(example):
        """Unify dataset structures."""
        return {
            "question": example["question.text"], 
            "answers": [answer["text"] for answer in example["annotations.short_answers"]]
        }

    dataset = load_dataset(args.dataset_name_path, cache_dir=os.path.abspath(args.dataset_cache_dir))

    # Remove unused columns and flatten structure.
    training_dev_dataset = dataset['train'].train_test_split(test_size=args.dev_ratio, shuffle=False)
    training_dataset = training_dev_dataset['train'].remove_columns(['id', 'document']).flatten()
    dev_dataset = training_dev_dataset['test'].remove_columns(['id', 'document']).flatten()
    eval_dataset = dataset['validation'].remove_columns(['id', 'document']).flatten()
    
    # Perform filtering and mapping
    filtered_training_dataset = training_dataset.filter(filter_fn).map(map_fn)
    filtered_dev_dataset = dev_dataset.filter(filter_fn).map(map_fn)
    filtered_eval_dataset = eval_dataset.filter(filter_fn).map(map_fn)

    # An exmaple of each dataset should contain the following columns:
    # example["question"]
    # example["answers"][num_answers]
    return filtered_training_dataset, filtered_dev_dataset, filtered_eval_dataset

def load_wq(args):
    """Load WebQuestions(WQ)."""
    dataset = load_dataset(args.dataset_name_path, cache_dir=os.path.abspath(args.dataset_cache_dir))

    # Remove unused columns and flatten structure.
    training_dev_dataset = dataset['train'].train_test_split(test_size=args.dev_ratio, shuffle=False)
    training_dataset = training_dev_dataset['train'].remove_columns(['url'])
    dev_dataset = training_dev_dataset['test'].remove_columns(['url'])
    eval_dataset = dataset['test'].remove_columns(['url'])
    
    # No need to filter
    filtered_training_dataset = training_dataset
    filtered_dev_dataset = dev_dataset
    filtered_eval_dataset = eval_dataset

    # An exmaple of each dataset should contain the following columns:
    # example["question"]
    # example["answers"][num_answers]
    return filtered_training_dataset, filtered_dev_dataset, filtered_eval_dataset


class DataCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
    def __call__(self, batch):
        example = batch[0]
        question = example["question"]
        answer_texts = []
        for answer in example["answers"]:
            answer_texts += [answer] if isinstance(answer, str) else answer
        answer_texts = list(set(answer_texts))
        if len(answer_texts) != 0:
            answer_ids = self.tokenizer(
                answer_texts, 
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            ).input_ids
        else:
            answer_ids = [[-1]]
        return question, answer_texts, answer_ids