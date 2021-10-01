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

def load_nq(args, tokenizer):
    """Load Natural Questions."""
    def filter_fn(example):
        """Remove answers having length less than 5."""
        for short_answer in example['annotations.short_answers']:
            if len(short_answer) != 0:
                for answer in short_answer['text']:
                    len_answer = len(tokenizer.encode(normalize_answer(answer), add_special_tokens=False))
                    if len_answer > 0 and len_answer <= args.max_answer_tokens:
                        return True
        return False

    dataset = load_dataset(args.dataset_name_path, cache_dir=os.path.abspath(args.dataset_cache_dir))

    # Remove unused columns and flatten structure.
    training_dev_dataset = dataset['train'].train_test_split(test_size=args.dev_ratio, shuffle=False)
    training_dataset = training_dev_dataset['train'].remove_columns(['id', 'document']).flatten()
    dev_dataset = training_dev_dataset['test'].remove_columns(['id', 'document']).flatten()
    eval_dataset = dataset['validation'].remove_columns(['id', 'document']).flatten()
    
    # Perform filtering
    filtered_training_dataset = training_dataset.filter(filter_fn)
    filtered_dev_dataset = dev_dataset.filter(filter_fn)
    filtered_eval_dataset = eval_dataset.filter(filter_fn)

    # An exmaple of each dataset should contain the following columns:
    # example["question.text"]
    # example["annotations.short_answers"][num_answers]["text"]
    return filtered_training_dataset, filtered_dev_dataset, filtered_eval_dataset