import os

from datasets import load_dataset

def load_nq(args):
    def filter_fn(example):
        if len(example['annotations.short_answers'][0]['text']) != 0 and \
            example['annotations.yes_no_answer'] == [-1]:
            return True
        return False
    dataset = load_dataset("natural_questions", cache_dir=os.path.abspath(args.dataset_cache_dir))
    training_dataset = dataset['train'].remove_columns(['id', 'document']).flatten()
    eval_dataset = dataset['validation'].remove_columns(['id', 'document']).flatten()
    
    training_dataset = training_dataset.filter(filter_fn)
    eval_dataset = eval_dataset.filter(filter_fn)

    return training_dataset, eval_dataset

