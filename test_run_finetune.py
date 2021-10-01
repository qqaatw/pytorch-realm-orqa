import unittest

import run_finetune


class FinetuneTester(unittest.TestCase):
    def test_finetune(self):
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args(["--is_train", "--num_training_steps", "20"])
        run_finetune.main(args)

    def test_eval(self):
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args(["--retriever_pretrained_name", "retriever", "--checkpoint_pretrained_name", "reader"])
        run_finetune.main(args)