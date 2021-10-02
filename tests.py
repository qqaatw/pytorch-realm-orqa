import os
import unittest

import predictor, run_finetune


class Tester(unittest.TestCase):
    def setUp(self):
        self.pt_searcher_path = "export/searcher/" if os.path.isdir("export/searcher/") else "qqaatw/realm-orqa-nq-searcher"
        self.pt_checkpoint_path = "export/reader/" if os.path.isdir("export/reader/") else "qqaatw/realm-orqa-nq-reader"

    def test_predictor(self):
        # Test loading from tf checkpoints
        parser = predictor.get_arg_parser()
        args = parser.parse_args(["--question", "Who is the pioneer in modern computer science?"])
        answer = predictor.main(args)

        self.assertEqual(answer, "alan mathison turing")

        # Test loading from pt checkopoints
        parser = predictor.get_arg_parser()
        args = parser.parse_args([
            "--from_pt_finetuned",
            "--question", "Who is the pioneer in modern computer science?", 
            "--retriever_pretrained_name", self.pt_searcher_path,
            "--checkpoint_pretrained_name", self.pt_checkpoint_path,])
        answer = predictor.main(args)
        
        self.assertEqual(answer, "alan mathison turing")

    def test_finetune(self):
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args(["--is_train", "--num_training_steps", "1"])
        run_finetune.main(args)

    def test_finetune_eval(self):
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args([
            "--retriever_pretrained_name", self.pt_searcher_path,
            "--checkpoint_pretrained_name", self.pt_checkpoint_path,])
        run_finetune.main(args)