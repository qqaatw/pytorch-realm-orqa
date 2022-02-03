import os
import unittest

import predictor, run_finetune


class Tester(unittest.TestCase):
    def setUp(self):
        self.pt_checkpoint_path = "google/realm-orqa-nq-openqa"

    def test_predictor(self):
        parser = predictor.get_arg_parser()
        args = parser.parse_args(["--question", "Who is the pioneer in modern computer science?"])
        answer = predictor.main(args)

        self.assertEqual(answer, "alan mathison turing")
    
    def test_predictor_with_additional_documents(self):
        parser = predictor.get_arg_parser()
        args = parser.parse_args([
            "--question", "What is the previous name of Meta Platform, Inc.?",
            "--additional_documents_path", "additional_documents.npy"
        ])
        answer = predictor.main(args)

        self.assertEqual(answer, "facebook, inc.")

    def test_finetune(self):
        return
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args(["--is_train", "--num_training_steps", "1"])
        run_finetune.main(args)

    def test_finetune_eval(self):
        return
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args([
            "--checkpoint_pretrained_name", self.pt_checkpoint_path,])
        run_finetune.main(args)