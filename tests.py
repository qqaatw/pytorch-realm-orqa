import unittest
import torch
import predictor, run_finetune


class Tester(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pt_checkpoint_pretrained_name = "./export_nq_newqa/realm-orqa-nq-openqa"
        
        self.model_dir = "./"
        self.pt_checkpoint_name = "checkpoint"
        self.pt_checkpoint_step = 158330

    def test_predictor(self):
        parser = predictor.get_arg_parser()
        args = parser.parse_args([
            "--question", "Who is the pioneer in modern computer science?",
            "--checkpoint_pretrained_name", self.pt_checkpoint_pretrained_name,
        ])
        answer = predictor.main(args)

        self.assertEqual(answer, "alan mathison turing")
    
    def test_predictor_with_additional_documents(self):
        parser = predictor.get_arg_parser()
        args = parser.parse_args([
            "--question", "What is the previous name of Meta Platform, Inc.?",
            "--checkpoint_pretrained_name", self.pt_checkpoint_pretrained_name,
            "--additional_documents_path", "additional_documents.npy",
        ])
        answer = predictor.main(args)

        self.assertEqual(answer, "facebook, inc.")

    def test_finetune(self):
        return
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args(["--is_train", "--num_training_steps", "1"])
        run_finetune.main(args)

    def test_finetune_eval(self):
        parser = run_finetune.get_arg_parser()
        args = parser.parse_args([
            "--model_dir", self.model_dir,
            "--checkpoint_name", self.pt_checkpoint_name,
            "--checkpoint_step", str(self.pt_checkpoint_step),
            "--device", self.device,
        ])
        run_finetune.main(args)