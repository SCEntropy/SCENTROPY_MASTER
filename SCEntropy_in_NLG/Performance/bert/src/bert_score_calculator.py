from transformers import AutoTokenizer, AutoModel
from bert_score import score
import os
import torch
from typing import Tuple


class BERTScoreCalculator:
    """
    BERTScore Calculator class
    Used to calculate BERTScore (Precision, Recall, F1) between reference and generated texts
    """
    
    def __init__(self, model_type: str = "bert-base-uncased", local_model_path: str = None):
        """
        Initialize BERTScore calculator
        
        Args:
            model_type: Model type, default is "bert-base-uncased"
            local_model_path: Local model path, if provided use local model
        """
        if local_model_path and os.path.exists(local_model_path):
            self.model_type = local_model_path
        else:
            self.model_type = model_type
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def download_model_if_needed(self):
        """
        Automatically download model if it doesn't exist
        """
        try:
            # Try to load tokenizer and model to check if they exist
            tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            model = AutoModel.from_pretrained(self.model_type)
        except Exception as e:
            raise
    
    def compute_bertscore(self, candidates: list, references: list) -> Tuple[float, float, float]:
        """
        Calculate BERTScore (Precision, Recall, F1)
        
        Args:
            candidates: List of candidate texts
            references: List of reference texts
            
        Returns:
            tuple: Average values of (Precision, Recall, F1)
        """
        if len(candidates) != len(references):
            raise ValueError("The number of candidate texts and reference texts must be the same")
        
        P, R, F1 = score(
            cands=candidates,
            refs=references,
            model_type=self.model_type,
            num_layers=AutoModel.from_pretrained(self.model_type).config.num_hidden_layers,
            lang="en",
            rescale_with_baseline=False,
            device=self.device
        )
        
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def calculate_scores(self, reference: str, unconstrained_candidate: str, constrained_candidate: str) -> dict:
        """
        Calculate BERTScore for unconstrained and constrained generated texts
        
        Args:
            reference: Reference text
            unconstrained_candidate: Unconstrained generated text
            constrained_candidate: Constrained generated text
            
        Returns:
            dict: Dictionary containing all scores
        """
        # Calculate BERTScore for unconstrained generation
        unconstrained_scores = self.compute_bertscore([unconstrained_candidate], [reference])
        
        # Calculate BERTScore for constrained generation
        constrained_scores = self.compute_bertscore([constrained_candidate], [reference])
        
        return {
            "reference": reference,
            "unconstrained": {
                "text": unconstrained_candidate,
                "precision": unconstrained_scores[0],
                "recall": unconstrained_scores[1],
                "f1": unconstrained_scores[2]
            },
            "constrained": {
                "text": constrained_candidate,
                "precision": constrained_scores[0],
                "recall": constrained_scores[1],
                "f1": constrained_scores[2]
            }
        }