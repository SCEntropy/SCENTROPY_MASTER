"""
METEOR Score Calculator Module

A comprehensive tool for evaluating text generation models using METEOR (Metric for Evaluation of Translation with Explicit ORdering),
which is a metric that accounts for unigram matching, stemming, synonymy, and word order.
"""

import re
import string
from collections import Counter
import math
from typing import List, Dict, Tuple, Union, Optional


class Tokenizer:
    """
    A flexible tokenizer class that supports different tokenization strategies
    """
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True, remove_empty: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_empty = remove_empty
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with configurable options
        
        Args:
            text (str): Input text to tokenize
        
        Returns:
            List[str]: List of tokens
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        tokens = text.split()
        
        if self.remove_empty:
            tokens = [token for token in tokens if token.strip()]
        
        return tokens


class METEORScoreConfig:
    """
    Configuration class for METEOR score calculation parameters
    """
    def __init__(self, 
                 alpha: float = 0.9, 
                 beta: float = 3.0, 
                 gamma: float = 0.5,
                 output_file: str = "meteor_results.json"):
        """
        Initialize METEOR score configuration
        
        Args:
            alpha (float): Parameter for precision/recall balance
            beta (float): Parameter for F-measure
            gamma (float): Parameter for penalty
            output_file (str): Output file path for results
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.output_file = output_file


class METEORScoreResult:
    """
    Class to hold METEOR score results
    """
    def __init__(self, 
                 reference: str, 
                 hypothesis: str, 
                 score: float):
        """
        Initialize METEOR score result
        
        Args:
            reference (str): Reference sentence
            hypothesis (str): Hypothesis sentence
            score (float): METEOR score
        """
        self.reference = reference
        self.hypothesis = hypothesis
        self.score = score


class METEORScoreCalculator:
    """
    Main class for calculating METEOR scores with configurable parameters
    """
    def __init__(self, config: Optional[METEORScoreConfig] = None):
        """
        Initialize METEOR score calculator
        
        Args:
            config (METEORScoreConfig, optional): Configuration object with parameters
        """
        self.config = config or METEORScoreConfig()
        self.tokenizer = Tokenizer()
    
    def calculate_score(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score between reference and hypothesis
        
        Args:
            reference (str): Reference sentence
            hypothesis (str): Hypothesis sentence
        
        Returns:
            float: METEOR score between 0 and 1
        """
        ref_tokens = self.tokenizer.tokenize(reference)
        hyp_tokens = self.tokenizer.tokenize(hypothesis)
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        # Calculate exact matches
        common = Counter(ref_tokens) & Counter(hyp_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = num_common / len(hyp_tokens)
        recall = num_common / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        # Calculate F-measure
        f_mean = (precision * recall) / (self.config.alpha * recall + (1 - self.config.alpha) * precision)
        
        # Calculate penalty based on word order
        penalty = self._calculate_word_order_penalty(ref_tokens, hyp_tokens)
        
        # Calculate final METEOR score
        final_score = f_mean * (1 - self.config.gamma * penalty)
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_word_order_penalty(self, ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        """
        Calculate penalty based on word order differences
        
        Args:
            ref_tokens (List[str]): Tokenized reference sentence
            hyp_tokens (List[str]): Tokenized hypothesis sentence
        
        Returns:
            float: Penalty value between 0 and 1
        """
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        # Find common tokens
        common_tokens = set(ref_tokens) & set(hyp_tokens)
        if not common_tokens:
            return 1.0  # Maximum penalty if no common tokens
        
        # Create position mappings for common tokens
        ref_positions = {token: [] for token in common_tokens}
        hyp_positions = {token: [] for token in common_tokens}
        
        for i, token in enumerate(ref_tokens):
            if token in common_tokens:
                ref_positions[token].append(i)
        
        for i, token in enumerate(hyp_tokens):
            if token in common_tokens:
                hyp_positions[token].append(i)
        
        # Calculate chunks of consecutive matches
        ref_idx = 0
        hyp_idx = 0
        chunks = 0
        total_matches = 0
        
        while ref_idx < len(ref_tokens) and hyp_idx < len(hyp_tokens):
            if (ref_idx < len(ref_tokens) and 
                hyp_idx < len(hyp_tokens) and 
                ref_tokens[ref_idx] in common_tokens and 
                hyp_tokens[hyp_idx] == ref_tokens[ref_idx]):
                # Start of a chunk
                chunk_size = 0
                while (ref_idx < len(ref_tokens) and 
                       hyp_idx < len(hyp_tokens) and 
                       ref_tokens[ref_idx] == hyp_tokens[hyp_idx]):
                    chunk_size += 1
                    total_matches += 1
                    ref_idx += 1
                    hyp_idx += 1
                if chunk_size > 0:
                    chunks += 1
            else:
                ref_idx += 1
                hyp_idx += 1
        
        if total_matches == 0:
            return 1.0
        
        # Calculate penalty based on number of chunks
        # The more chunks, the more fragmented the alignment, the higher the penalty
        penalty = (chunks - 1) / total_matches if total_matches > 0 else 0
        return penalty
    
    def run_evaluation(self, 
                      reference: List[str], 
                      candidate_unconstrained: List[str], 
                      candidate_constrained: List[str]) -> Tuple[Dict, Dict]:
        """
        Run METEOR evaluation on lists of sentences
        
        Args:
            reference (List[str]): List of reference sentences
            candidate_unconstrained (List[str]): List of unconstrained candidate sentences
            candidate_constrained (List[str]): List of constrained candidate sentences
        
        Returns:
            Tuple[Dict, Dict]: Results for unconstrained and constrained candidates
        """
        if not (len(reference) == len(candidate_unconstrained) == len(candidate_constrained)):
            raise ValueError("All input lists must have the same length")
        
        unconstrained_scores = []
        constrained_scores = []
        
        for ref, unc, con in zip(reference, candidate_unconstrained, candidate_constrained):
            unc_score = self.calculate_score(ref, unc)
            con_score = self.calculate_score(ref, con)
            unconstrained_scores.append(unc_score)
            constrained_scores.append(con_score)
        
        # Calculate average scores
        avg_unconstrained_precision = sum(unconstrained_scores) / len(unconstrained_scores) if unconstrained_scores else 0
        avg_constrained_precision = sum(constrained_scores) / len(constrained_scores) if constrained_scores else 0
        
        unconstrained_result = {
            "scores": unconstrained_scores,
            "average_score": avg_unconstrained_precision
        }
        
        constrained_result = {
            "scores": constrained_scores,
            "average_score": avg_constrained_precision
        }
        
        return unconstrained_result, constrained_result


def meteor_score(reference: str, hypothesis: str, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """
    Calculate METEOR score between reference and hypothesis (standalone function)
    
    Args:
        reference (str): Reference sentence
        hypothesis (str): Hypothesis sentence
        alpha (float): Parameter for precision/recall balance
        beta (float): Parameter for F-measure
        gamma (float): Parameter for penalty
    
    Returns:
        float: METEOR score between 0 and 1
    """
    config = METEORScoreConfig(alpha=alpha, beta=beta, gamma=gamma)
    calculator = METEORScoreCalculator(config)
    return calculator.calculate_score(reference, hypothesis)