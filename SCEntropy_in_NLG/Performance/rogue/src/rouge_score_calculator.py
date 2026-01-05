"""
ROUGE Score Calculator Module

A comprehensive tool for evaluating text generation models using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics,
which are used to evaluate the quality of generated text by comparing it to reference text.
"""

from rouge_score import rouge_scorer
from typing import List, Dict, Any, Tuple
import statistics


class ROUGEScoreConfig:
    """
    Configuration class for ROUGE score calculation parameters
    """
    def __init__(self, 
                 rouge_types: List[str] = None,
                 use_stemmer: bool = True,
                 output_file: str = "rouge_results.json"):
        """
        Initialize ROUGE score configuration
        
        Args:
            rouge_types (List[str]): List of ROUGE types to calculate (e.g., ['rouge1', 'rouge2', 'rougeL'])
            use_stemmer (bool): Whether to use stemming in calculations
            output_file (str): Output file path for results
        """
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.use_stemmer = use_stemmer
        self.output_file = output_file


class ROUGEScoreResult:
    """
    Class to hold ROUGE score results
    """
    def __init__(self, 
                 reference: str, 
                 hypothesis: str, 
                 scores: Dict[str, Any]):
        """
        Initialize ROUGE score result
        
        Args:
            reference (str): Reference sentence
            hypothesis (str): Hypothesis sentence
            scores (Dict[str, Any]): ROUGE scores dictionary
        """
        self.reference = reference
        self.hypothesis = hypothesis
        self.scores = scores


class ROUGEScoreCalculator:
    """
    Main class for calculating ROUGE scores with configurable parameters
    """
    def __init__(self, config: ROUGEScoreConfig = None):
        """
        Initialize ROUGE score calculator
        
        Args:
            config (ROUGEScoreConfig, optional): Configuration object with parameters
        """
        self.config = config or ROUGEScoreConfig()
        self.scorer = rouge_scorer.RougeScorer(self.config.rouge_types, use_stemmer=self.config.use_stemmer)
    
    def calculate_score(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculate ROUGE scores between reference and hypothesis
        
        Args:
            reference (str): Reference sentence
            hypothesis (str): Hypothesis sentence
        
        Returns:
            Dict[str, Any]: Dictionary containing ROUGE scores
        """
        raw_scores = self.scorer.score(reference, hypothesis)
        formatted_scores = self.format_scores(raw_scores)
        return formatted_scores
    
    def format_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format ROUGE scores to a more readable dictionary
        
        Args:
            scores (Dict[str, Any]): Raw scores returned by the evaluator
        
        Returns:
            Dict[str, Any]: Formatted scores dictionary
        """
        formatted = {}
        for rouge_type, score_values in scores.items():
            formatted[rouge_type] = {
                'precision': round(score_values.precision * 100, 2),
                'recall': round(score_values.recall * 100, 2),
                'fmeasure': round(score_values.fmeasure * 100, 2)
            }
        return formatted
    
    def run_evaluation(self, 
                      reference: List[str], 
                      candidate: List[str]) -> Dict[str, Any]:
        """
        Run ROUGE evaluation on lists of sentences
        
        Args:
            reference (List[str]): List of reference sentences
            candidate (List[str]): List of candidate sentences
        
        Returns:
            Dict[str, Any]: Results dictionary with individual and average scores
        """
        if len(reference) != len(candidate):
            raise ValueError("Reference and candidate lists must have the same length")
        
        individual_scores = []
        all_rouge_types = {}
        
        for ref, cand in zip(reference, candidate):
            score = self.calculate_score(ref, cand)
            individual_scores.append({
                'reference': ref,
                'candidate': cand,
                'scores': score
            })
            
            # Collect scores for averaging
            for rouge_type in score:
                if rouge_type not in all_rouge_types:
                    all_rouge_types[rouge_type] = {
                        'precision': [],
                        'recall': [],
                        'fmeasure': []
                    }
                all_rouge_types[rouge_type]['precision'].append(score[rouge_type]['precision'])
                all_rouge_types[rouge_type]['recall'].append(score[rouge_type]['recall'])
                all_rouge_types[rouge_type]['fmeasure'].append(score[rouge_type]['fmeasure'])
        
        # Calculate average scores
        average_scores = {}
        for rouge_type, values in all_rouge_types.items():
            average_scores[rouge_type] = {
                'precision': round(statistics.mean(values['precision']), 2),
                'recall': round(statistics.mean(values['recall']), 2),
                'fmeasure': round(statistics.mean(values['fmeasure']), 2)
            }
        
        results = {
            'individual_scores': individual_scores,
            'average_scores': average_scores,
            'total_sentences': len(reference)
        }
        
        return results
    
    def compare_candidates(self, 
                          reference: str, 
                          candidate1: str, 
                          candidate2: str) -> Dict[str, Any]:
        """
        Compare two candidate sentences against the same reference
        
        Args:
            reference (str): Reference sentence
            candidate1 (str): First candidate sentence
            candidate2 (str): Second candidate sentence
        
        Returns:
            Dict[str, Any]: Comparison results
        """
        score1 = self.calculate_score(reference, candidate1)
        score2 = self.calculate_score(reference, candidate2)
        
        comparison = {
            'reference': reference,
            'candidate1': {
                'text': candidate1,
                'scores': score1
            },
            'candidate2': {
                'text': candidate2,
                'scores': score2
            }
        }
        
        return comparison


def rouge_score(reference: str, hypothesis: str, rouge_types: List[str] = None) -> Dict[str, Any]:
    """
    Calculate ROUGE score between reference and hypothesis (standalone function)
    
    Args:
        reference (str): Reference sentence
        hypothesis (str): Hypothesis sentence
        rouge_types (List[str]): List of ROUGE types to calculate
    
    Returns:
        Dict[str, Any]: ROUGE scores dictionary
    """
    config = ROUGEScoreConfig(rouge_types=rouge_types)
    calculator = ROUGEScoreCalculator(config)
    return calculator.calculate_score(reference, hypothesis)