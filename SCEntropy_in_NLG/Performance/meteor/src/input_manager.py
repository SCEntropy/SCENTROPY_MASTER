"""
Input Manager Module

Handles loading, validating, and managing input data from various formats for METEOR evaluation.
"""

import json
import csv
from typing import List, Dict, Any, Tuple, Optional
import os


class InputManager:
    """
    Class for handling different input formats and validation for METEOR evaluation
    """
    
    @staticmethod
    def load_from_json(file_path: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from JSON file
        
        Args:
            file_path (str): Path to the JSON input file
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Reference, unconstrained, and constrained sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not all(key in data for key in ['reference', 'candidate_unconstrained', 'candidate_constrained']):
            raise ValueError("JSON file must contain 'reference', 'candidate_unconstrained', and 'candidate_constrained' keys")
        
        reference = data['reference']
        candidate_unconstrained = data['candidate_unconstrained']
        candidate_constrained = data['candidate_constrained']
        
        InputManager._validate_input_lengths(reference, candidate_unconstrained, candidate_constrained)
        
        return reference, candidate_unconstrained, candidate_constrained
    
    @staticmethod
    def load_from_csv(file_path: str, 
                      ref_col: str = 'reference', 
                      unc_col: str = 'candidate_unconstrained', 
                      con_col: str = 'candidate_constrained') -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV input file
            ref_col (str): Name of the reference column
            unc_col (str): Name of the unconstrained candidate column
            con_col (str): Name of the constrained candidate column
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Reference, unconstrained, and constrained sentences
        """
        reference = []
        candidate_unconstrained = []
        candidate_constrained = []
        
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if ref_col not in row or unc_col not in row or con_col not in row:
                    raise ValueError(f"CSV file must contain columns: {ref_col}, {unc_col}, {con_col}")
                
                reference.append(row[ref_col])
                candidate_unconstrained.append(row[unc_col])
                candidate_constrained.append(row[con_col])
        
        InputManager._validate_input_lengths(reference, candidate_unconstrained, candidate_constrained)
        
        return reference, candidate_unconstrained, candidate_constrained
    
    @staticmethod
    def load_from_text_files(ref_file: str, unc_file: str, con_file: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from separate text files
        
        Args:
            ref_file (str): Path to the reference text file
            unc_file (str): Path to the unconstrained candidate text file
            con_file (str): Path to the constrained candidate text file
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Reference, unconstrained, and constrained sentences
        """
        def read_lines(file_path: str) -> List[str]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        
        reference = read_lines(ref_file)
        candidate_unconstrained = read_lines(unc_file)
        candidate_constrained = read_lines(con_file)
        
        InputManager._validate_input_lengths(reference, candidate_unconstrained, candidate_constrained)
        
        return reference, candidate_unconstrained, candidate_constrained
    
    @staticmethod
    def _validate_input_lengths(reference: List[str], 
                               candidate_unconstrained: List[str], 
                               candidate_constrained: List[str]) -> None:
        """
        Validate that all input lists have the same length
        
        Args:
            reference (List[str]): List of reference sentences
            candidate_unconstrained (List[str]): List of unconstrained candidate sentences
            candidate_constrained (List[str]): List of constrained candidate sentences
        """
        if not (len(reference) == len(candidate_unconstrained) == len(candidate_constrained)):
            raise ValueError(
                f"All input lists must have the same length. "
                f"Got reference: {len(reference)}, "
                f"unconstrained: {len(candidate_unconstrained)}, "
                f"constrained: {len(candidate_constrained)}"
            )
    
    @staticmethod
    def validate_data(reference: List[str], 
                     candidate_unconstrained: List[str], 
                     candidate_constrained: List[str]) -> bool:
        """
        Validate input data
        
        Args:
            reference (List[str]): List of reference sentences
            candidate_unconstrained (List[str]): List of unconstrained candidate sentences
            candidate_constrained (List[str]): List of constrained candidate sentences
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if lists are empty
        if not reference or not candidate_unconstrained or not candidate_constrained:
            return False
        
        # Check if all lists have the same length
        if not (len(reference) == len(candidate_unconstrained) == len(candidate_constrained)):
            return False
        
        # Check if any sentence is not a string
        for lst in [reference, candidate_unconstrained, candidate_constrained]:
            if not all(isinstance(item, str) for item in lst):
                return False
        
        return True
    
    @staticmethod
    def load_from_single_txt(file_path: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from a single text file with the format:
        ### REFERENCE ###
        sentence1
        
        ### UNCONSTRAINED ###
        sentence1
        
        ### CONSTRAINED ###
        sentence1
        
        Args:
            file_path (str): Path to the text file
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Reference, unconstrained, and constrained sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the positions of the section headers
        reference_start = content.find('### REFERENCE ###')
        unconst_start = content.find('### UNCONSTRAINED ###')
        constr_start = content.find('### CONSTRAINED ###')
        
        if reference_start == -1 or unconst_start == -1 or constr_start == -1:
            raise ValueError(f"File {file_path} must contain REFERENCE, UNCONSTRAINED, and CONSTRAINED sections")
        
        # Extract content between sections
        reference_text = content[reference_start + len('### REFERENCE ###'):unconst_start].strip()
        unconst_text = content[unconst_start + len('### UNCONSTRAINED ###'):constr_start].strip()
        constr_text = content[constr_start + len('### CONSTRAINED ###'):].strip()
        
        # Split each section into lines and remove empty lines
        reference = [line.strip() for line in reference_text.split('\n') if line.strip()]
        candidate_unconstrained = [line.strip() for line in unconst_text.split('\n') if line.strip()]
        candidate_constrained = [line.strip() for line in constr_text.split('\n') if line.strip()]

        # Filter out any empty strings that might remain after stripping
        reference = [s for s in reference if s]
        candidate_unconstrained = [s for s in candidate_unconstrained if s]
        candidate_constrained = [s for s in candidate_constrained if s]

        InputManager._validate_input_lengths(reference, candidate_unconstrained, candidate_constrained)

        return reference, candidate_unconstrained, candidate_constrained

    @staticmethod
    def create_sample_input(file_path: str = "sample_input.json") -> None:
        """
        Create a sample input file for demonstration purposes
        
        Args:
            file_path (str): Path where the sample file will be created
        """
        sample_data = {
            "reference": [
                "The quick brown fox jumps over the lazy dog",
                "Natural language processing is a subfield of artificial intelligence"
            ],
            "candidate_unconstrained": [
                "A fast brown fox leaps over the sleepy dog",
                "NLP is a branch of AI that deals with language"
            ],
            "candidate_constrained": [
                "The fast brown fox jumps over the lazy dog",
                "Natural language processing is a field of artificial intelligence"
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample input file created at: {file_path}")


def save_results(results: Dict[str, Any], output_file: str = "meteor_results.json") -> None:
    """
    Save evaluation results to a JSON file
    
    Args:
        results (Dict[str, Any]): Results dictionary to save
        output_file (str): Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


def save_results_txt(results: Dict[str, Any], output_file: str = "meteor_results.txt") -> None:
    """
    Save evaluation results to a TXT file with grouped format.
    Each group contains: reference sentence, unconstrained sentence with score, 
    constrained sentence with score.
    
    Args:
        results (Dict[str, Any]): Results dictionary to save
        output_file (str): Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        reference = results.get('reference', [])
        candidate_unconstrained = results.get('candidate_unconstrained', [])
        candidate_constrained = results.get('candidate_constrained', [])
        unconstrained_scores = results.get('unconstrained_scores', {}).get('individual_scores', [])
        constrained_scores = results.get('constrained_scores', {}).get('individual_scores', [])
        
        for i in range(len(reference)):
            f.write(f"=== Group {i+1} ===\n")
            f.write(f"Reference: {reference[i]}\n")
            unc_score = unconstrained_scores[i] if i < len(unconstrained_scores) else 'N/A'
            con_score = constrained_scores[i] if i < len(constrained_scores) else 'N/A'
            f.write(f"Unconstrained: {candidate_unconstrained[i]}\n")
            f.write(f"Unconstrained Score: {unc_score:.4f}\n" if isinstance(unc_score, (int, float)) else f"Unconstrained Score: {unc_score}\n")
            f.write(f"Constrained: {candidate_constrained[i]}\n")
            f.write(f"Constrained Score: {con_score:.4f}\n" if isinstance(con_score, (int, float)) else f"Constrained Score: {con_score}\n")
            f.write("\n")
        
        # Write summary at the end
        f.write("=== Summary ===\n")
        f.write(f"Total sentence pairs: {len(reference)}\n")
        f.write(f"Average Unconstrained Score: {results['unconstrained_scores']['average_score']:.4f}\n")
        f.write(f"Average Constrained Score: {results['constrained_scores']['average_score']:.4f}\n")
    
    print(f"Results saved to: {output_file}")