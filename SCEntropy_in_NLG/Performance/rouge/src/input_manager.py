"""
Input Manager Module

Handles loading, validating, and managing input data from various formats for ROUGE evaluation.
"""

import json
import csv
from typing import List, Dict, Any, Tuple, Optional
import os
import re


class InputManager:
    """
    Class for handling different input formats and validation for ROUGE evaluation
    """
    
    @staticmethod
    def load_from_json(file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load data from JSON file
        
        Args:
            file_path (str): Path to the JSON input file
        
        Returns:
            Tuple[List[str], List[str]]: Reference and candidate sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not all(key in data for key in ['reference', 'candidate']):
            raise ValueError("JSON file must contain 'reference' and 'candidate' keys")
        
        reference = data['reference']
        candidate = data['candidate']
        
        InputManager._validate_input_lengths(reference, candidate)
        
        return reference, candidate
    
    @staticmethod
    def load_from_csv(file_path: str, 
                      ref_col: str = 'reference', 
                      cand_col: str = 'candidate') -> Tuple[List[str], List[str]]:
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV input file
            ref_col (str): Name of the reference column
            cand_col (str): Name of the candidate column
        
        Returns:
            Tuple[List[str], List[str]]: Reference and candidate sentences
        """
        reference = []
        candidate = []
        
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if ref_col not in row or cand_col not in row:
                    raise ValueError(f"CSV file must contain columns: {ref_col}, {cand_col}")
                
                reference.append(row[ref_col])
                candidate.append(row[cand_col])
        
        InputManager._validate_input_lengths(reference, candidate)
        
        return reference, candidate
    
    @staticmethod
    def load_from_text_files(ref_file: str, cand_file: str) -> Tuple[List[str], List[str]]:
        """
        Load data from separate text files
        
        Args:
            ref_file (str): Path to the reference text file
            cand_file (str): Path to the candidate text file
        
        Returns:
            Tuple[List[str], List[str]]: Reference and candidate sentences
        """
        def read_lines(file_path: str) -> List[str]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        
        reference = read_lines(ref_file)
        candidate = read_lines(cand_file)
        
        InputManager._validate_input_lengths(reference, candidate)
        
        return reference, candidate

    @staticmethod
    def load_from_txt_file(file_path: str) -> Tuple[str, str, str]:
        """
        Load data from a .txt file with the format:
        ### REFERENCE ###
        Reference text
        ### UNCONSTRAINED ###
        Unconstrained candidate text
        ### CONSTRAINED ###
        Constrained candidate text
        
        Args:
            file_path (str): Path to the .txt input file
        
        Returns:
            Tuple[str, str, str]: Reference text, constrained candidate text, and unconstrained candidate text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by section markers
        sections = re.split(r'###\s*(\w+)\s*###', content)
        
        # Process sections to extract content
        data = {}
        for i in range(1, len(sections), 2):  # Process header-content pairs
            if i + 1 < len(sections):
                header = sections[i].strip().upper()
                content = sections[i + 1].strip()
                data[header] = content
        
        # Extract required sections
        if 'REFERENCE' not in data:
            raise ValueError("Input file must contain a 'REFERENCE' section")
        if 'UNCONSTRAINED' not in data:
            raise ValueError("Input file must contain an 'UNCONSTRAINED' section")
        if 'CONSTRAINED' not in data:
            raise ValueError("Input file must contain a 'CONSTRAINED' section")
        
        reference = data['REFERENCE']
        candidate_with_constraints = data['CONSTRAINED']
        candidate_without_constraints = data['UNCONSTRAINED']
        
        return reference, candidate_with_constraints, candidate_without_constraints

    @staticmethod
    def load_test_data() -> Tuple[str, str, str]:
        """
        Load predefined test data for demonstration purposes
        
        Returns:
            Tuple[str, str, str]: Reference text, constrained candidate text, and unconstrained candidate text
        """
        reference_text = "Implement thermal management techniques such as heat sinks, fans, or phasechange materials to maintain optimal operating temperatures, as high temperatures can increase energy consumption."
        generated_text_with_constraints = "To maintain optimal operating temperatures and prevent increased energy consumption caused by high temperatures, it is essential to implement thermal management techniques like heat sinks, fans, or phasechange materials."
        generated_text_without_constraints = "Effective thermal management does not just cut down on energy waste; it also boosts the components' longevity and performance."
        
        return reference_text, generated_text_with_constraints, generated_text_without_constraints
    
    @staticmethod
    def _validate_input_lengths(reference: List[str], 
                               candidate: List[str]) -> None:
        """
        Validate that all input lists have the same length
        
        Args:
            reference (List[str]): List of reference sentences
            candidate (List[str]): List of candidate sentences
        """
        if len(reference) != len(candidate):
            raise ValueError(
                f"Reference and candidate lists must have the same length. "
                f"Got reference: {len(reference)}, candidate: {len(candidate)}"
            )
    
    @staticmethod
    def validate_data(reference: List[str], 
                     candidate: List[str]) -> bool:
        """
        Validate input data
        
        Args:
            reference (List[str]): List of reference sentences
            candidate (List[str]): List of candidate sentences
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if lists are empty
        if not reference or not candidate:
            return False
        
        # Check if lists have the same length
        if len(reference) != len(candidate):
            return False
        
        # Check if any sentence is not a string
        for lst in [reference, candidate]:
            if not all(isinstance(item, str) for item in lst):
                return False
        
        return True
    
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
            "candidate": [
                "A fast brown fox leaps over the sleepy dog",
                "NLP is a branch of AI that deals with language"
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample input file created at: {file_path}")


def save_results(results: Dict[str, Any], output_file: str = "rouge_results.json") -> None:
    """
    Save evaluation results to a JSON file
    
    Args:
        results (Dict[str, Any]): Results dictionary to save
        output_file (str): Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")