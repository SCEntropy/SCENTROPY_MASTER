import os
from typing import Tuple, Optional


class InputManager:
    """
    Input data manager class
    Reads test data from .txt files
    """
    
    @staticmethod
    def read_input_file(file_path: str) -> Tuple[str, str, str]:
        """
        Read reference text, unconstrained generated text, and constrained generated text from .txt file
        
        Args:
            file_path: Input file path
            
        Returns:
            tuple: (reference, unconstrained_candidate, constrained_candidate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file does not exist: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse file content by format
        lines = content.split('\n')
        
        reference = ""
        unconstrained = ""
        constrained = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("### REFERENCE ###"):
                current_section = "reference"
            elif line.startswith("### UNCONSTRAINED ###"):
                current_section = "unconstrained"
            elif line.startswith("### CONSTRAINED ###"):
                current_section = "constrained"
            elif current_section == "reference" and line:
                reference = line
            elif current_section == "unconstrained" and line:
                unconstrained = line
            elif current_section == "constrained" and line:
                constrained = line
        
        # Validate if all required fields have been read
        if not reference:
            raise ValueError("Reference text not found in input file (REFERENCE)")
        if not unconstrained:
            raise ValueError("Unconstrained generated text not found in input file (UNCONSTRAINED)")
        if not constrained:
            raise ValueError("Constrained generated text not found in input file (CONSTRAINED)")
        
        return reference, unconstrained, constrained
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate if the input file path is valid
        
        Args:
            file_path: File path
            
        Returns:
            bool: Whether the file is valid
        """
        return os.path.exists(file_path) and file_path.lower().endswith('.txt')