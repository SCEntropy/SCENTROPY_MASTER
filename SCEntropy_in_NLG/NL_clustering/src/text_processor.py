import re
import sys
import time
from typing import List, Tuple


class DualOutput:
    """
    A class to handle dual output to both terminal and file
    """
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(output_filename: str = "output.txt"):
    """
    Set up dual output logging
    """
    sys.stdout = DualOutput(output_filename)
    print(f"Experiment start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {output_filename}\n{'='*50}")


def clean_text(response_text: str) -> str:
    """
    Clean response text by removing Markdown formatting
    """
    # Remove Markdown headers, list symbols, etc.
    cleaned_text = re.sub(r'###.*?\n', '', response_text)  # Remove headers
    cleaned_text = re.sub(r'-\s*', '', cleaned_text)  # Remove list symbols
    cleaned_text = re.sub(r'\*\*.*?\*\*', '', cleaned_text)  # Remove bold symbols
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Handle extra blank lines
    # Remove numbering in text (such as "5." or "4.")
    cleaned_text = re.sub(r'^\d+\.', '', cleaned_text)  # Remove starting numbers like "5."
    cleaned_text = re.sub(r'\n\d+\.', '\n', cleaned_text)  # Remove numbers after line breaks like "\n5."
    return cleaned_text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Automatically splits a piece of text into multiple sentences based on punctuation marks,
    and removes unnecessary spaces.
    Supports periods, question marks, and exclamation marks.
    """
    # Use regular expression to split the text by period, question mark, and exclamation mark
    sentences = re.split(r'(?<=[.?!])', text)  # Split by period, question mark, and exclamation mark, keeping the punctuation
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences
    return sentences