#!/usr/bin/env python3
"""
SCEntropy - Sentence Clustering Tool

This tool performs sentence clustering based on semantic similarity using pre-generated sentences.
Supports reading QUESTION format files with multiple sections.
"""

import os
import argparse
from src.main_processor import MainProcessor
from src.text_processor import setup_logging
import re


def read_sentences_from_file(file_path):
    """Read pre-generated sentences from a text file
    
    File format: Each sentence starts with 'Label N: sentence content'
    Multi-line sentences are supported - lines without 'Label N:' prefix
    are appended to the previous sentence.
    """
    sentences = []
    current_sentence = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n\r')
            
            # Check if line starts with Label pattern (supports both English and Chinese colons)
            match = re.match(r'(?:Label)\s*\d+[:：]\s*(.+)', line, re.IGNORECASE)
            
            if match:
                # Save previous sentence if exists
                if current_sentence is not None:
                    sentences.append(current_sentence.strip())
                # Start new sentence
                current_sentence = match.group(1).strip()
            elif line.strip():
                # Append to current sentence if line is not empty
                if current_sentence is not None:
                    current_sentence += '\n' + line
                # If no current sentence and line doesn't match pattern, treat as standalone
                else:
                    current_sentence = line.strip()
        
        # Don't forget to add the last sentence
        if current_sentence is not None and current_sentence.strip():
            sentences.append(current_sentence.strip())
    
    return sentences


def parse_question_file(file_path):
    """Parse QUESTION format file with multiple sections
    
    File format:
    ###Question###
    question content
    
    ###Raw data###
    Label 0: ...
    Label 1: ...
    
    ###Constrained generation results###
    Label 0: ...
    
    ###Unconstrained generation result###
    Label 0: ...
    
    Returns:
        dict with keys: 'question', 'raw_data', 'constrained', 'unconstrained'
        Each value (except question) is a list of sentences
    """
    sections = {
        'question': '',
        'raw_data': [],
        'constrained': [],
        'unconstrained': []
    }
    
    # Section markers mapping
    section_markers = {
        '###Question###': 'question',
        '###Raw data###': 'raw_data',
        '###Constrained generation results###': 'constrained',
        '###Unconstrained generation result###': 'unconstrained'
    }
    
    current_section = None
    current_sentence = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n\r')
            
            # Check for section markers
            if line.strip() in section_markers:
                # Save previous sentence if exists
                if current_sentence is not None and current_section and current_section != 'question':
                    sections[current_section].append(current_sentence.strip())
                current_section = section_markers[line.strip()]
                current_sentence = None
                continue
            
            if current_section is None:
                continue
            
            # Handle question section (just store the text)
            if current_section == 'question':
                if line.strip():
                    sections['question'] = line.strip()
                continue
            
            # Handle data sections with Label pattern
            match = re.match(r'(?:Label)\s*\d+[:：]\s*(.+)', line, re.IGNORECASE)
            
            if match:
                # Save previous sentence if exists
                if current_sentence is not None:
                    sections[current_section].append(current_sentence.strip())
                current_sentence = match.group(1).strip()
            elif line.strip() and current_sentence is not None:
                # Append to current sentence
                current_sentence += '\n' + line
        
        # Save the last sentence
        if current_sentence is not None and current_section and current_section != 'question':
            sections[current_section].append(current_sentence.strip())
    
    return sections


def generate_text_dendrogram(sentences, tracked_clusters, title="Clustering Dendrogram"):
    """Generate a text-based dendrogram from clustering results
    
    Format matches the 'Final Safe Result' section in constraint result files.
    
    Args:
        sentences: List of original sentences
        tracked_clusters: List of clustering rounds, each round contains clusters of original label indices
        title: Title for the dendrogram
    
    Returns:
        String representation of the text dendrogram
    """
    output = []
    output.append("=" * 60)
    output.append(f"=== {title} ===")
    output.append("=" * 60)
    output.append("")
    
    # Display all sentences with labels
    output.append("All sentences and their label information:")
    for i, sent in enumerate(sentences):
        sent_display = sent.replace('\n', ' ')
        output.append(f"Label {i}: {sent_display}")
    output.append("")
    
    # Tracked Cluster Information (matching the format in constraint result files)
    output.append("Tracked Cluster Information:")
    
    if not tracked_clusters:
        # No clustering happened, show initial state
        initial = [[i] for i in range(len(sentences))]
        output.append(f"Round 0: {initial}")
    else:
        for round_idx, round_info in enumerate(tracked_clusters):
            output.append(f"Round {round_idx}: {round_info}")
    
    output.append("")
    output.append("=" * 60)
    
    return "\n".join(output)


def process_question_file(file_path, entropy_threshold, embedding_model, output_file):
    """Process a QUESTION format file with three sections
    
    Args:
        file_path: Path to the QUESTION file
        entropy_threshold: Entropy threshold for clustering
        embedding_model: Model for sentence embeddings
        output_file: Output file for logging
    """
    print(f"\nParsing QUESTION file: {file_path}")
    sections = parse_question_file(file_path)
    
    print(f"\nQuestion: {sections['question']}")
    print(f"\nFound sections:")
    print(f"  - Raw data: {len(sections['raw_data'])} sentences")
    print(f"  - Constrained generation: {len(sections['constrained'])} sentences")
    print(f"  - Unconstrained generation: {len(sections['unconstrained'])} sentences")
    
    # Create processor
    processor = MainProcessor(
        entropy_threshold=entropy_threshold,
        embedding_model=embedding_model
    )
    
    output_dir = os.path.dirname(os.path.abspath(file_path))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    all_dendrograms = []
    
    # Process each section
    section_names = [
        ('raw_data', 'Raw Data'),
        ('constrained', 'Constrained Generation Results'),
        ('unconstrained', 'Unconstrained Generation Results')
    ]
    
    for section_key, section_title in section_names:
        sentences = sections[section_key]
        
        if not sentences:
            print(f"\nWarning: No sentences found in {section_title} section, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {section_title}...")
        print(f"{'='*60}")
        
        # Display sentences
        print(f"\nSentences in this section:")
        for i, sent in enumerate(sentences):
            display_sent = sent.replace('\n', ' ')[:80]
            print(f"  Label {i}: {display_sent}..." if len(sent) > 80 else f"  Label {i}: {sent.replace(chr(10), ' ')}")
        
        # Perform clustering
        print(f"\nPerforming clustering for {section_title}...")
        tracked_clusters = processor.process_sentences_from_file(sentences, output_dir)
        
        # Generate text dendrogram
        dendrogram_title = section_title
        dendrogram = generate_text_dendrogram(sentences, tracked_clusters, dendrogram_title)
        
        all_dendrograms.append(dendrogram)
    
    # Save all dendrograms to file and print final result
    output_path = os.path.join(output_dir, f"{base_name}_dendrograms.txt")
    
    # Build combined final safe result
    final_result = []
    final_result.append("\n" + "=" * 60)
    final_result.append("=== Final Safe Result ===")
    final_result.append("=" * 60)
    final_result.append("")
    
    for dendro in all_dendrograms:
        final_result.append(dendro)
        final_result.append("")
    
    final_result_str = "\n".join(final_result)
    
    # Print to console
    print(final_result_str)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Question: {sections['question']}\n")
        f.write(final_result_str)
    
    print(f"\n{'='*60}")
    print(f"All dendrograms saved to: {output_path}")
    print(f"{'='*60}")
    
    return all_dendrograms


def main():
    parser = argparse.ArgumentParser(description="SCEntropy - Sentence Clustering Tool")
    parser.add_argument("--sentences-file", type=str, required=True,
                        help="Path to the .txt file containing pre-generated sentences or QUESTION format file")
    parser.add_argument("--entropy-threshold", type=float, default=1.0, 
                        help="Entropy threshold for clustering")
    parser.add_argument("--output-file", type=str, default="output.txt", 
                        help="Output file for logging")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Path to sentence transformer model or model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--question-mode", action="store_true",
                        help="Enable QUESTION file parsing mode (file contains ###Section### markers)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.output_file)
    
    # Check if sentences file exists
    if not os.path.exists(args.sentences_file):
        print(f"Error: Sentences file '{args.sentences_file}' does not exist.")
        return
    
    # Detect QUESTION format or use explicit flag
    is_question_format = args.question_mode
    if not is_question_format:
        # Auto-detect by checking file content
        with open(args.sentences_file, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Read first 500 chars
            if '###Question###' in content or '###Raw data###' in content:
                is_question_format = True
    
    if is_question_format:
        # Process QUESTION format file
        process_question_file(
            args.sentences_file,
            args.entropy_threshold,
            args.embedding_model,
            args.output_file
        )
    else:
        # Original processing mode
        print(f"Reading pre-generated sentences from file: {args.sentences_file}")
        sentences = read_sentences_from_file(args.sentences_file)
        
        if not sentences:
            print("Error: No sentences found in the file.")
            return
        
        print(f"Loaded {len(sentences)} sentences from file.")
        
        # Create processor for clustering
        processor = MainProcessor(
            entropy_threshold=args.entropy_threshold,
            embedding_model=args.embedding_model
        )
        
        print("Starting sentence clustering...")
        
        # Get output directory from sentences file path
        output_dir = os.path.dirname(os.path.abspath(args.sentences_file))
        tracking_results = processor.process_sentences_from_file(sentences, output_dir)
        
        # Generate text dendrogram for original mode too
        dendrogram = generate_text_dendrogram(sentences, tracking_results, "Clustering Dendrogram")
        print(f"\n{dendrogram}")
        
        print("\nClustering completed successfully!")
        print(f"Total sentences processed: {len(sentences)}")


if __name__ == "__main__":
    main()
