"""
ROUGE Evaluator - Command Line Interface

Main entry point for the ROUGE evaluation tool with command-line interface.
"""

import argparse
import sys
from typing import List, Dict, Any
import json

from src.rouge_score_calculator import ROUGEScoreCalculator, ROUGEScoreConfig
from src.input_manager import InputManager, save_results


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="ROUGE Score Evaluator - A tool for evaluating text generation models using ROUGE metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              (process all data1.txt to data100.txt)
  %(prog)s --input-file data1.txt       (process only data1.txt)
  %(prog)s --output-file results.json
  %(prog)s --create-sample
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('input options')
    input_group.add_argument(
        '--input-file',
        type=str,
        help='Path to specific .txt input file (if not specified, reads all data1.txt to data100.txt)'
    )
    
    # Configuration options
    config_group = parser.add_argument_group('configuration options')
    config_group.add_argument(
        '--rouge-types',
        type=str,
        nargs='+',
        default=['rouge1', 'rouge2', 'rougeL'],
        help='ROUGE types to calculate (default: rouge1 rouge2 rougeL)'
    )
    config_group.add_argument(
        '--no-stemmer',
        action='store_true',
        help='Disable stemming in ROUGE calculations'
    )
    config_group.add_argument(
        '--output-file',
        type=str,
        default='rouge_results.txt',
        help='Output file path for results (default: rouge_results.txt)'
    )
    
    # Utility options
    utility_group = parser.add_argument_group('utility options')
    utility_group.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample input file for demonstration'
    )
    utility_group.add_argument(
        '--run-demo',
        action='store_true',
        help='Run a demonstration with predefined test data'
    )
    
    return parser


def load_input_data(args: argparse.Namespace) -> tuple:
    """
    Load input data based on the provided arguments
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        tuple: (reference, candidate, optional_second_candidate) lists
    """
    # Handle sample creation first
    if args.create_sample:
        InputManager.create_sample_input()
        sys.exit(0)
    
    # Handle demo run separately - it doesn't require input files
    if args.run_demo:
        ref_text, cand_with_constraints, cand_without_constraints = InputManager.load_test_data()
        return [ref_text], [cand_with_constraints], [cand_without_constraints]
    
    # Load data from specific .txt file if specified
    if args.input_file:
        reference, candidate_with_constraints, candidate_without_constraints = InputManager.load_from_txt_file(args.input_file)
        return [reference], [candidate_with_constraints], [candidate_without_constraints]
    
    # Default: Load all data files from data1.txt to data100.txt
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Input folder is at ../input relative to this script
    input_dir = os.path.join(script_dir, "..", "input")
    
    all_references = []
    all_candidates_with_constraints = []
    all_candidates_without_constraints = []
    
    for i in range(1, 101):
        file_path = os.path.join(input_dir, f"data{i}.txt")
        if os.path.exists(file_path):
            try:
                reference, candidate_with_constraints, candidate_without_constraints = InputManager.load_from_txt_file(file_path)
                all_references.append(reference)
                all_candidates_with_constraints.append(candidate_with_constraints)
                all_candidates_without_constraints.append(candidate_without_constraints)
                print(f"Loaded: data{i}.txt")
            except Exception as e:
                print(f"Warning: Failed to load data{i}.txt - {str(e)}")
        else:
            print(f"Warning: data{i}.txt not found, skipping...")
    
    if not all_references:
        raise ValueError("No valid data files found. Please ensure data1.txt to data100.txt exist in the script directory.")
    
    print(f"\nTotal files loaded: {len(all_references)}")
    return all_references, all_candidates_with_constraints, all_candidates_without_constraints


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run ROUGE evaluation with the provided arguments
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # For demo mode, we'll run two evaluations: constrained and unconstrained
    if args.run_demo:
        # Get test data
        ref_text, candidate_with_constraints, candidate_without_constraints = InputManager.load_test_data()
        reference = [ref_text]
        
        # Create configuration
        config = ROUGEScoreConfig(
            rouge_types=args.rouge_types,
            use_stemmer=not args.no_stemmer,
            output_file=args.output_file
        )
        
        # Initialize calculator
        calculator = ROUGEScoreCalculator(config)
        
        # Run evaluation for constrained text
        constrained_result = calculator.run_evaluation(
            reference, 
            [candidate_with_constraints]
        )
        
        # Run evaluation for unconstrained text
        unconstrained_result = calculator.run_evaluation(
            reference, 
            [candidate_without_constraints]
        )
        
        # Prepare results
        results = {
            "reference": reference,
            "candidate_with_constraints": [candidate_with_constraints],
            "candidate_without_constraints": [candidate_without_constraints],
            "constrained_scores": {
                "average_scores": constrained_result["average_scores"],
                "individual_scores": constrained_result["individual_scores"]
            },
            "unconstrained_scores": {
                "average_scores": unconstrained_result["average_scores"],
                "individual_scores": unconstrained_result["individual_scores"]
            }
        }
        
        return results
    else:
        # Load input data for non-demo mode
        reference, candidate_with_constraints, candidate_without_constraints = load_input_data(args)
        
        # Validate data
        if not InputManager.validate_data(reference, candidate_with_constraints):
            raise ValueError("Invalid input data: Lists must have the same length and contain only strings")
        
        # Create configuration
        config = ROUGEScoreConfig(
            rouge_types=args.rouge_types,
            use_stemmer=not args.no_stemmer,
            output_file=args.output_file
        )
        
        # Initialize calculator
        calculator = ROUGEScoreCalculator(config)
        
        # Run evaluation for constrained text
        constrained_result = calculator.run_evaluation(
            reference, 
            candidate_with_constraints
        )
        
        # Run evaluation for unconstrained text
        unconstrained_result = calculator.run_evaluation(
            reference, 
            candidate_without_constraints
        )
        
        # Prepare results
        results = {
            "reference": reference,
            "candidate_with_constraints": candidate_with_constraints,
            "candidate_without_constraints": candidate_without_constraints,
            "constrained_scores": {
                "average_scores": constrained_result["average_scores"],
                "individual_scores": constrained_result["individual_scores"]
            },
            "unconstrained_scores": {
                "average_scores": unconstrained_result["average_scores"],
                "individual_scores": unconstrained_result["individual_scores"]
            }
        }
        
        return results


def print_demo_results(results: Dict[str, Any]) -> None:
    """
    Print formatted results for demo mode
    
    Args:
        results: Evaluation results
    """
    print("Comparison of ROUGE scores between constrained and unconstrained generation:\n")
    
    # Extract constrained scores
    constrained_scores = results['constrained_scores']['average_scores']
    print("For Generated Text with Constraints:")
    for rouge_type, scores in constrained_scores.items():
        print(f"ROUGE-{rouge_type.upper()}: F1: {scores['fmeasure']}, Precision: {scores['precision']}, Recall: {scores['recall']}")
    print()
    
    # Extract unconstrained scores
    unconstrained_scores = results['unconstrained_scores']['average_scores']
    print("For Generated Text without Constraints:")
    for rouge_type, scores in unconstrained_scores.items():
        print(f"ROUGE-{rouge_type.upper()}: F1: {scores['fmeasure']}, Precision: {scores['precision']}, Recall: {scores['recall']}")
    
    # Save detailed results to file
    with open('rouge_scores_comparison.txt', 'w', encoding='utf-8') as file:
        file.write("Comparison of ROUGE scores between constrained and unconstrained generation:\n\n")
        
        file.write("For Generated Text with Constraints:\n")
        for rouge_type, scores in constrained_scores.items():
            file.write(f"ROUGE-{rouge_type.upper()}: F1: {scores['fmeasure']}, Precision: {scores['precision']}, Recall: {scores['recall']}\n")
        file.write("\n")
        
        file.write("For Generated Text without Constraints:\n")
        for rouge_type, scores in unconstrained_scores.items():
            file.write(f"ROUGE-{rouge_type.upper()}: F1: {scores['fmeasure']}, Precision: {scores['precision']}, Recall: {scores['recall']}\n")
        
        # Also save raw scores (with more decimal places for detailed analysis)
        file.write("\nDetailed scores:\n")
        file.write("With Constraints - ROUGE-1: F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
            results['constrained_scores']['individual_scores'][0]['scores']['rouge1']['fmeasure']/100,
            results['constrained_scores']['individual_scores'][0]['scores']['rouge1']['precision']/100,
            results['constrained_scores']['individual_scores'][0]['scores']['rouge1']['recall']/100
        ))
        file.write("Without Constraints - ROUGE-1: F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
            results['unconstrained_scores']['individual_scores'][0]['scores']['rouge1']['fmeasure']/100,
            results['unconstrained_scores']['individual_scores'][0]['scores']['rouge1']['precision']/100,
            results['unconstrained_scores']['individual_scores'][0]['scores']['rouge1']['recall']/100
        ))


def save_grouped_results_txt(results: Dict[str, Any], output_file: str = "rouge_results.txt") -> None:
    """
    Save evaluation results grouped by each sentence pair to a text file
    
    Args:
        results (Dict[str, Any]): Results dictionary to save
        output_file (str): Output text file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ROUGE Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        references = results.get('reference', [])
        candidates_constrained = results.get('candidate_with_constraints', [])
        candidates_unconstrained = results.get('candidate_without_constraints', [])
        constrained_individual = results.get('constrained_scores', {}).get('individual_scores', [])
        unconstrained_individual = results.get('unconstrained_scores', {}).get('individual_scores', [])
        
        num_pairs = len(references)
        
        for i in range(num_pairs):
            f.write(f"[Sample {i + 1}]\n")
            f.write("-" * 40 + "\n")
            
            # Reference
            f.write(f"Reference:\n{references[i]}\n\n")
            
            # Unconstrained generation and scores
            f.write(f"Unconstrained Generation:\n{candidates_unconstrained[i]}\n")
            if i < len(unconstrained_individual):
                unconstrained_scores = unconstrained_individual[i].get('scores', {})
                f.write("Unconstrained Scores: ")
                score_parts = []
                for rouge_type, scores in unconstrained_scores.items():
                    score_parts.append(f"{rouge_type.upper()}(F1={scores['fmeasure']:.2f}, P={scores['precision']:.2f}, R={scores['recall']:.2f})")
                f.write(", ".join(score_parts) + "\n\n")
            
            # Constrained generation and scores
            f.write(f"Constrained Generation:\n{candidates_constrained[i]}\n")
            if i < len(constrained_individual):
                constrained_scores = constrained_individual[i].get('scores', {})
                f.write("Constrained Scores: ")
                score_parts = []
                for rouge_type, scores in constrained_scores.items():
                    score_parts.append(f"{rouge_type.upper()}(F1={scores['fmeasure']:.2f}, P={scores['precision']:.2f}, R={scores['recall']:.2f})")
                f.write(", ".join(score_parts) + "\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Average scores summary
        f.write("AVERAGE SCORES SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        constrained_avg = results.get('constrained_scores', {}).get('average_scores', {})
        unconstrained_avg = results.get('unconstrained_scores', {}).get('average_scores', {})
        
        f.write("Constrained Average:\n")
        for rouge_type, scores in constrained_avg.items():
            f.write(f"  {rouge_type.upper()}: F1={scores['fmeasure']:.2f}, Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}\n")
        
        f.write("\nUnconstrained Average:\n")
        for rouge_type, scores in unconstrained_avg.items():
            f.write(f"  {rouge_type.upper()}: F1={scores['fmeasure']:.2f}, Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}\n")
    
    print(f"Results saved to: {output_file}")


def main():
    """
    Main entry point for the ROUGE evaluator
    """
    parser = create_argument_parser()
    
    try:
        args = parser.parse_args()
        
        # Run evaluation
        results = run_evaluation(args)
        
        # Save results in TXT format only
        txt_output_file = args.output_file.rsplit('.', 1)[0] + '.txt'
        save_grouped_results_txt(results, txt_output_file)
        
        # Print summary
        if args.run_demo:
            print_demo_results(results)
        
        print("\nEvaluation Summary:")
        print(f"Total sentence pairs: {len(results['reference'])}")
        print(f"Results saved to: {txt_output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()