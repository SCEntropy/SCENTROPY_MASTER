"""METEOR Evaluator - Command Line Interface

Main entry point for the METEOR evaluation tool with command-line interface.
"""

import argparse
import sys
import os
from typing import List, Dict, Any
import json

from src.meteor_score_calculator import METEORScoreCalculator, METEORScoreConfig
from src.input_manager import InputManager, save_results, save_results_txt


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="METEOR Score Evaluator - A tool for evaluating text generation models using METEOR metric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Evaluate all data1.txt to data100.txt
  %(prog)s --input-file data1.txt    # Evaluate a specific file
  %(prog)s --input-file data2.txt --alpha 0.8 --beta 2.0 --gamma 0.6
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('input options')
    input_group.add_argument(
        '--input-file',
        type=str,
        required=False,
        default=None,
        help='Path to a specific input text file (if not provided, will process data1.txt to data100.txt)'
    )
    
    # Configuration options
    config_group = parser.add_argument_group('configuration options')
    config_group.add_argument(
        '--alpha',
        type=float,
        default=0.9,
        help='Parameter for precision/recall balance (default: 0.9)'
    )
    config_group.add_argument(
        '--beta',
        type=float,
        default=3.0,
        help='Parameter for F-measure (default: 3.0)'
    )
    config_group.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Parameter for penalty (default: 0.5)'
    )
    config_group.add_argument(
        '--output-file',
        type=str,
        default='meteor_results.txt',
        help='Output file path for results (default: meteor_results.txt)'
    )
    
    return parser


def load_input_data(args: argparse.Namespace) -> tuple:
    """
    Load input data based on the provided arguments
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        tuple: (reference, candidate_unconstrained, candidate_constrained) lists
    """
    # Load data from the single text file
    return InputManager.load_from_single_txt(args.input_file)


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run METEOR evaluation with the provided arguments
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Load input data
    reference, candidate_unconstrained, candidate_constrained = load_input_data(args)
    
    # Validate data
    if not InputManager.validate_data(reference, candidate_unconstrained, candidate_constrained):
        raise ValueError("Invalid input data: All lists must have the same length and contain only strings")
    
    # Create configuration
    config = METEORScoreConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        output_file=args.output_file
    )
    
    # Initialize calculator
    calculator = METEORScoreCalculator(config)
    
    # Run evaluation
    unconstrained_result, constrained_result = calculator.run_evaluation(
        reference, 
        candidate_unconstrained, 
        candidate_constrained
    )
    
    # Prepare results
    results = {
        "reference": reference,
        "candidate_unconstrained": candidate_unconstrained,
        "candidate_constrained": candidate_constrained,
        "unconstrained_scores": {
            "average_score": unconstrained_result["average_score"],
            "individual_scores": unconstrained_result["scores"]
        },
        "constrained_scores": {
            "average_score": constrained_result["average_score"],
            "individual_scores": constrained_result["scores"]
        }
    }
    
    return results


def run_batch_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run METEOR evaluation on all data files (data1.txt to data100.txt)
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dict[str, Any]: Aggregated evaluation results
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Input folder is at ../input relative to this script
    input_dir = os.path.join(script_dir, "..", "input")
    
    all_results = []
    total_unconstrained_scores = []
    total_constrained_scores = []
    all_references = []
    all_unconstrained = []
    all_constrained = []
    
    print("Processing data1.txt to data100.txt...")
    
    for i in range(1, 101):
        input_file = os.path.join(input_dir, f"data{i}.txt")
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        try:
            # Load data from file
            reference, candidate_unconstrained, candidate_constrained = InputManager.load_from_single_txt(input_file)
            
            # Validate data
            if not InputManager.validate_data(reference, candidate_unconstrained, candidate_constrained):
                print(f"Warning: Invalid data in data{i}.txt, skipping...")
                continue
            
            # Create configuration
            config = METEORScoreConfig(
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                output_file=args.output_file
            )
            
            # Initialize calculator
            calculator = METEORScoreCalculator(config)
            
            # Run evaluation
            unconstrained_result, constrained_result = calculator.run_evaluation(
                reference, 
                candidate_unconstrained, 
                candidate_constrained
            )
            
            # Collect all data for txt output
            all_references.extend(reference)
            all_unconstrained.extend(candidate_unconstrained)
            all_constrained.extend(candidate_constrained)
            total_unconstrained_scores.extend(unconstrained_result["scores"])
            total_constrained_scores.extend(constrained_result["scores"])
            
            file_result = {
                "file": f"data{i}.txt",
                "unconstrained_avg": unconstrained_result["average_score"],
                "constrained_avg": constrained_result["average_score"]
            }
            all_results.append(file_result)
            
            print(f"  data{i}.txt - Unconstrained: {unconstrained_result['average_score']:.4f}, Constrained: {constrained_result['average_score']:.4f}")
            
        except Exception as e:
            print(f"Warning: Error processing data{i}.txt: {str(e)}, skipping...")
            continue
    
    # Calculate overall averages
    overall_unconstrained_avg = sum(total_unconstrained_scores) / len(total_unconstrained_scores) if total_unconstrained_scores else 0
    overall_constrained_avg = sum(total_constrained_scores) / len(total_constrained_scores) if total_constrained_scores else 0
    
    # Prepare aggregated results with full data for txt output
    results = {
        "total_files_processed": len(all_results),
        "total_sentence_pairs": len(total_unconstrained_scores),
        "overall_unconstrained_average": overall_unconstrained_avg,
        "overall_constrained_average": overall_constrained_avg,
        "per_file_results": all_results,
        # Include full data for txt output
        "reference": all_references,
        "candidate_unconstrained": all_unconstrained,
        "candidate_constrained": all_constrained,
        "unconstrained_scores": {
            "average_score": overall_unconstrained_avg,
            "individual_scores": total_unconstrained_scores
        },
        "constrained_scores": {
            "average_score": overall_constrained_avg,
            "individual_scores": total_constrained_scores
        }
    }
    
    return results


def main():
    """
    Main entry point for the METEOR evaluator
    """
    parser = create_argument_parser()
    
    try:
        args = parser.parse_args()
        
        if args.input_file:
            # Single file mode
            results = run_evaluation(args)
            
            # Save results as txt format
            save_results_txt(results, args.output_file)
            
            # Print summary
            print("\nEvaluation Summary:")
            print(f"Total sentence pairs: {len(results['reference'])}")
            print(f"Average unconstrained METEOR score: {results['unconstrained_scores']['average_score']:.4f}")
            print(f"Average constrained METEOR score: {results['constrained_scores']['average_score']:.4f}")
        else:
            # Batch mode: process data1.txt to data100.txt
            results = run_batch_evaluation(args)
            
            # Save results as txt format
            save_results_txt(results, args.output_file)
            
            # Print summary
            print("\n" + "="*50)
            print("Overall Evaluation Summary:")
            print(f"Total files processed: {results['total_files_processed']}")
            print(f"Total sentence pairs: {results['total_sentence_pairs']}")
            print(f"Overall unconstrained METEOR score: {results['overall_unconstrained_average']:.4f}")
            print(f"Overall constrained METEOR score: {results['overall_constrained_average']:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    