import argparse
import os
import sys
from src.input_manager import InputManager
from src.bert_score_calculator import BERTScoreCalculator


def run_single_file(args, input_file):
    """
    Run BERTScore evaluation on a single file
    
    Args:
        args: Parsed command-line arguments
        input_file: Path to the input file
        
    Returns:
        dict: Evaluation results
    """
    # Read input file
    reference, unconstrained_candidate, constrained_candidate = InputManager.read_input_file(input_file)
    
    # Initialize BERTScore calculator
    calculator = BERTScoreCalculator(model_type=args.model_type, local_model_path=args.local_model_path)
    
    # Download model automatically
    calculator.download_model_if_needed()
    
    # Calculate BERTScore
    results = calculator.calculate_scores(reference, unconstrained_candidate, constrained_candidate)
    
    return results


def run_batch_evaluation(args):
    """
    Run BERTScore evaluation on all data files (data1.txt to data100.txt)
    
    Args:
        args: Parsed command-line arguments
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Input folder is at ../input relative to this script
    input_dir = os.path.join(script_dir, "..", "input")
    
    all_results = []
    total_unconstrained = {'precision': [], 'recall': [], 'f1': []}
    total_constrained = {'precision': [], 'recall': [], 'f1': []}
    
    # Initialize calculator once for all files
    calculator = BERTScoreCalculator(model_type=args.model_type, local_model_path=args.local_model_path)
    calculator.download_model_if_needed()
    
    for i in range(1, 101):
        input_file = os.path.join(input_dir, f"data{i}.txt")
        
        if not os.path.exists(input_file):
            continue
        
        try:
            # Read input file
            reference, unconstrained_candidate, constrained_candidate = InputManager.read_input_file(input_file)
            
            # Calculate BERTScore
            results = calculator.calculate_scores(reference, unconstrained_candidate, constrained_candidate)
            
            # Collect scores
            total_unconstrained['precision'].append(results['unconstrained']['precision'])
            total_unconstrained['recall'].append(results['unconstrained']['recall'])
            total_unconstrained['f1'].append(results['unconstrained']['f1'])
            total_constrained['precision'].append(results['constrained']['precision'])
            total_constrained['recall'].append(results['constrained']['recall'])
            total_constrained['f1'].append(results['constrained']['f1'])
            
            file_result = {
                "file": f"data{i}.txt",
                "unconstrained": results['unconstrained'],
                "constrained": results['constrained']
            }
            all_results.append(file_result)
            
        except Exception as e:
            continue
    
    # Calculate overall averages
    if all_results:
        avg_unconstrained = {
            'precision': sum(total_unconstrained['precision']) / len(total_unconstrained['precision']),
            'recall': sum(total_unconstrained['recall']) / len(total_unconstrained['recall']),
            'f1': sum(total_unconstrained['f1']) / len(total_unconstrained['f1'])
        }
        avg_constrained = {
            'precision': sum(total_constrained['precision']) / len(total_constrained['precision']),
            'recall': sum(total_constrained['recall']) / len(total_constrained['recall']),
            'f1': sum(total_constrained['f1']) / len(total_constrained['f1'])
        }
        
        # Determine output filename
        output_file = args.output_file if args.output_file else "bertscore_results_all.txt"
        
        # Save results to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== BERTScore Batch Evaluation Results ===\n")
            f.write(f"Model type: {args.model_type}\n")
            f.write(f"Total files processed: {len(all_results)}\n\n")
            
            f.write("=== Overall Average Scores ===\n")
            f.write("Unconstrained generation:\n")
            f.write(f"  Precision: {avg_unconstrained['precision']:.4f}\n")
            f.write(f"  Recall:    {avg_unconstrained['recall']:.4f}\n")
            f.write(f"  F1:        {avg_unconstrained['f1']:.4f}\n\n")
            f.write("Constrained generation:\n")
            f.write(f"  Precision: {avg_constrained['precision']:.4f}\n")
            f.write(f"  Recall:    {avg_constrained['recall']:.4f}\n")
            f.write(f"  F1:        {avg_constrained['f1']:.4f}\n\n")
            
            f.write("=== Per-File Results ===\n")
            for result in all_results:
                f.write(f"\n{result['file']}:\n")
                f.write(f"  Unconstrained - P: {result['unconstrained']['precision']:.4f}, R: {result['unconstrained']['recall']:.4f}, F1: {result['unconstrained']['f1']:.4f}\n")
                f.write(f"  Constrained   - P: {result['constrained']['precision']:.4f}, R: {result['constrained']['recall']:.4f}, F1: {result['constrained']['f1']:.4f}\n")
        
        print("Evaluation completed.")


def main():
    parser = argparse.ArgumentParser(
        description="BERTScore Evaluator",
        epilog="""
Examples:
  %(prog)s                           # Evaluate all data1.txt to data100.txt
  %(prog)s --input-file data1.txt    # Evaluate a specific file
  %(prog)s --input-file data2.txt --model-type bert-large-uncased
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-file", 
        type=str, 
        required=False,
        default=None,
        help="Path to a specific input .txt file (if not provided, will process data1.txt to data100.txt)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bert-base-uncased",
        help="BERT model type to use (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Local model path, if provided use local model"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output result file path (default: generated based on input filename)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.input_file:
            # Single file mode
            if not InputManager.validate_file_path(args.input_file):
                sys.exit(1)
            
            results = run_single_file(args, args.input_file)
            
            # Determine output filename
            if args.output_file is None:
                input_name = os.path.splitext(os.path.basename(args.input_file))[0]
                output_file = f"bertscore_results_{input_name}.txt"
            else:
                output_file = args.output_file
            
            # Save results to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== BERTScore Evaluation Results ===\n")
                f.write(f"Input file: {args.input_file}\n")
                f.write(f"Model type: {args.model_type}\n\n")
                
                f.write("Reference text:\n")
                f.write(results['reference'] + "\n\n")
                
                f.write("Unconstrained generation:\n")
                f.write(results['unconstrained']['text'] + "\n")
                f.write(f"Precision: {results['unconstrained']['precision']:.4f}\n")
                f.write(f"Recall:    {results['unconstrained']['recall']:.4f}\n")
                f.write(f"F1:        {results['unconstrained']['f1']:.4f}\n\n")
                
                f.write("Constrained generation:\n")
                f.write(results['constrained']['text'] + "\n")
                f.write(f"Precision: {results['constrained']['precision']:.4f}\n")
                f.write(f"Recall:    {results['constrained']['recall']:.4f}\n")
                f.write(f"F1:        {results['constrained']['f1']:.4f}\n")
            
            print("Evaluation completed.")
        else:
            # Batch mode: process data1.txt to data100.txt
            run_batch_evaluation(args)
        
    except FileNotFoundError:
        sys.exit(1)
    except ValueError:
        sys.exit(1)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()