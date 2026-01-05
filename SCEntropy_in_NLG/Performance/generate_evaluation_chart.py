"""
Evaluation Results Visualization
Generate a comprehensive PDF chart showing BERT, METEOR, and ROUGE evaluation results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import re

# Set matplotlib to use a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def parse_bert_results(file_path):
    """Parse BERTScore results from text file"""
    results = {'unconstrained': {}, 'constrained': {}}
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Extract unconstrained scores
        unc_match = re.search(r'Unconstrained generation:.*?Precision:\s*([\d.]+).*?Recall:\s*([\d.]+).*?F1:\s*([\d.]+)', 
                             content, re.DOTALL)
        if unc_match:
            results['unconstrained'] = {
                'precision': float(unc_match.group(1)),
                'recall': float(unc_match.group(2)),
                'f1': float(unc_match.group(3))
            }
        
        # Extract constrained scores
        con_match = re.search(r'Constrained generation:.*?Precision:\s*([\d.]+).*?Recall:\s*([\d.]+).*?F1:\s*([\d.]+)', 
                             content, re.DOTALL)
        if con_match:
            results['constrained'] = {
                'precision': float(con_match.group(1)),
                'recall': float(con_match.group(2)),
                'f1': float(con_match.group(3))
            }
    
    return results


def parse_meteor_results(file_path):
    """Parse METEOR results from text file"""
    results = {'unconstrained': 0, 'constrained': 0}
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Extract unconstrained score
        unc_match = re.search(r'Unconstrained METEOR:\s*([\d.]+)', content)
        if unc_match:
            results['unconstrained'] = float(unc_match.group(1))
        
        # Extract constrained score
        con_match = re.search(r'Constrained METEOR:\s*([\d.]+)', content)
        if con_match:
            results['constrained'] = float(con_match.group(1))
    
    return results


def parse_rouge_results(file_path):
    """Parse ROUGE results from text file"""
    results = {'unconstrained': {}, 'constrained': {}}
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Extract constrained scores
        con_section = re.search(r'Constrained generation:(.*?)Unconstrained generation:', content, re.DOTALL)
        if con_section:
            con_text = con_section.group(1)
            for rouge_type in ['ROUGE1', 'ROUGE2', 'ROUGEL']:
                match = re.search(rf'{rouge_type}:.*?F1=([\d.]+),.*?P=([\d.]+),.*?R=([\d.]+)', con_text)
                if match:
                    results['constrained'][rouge_type] = {
                        'f1': float(match.group(1)),
                        'precision': float(match.group(2)),
                        'recall': float(match.group(3))
                    }
        
        # Extract unconstrained scores
        unc_section = re.search(r'Unconstrained generation:(.*?)={60}', content, re.DOTALL)
        if unc_section:
            unc_text = unc_section.group(1)
            for rouge_type in ['ROUGE1', 'ROUGE2', 'ROUGEL']:
                match = re.search(rf'{rouge_type}:.*?F1=([\d.]+),.*?P=([\d.]+),.*?R=([\d.]+)', unc_text)
                if match:
                    results['unconstrained'][rouge_type] = {
                        'f1': float(match.group(1)),
                        'precision': float(match.group(2)),
                        'recall': float(match.group(3))
                    }
    
    return results


def create_evaluation_chart(bert_results, meteor_results, rouge_results, output_file='evaluation_results.pdf'):
    """Create a comprehensive evaluation chart in PDF format"""
    
    # Create PDF
    with PdfPages(output_file) as pdf:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('NLG Evaluation Results: Constrained vs Unconstrained Generation', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Color scheme
        color_unconstrained = '#E74C3C'  # Red
        color_constrained = '#3498DB'    # Blue
        
        # ==================== Subplot 1: BERTScore ====================
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Precision', 'Recall', 'F1']
        unconstrained_bert = [bert_results['unconstrained'].get(m.lower(), 0) for m in metrics]
        constrained_bert = [bert_results['constrained'].get(m.lower(), 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, unconstrained_bert, width, label='Unconstrained', 
                       color=color_unconstrained, alpha=0.8)
        bars2 = ax1.bar(x + width/2, constrained_bert, width, label='Constrained', 
                       color=color_constrained, alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Metrics', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax1.set_title('BERTScore', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ==================== Subplot 2: METEOR ====================
        ax2 = plt.subplot(2, 3, 2)
        meteor_scores = [meteor_results['unconstrained'], meteor_results['constrained']]
        labels = ['Unconstrained', 'Constrained']
        colors = [color_unconstrained, color_constrained]
        
        bars = ax2.bar(labels, meteor_scores, color=colors, alpha=0.8, width=0.6)
        
        # Add value labels
        for bar, score in zip(bars, meteor_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax2.set_title('METEOR', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, max(meteor_scores) * 1.3 if max(meteor_scores) > 0 else 0.5)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ==================== Subplot 3: ROUGE F1 Comparison ====================
        ax3 = plt.subplot(2, 3, 3)
        rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        unconstrained_f1 = [rouge_results['unconstrained'].get(rt.replace('-', ''), {}).get('f1', 0) 
                           for rt in rouge_types]
        constrained_f1 = [rouge_results['constrained'].get(rt.replace('-', ''), {}).get('f1', 0) 
                         for rt in rouge_types]
        
        x = np.arange(len(rouge_types))
        bars1 = ax3.bar(x - width/2, unconstrained_f1, width, label='Unconstrained', 
                       color=color_unconstrained, alpha=0.8)
        bars2 = ax3.bar(x + width/2, constrained_f1, width, label='Constrained', 
                       color=color_constrained, alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('ROUGE Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax3.set_title('ROUGE F1 Scores', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(rouge_types)
        ax3.legend()
        max_f1 = max(max(unconstrained_f1), max(constrained_f1))
        ax3.set_ylim(0, max_f1 * 1.2 if max_f1 > 0 else 0.5)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ==================== Subplot 4-6: ROUGE Detailed (P, R, F1) ====================
        for idx, rouge_type in enumerate(['ROUGE1', 'ROUGE2', 'ROUGEL']):
            ax = plt.subplot(2, 3, 4 + idx)
            
            metrics = ['Precision', 'Recall', 'F1']
            unc_scores = [rouge_results['unconstrained'].get(rouge_type, {}).get(m.lower(), 0) 
                         for m in metrics]
            con_scores = [rouge_results['constrained'].get(rouge_type, {}).get(m.lower(), 0) 
                         for m in metrics]
            
            x = np.arange(len(metrics))
            bars1 = ax.bar(x - width/2, unc_scores, width, label='Unconstrained', 
                          color=color_unconstrained, alpha=0.8)
            bars2 = ax.bar(x + width/2, con_scores, width, label='Constrained', 
                          color=color_constrained, alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Metrics', fontsize=10, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10, fontweight='bold')
            ax.set_title(f'{rouge_type.replace("ROUGE", "ROUGE-")} Details', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.legend(fontsize=9)
            ax.set_ylim(0, max(max(unc_scores), max(con_scores)) * 1.25 if max(max(unc_scores), max(con_scores)) > 0 else 0.5)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=300)
        plt.close()
        
        print(f"✓ Chart saved to: {os.path.abspath(output_file)}")


def main():
    """Main function to generate evaluation chart"""
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define result file paths
    bert_file = os.path.join(script_dir, 'bert', 'bertscore_results_all.txt')
    meteor_file = os.path.join(script_dir, 'meteor', 'meteor_results.txt')
    rouge_file = os.path.join(script_dir, 'rogue', 'rouge_results.txt')
    output_file = os.path.join(script_dir, 'evaluation_results.pdf')
    
    print("Reading evaluation results...")
    print(f"  - BERTScore: {bert_file}")
    print(f"  - METEOR: {meteor_file}")
    print(f"  - ROUGE: {rouge_file}")
    
    # Parse results
    bert_results = parse_bert_results(bert_file)
    meteor_results = parse_meteor_results(meteor_file)
    rouge_results = parse_rouge_results(rouge_file)
    
    print("\nGenerating PDF chart...")
    
    # Create chart
    create_evaluation_chart(bert_results, meteor_results, rouge_results, output_file)
    
    print("\nEvaluation chart generation completed!")


if __name__ == "__main__":
    main()
