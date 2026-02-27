# =========================
# Compare Baseline vs Distillation Results
# =========================
"""
This script compares training results between:
1. Baseline student training (no distillation)
2. Knowledge distillation training

Use this to quantify the benefit of knowledge distillation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# =========================
# SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Paths to results
BASELINE_RESULTS = './res/baseline_student/baseline_training_results.csv'
DISTILL_RESULTS = './res/distill/distillation_results.csv'
OUTPUT_DIR = './res/comparison/'

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# =========================
# LOAD DATA
# =========================
logger.info("Loading results...")

try:
    baseline_df = pd.read_csv(BASELINE_RESULTS)
    logger.info(f"✓ Loaded baseline results from {BASELINE_RESULTS}")
except FileNotFoundError:
    logger.error(f"✗ Could not find baseline results at {BASELINE_RESULTS}")
    exit(1)

try:
    distill_df = pd.read_csv(DISTILL_RESULTS)
    logger.info(f"✓ Loaded distillation results from {DISTILL_RESULTS}")
except FileNotFoundError:
    logger.error(f"✗ Could not find distillation results at {DISTILL_RESULTS}")
    exit(1)

# =========================
# COMPARISON ANALYSIS
# =========================
logger.info("\n" + "="*80)
logger.info("BASELINE VS DISTILLATION COMPARISON")
logger.info("="*80)

# Get unique student sizes
student_sizes = sorted(baseline_df['student_size'].unique())

comparison_data = []

logger.info("\n" + "-"*80)
logger.info(f"{'Size':<10} {'Baseline':<15} {'Distilled':<15} {'Gain':<15} {'Gain %':<10}")
logger.info("-"*80)

for size in student_sizes:
    base_row = baseline_df[baseline_df['student_size'] == size]
    dist_rows = distill_df[distill_df['student_size'] == size]
    
    if base_row.empty:
        logger.warning(f"No baseline results for {size}")
        continue
    
    base_auc = base_row['best_macro_auroc'].values[0]
    base_params = base_row['student_params'].values[0]
    base_time = base_row['training_duration_sec'].values[0]
    
    if not dist_rows.empty:
        # Average across all teachers if multiple
        dist_auc = dist_rows['best_macro_auroc'].max()
        dist_params = dist_rows['student_params'].values[0]
        dist_time = dist_rows['training_duration_sec'].mean()
        
        gain = dist_auc - base_auc
        gain_pct = (gain / base_auc) * 100
        
        logger.info(
            f"{size:<10} {base_auc:>8.4f}      "
            f"{dist_auc:>8.4f}      "
            f"{gain:>+8.4f}      "
            f"{gain_pct:>+7.2f}%"
        )
        
        comparison_data.append({
            'student_size': size,
            'baseline_auroc': base_auc,
            'distilled_auroc': dist_auc,
            'auc_gain': gain,
            'auc_gain_pct': gain_pct,
            'baseline_time_sec': base_time,
            'distilled_time_sec': dist_time,
            'time_ratio': dist_time / base_time,
            'params': base_params,
        })
    else:
        logger.warning(f"No distillation results for {size}")

logger.info("-"*80)

# =========================
# SUMMARY STATISTICS
# =========================
comparison_df = pd.DataFrame(comparison_data)

if not comparison_df.empty:
    logger.info("\nSUMMARY STATISTICS:")
    logger.info(f"Average KD Gain: {comparison_df['auc_gain'].mean():+.4f} AUROC")
    logger.info(f"Average KD Gain %: {comparison_df['auc_gain_pct'].mean():+.2f}%")
    logger.info(f"Min Gain: {comparison_df['auc_gain'].min():+.4f} ({comparison_df.loc[comparison_df['auc_gain'].idxmin(), 'student_size']})")
    logger.info(f"Max Gain: {comparison_df['auc_gain'].max():+.4f} ({comparison_df.loc[comparison_df['auc_gain'].idxmax(), 'student_size']})")
    
    # Training time comparison
    logger.info(f"\nTraining Time Comparison:")
    logger.info(f"Average baseline time: {comparison_df['baseline_time_sec'].mean()/60:.2f} min")
    logger.info(f"Average distilled time: {comparison_df['distilled_time_sec'].mean()/60:.2f} min")
    logger.info(f"Average slowdown: {comparison_df['time_ratio'].mean():.2f}x")
    
    # Save comparison CSV
    comparison_csv = f'{OUTPUT_DIR}baseline_vs_distilled.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info(f"\n✓ Comparison saved to: {comparison_csv}")

# =========================
# VISUALIZATION
# =========================
try:
    sns.set_style("whitegrid")
    
    # Figure 1: AUROC Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: AUROC by size
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, comparison_df['baseline_auroc'], width, 
                label='Baseline', alpha=0.8, color='steelblue')
    axes[0].bar(x_pos + width/2, comparison_df['distilled_auroc'], width,
                label='Distilled', alpha=0.8, color='coral')
    axes[0].set_xlabel('Student Size', fontsize=11)
    axes[0].set_ylabel('Macro AUROC', fontsize=11)
    axes[0].set_title('AUROC: Baseline vs Distilled', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison_df['student_size'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: KD Gain
    colors = ['green' if x > 0 else 'red' for x in comparison_df['auc_gain_pct']]
    axes[1].barh(comparison_df['student_size'], comparison_df['auc_gain_pct'], 
                 color=colors, alpha=0.8)
    axes[1].set_xlabel('AUROC Gain (%)', fontsize=11)
    axes[1].set_title('Knowledge Distillation Gain', fontsize=12, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(comparison_df['auc_gain_pct']):
        axes[1].text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}auroc_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ AUROC comparison plot saved: auroc_comparison.png")
    plt.close()
    
    # Figure 2: Training Time Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time by size
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, comparison_df['baseline_time_sec']/60, width,
                label='Baseline', alpha=0.8, color='steelblue')
    axes[0].bar(x_pos + width/2, comparison_df['distilled_time_sec']/60, width,
                label='Distilled', alpha=0.8, color='coral')
    axes[0].set_xlabel('Student Size', fontsize=11)
    axes[0].set_ylabel('Training Time (minutes)', fontsize=11)
    axes[0].set_title('Training Time: Baseline vs Distilled', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison_df['student_size'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Time slowdown ratio
    axes[1].barh(comparison_df['student_size'], comparison_df['time_ratio'],
                 color='darkorange', alpha=0.8)
    axes[1].set_xlabel('Time Ratio (Distilled / Baseline)', fontsize=11)
    axes[1].set_title('Distillation Training Overhead', fontsize=12, fontweight='bold')
    axes[1].axvline(x=1, color='black', linestyle='--', linewidth=1, label='No overhead')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(comparison_df['time_ratio']):
        axes[1].text(v + 0.05, i, f'{v:.2f}x', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}time_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Time comparison plot saved: time_comparison.png")
    plt.close()
    
    # Figure 3: Efficiency Plot (AUROC gain vs time overhead)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(comparison_df['auc_gain_pct'], comparison_df['time_ratio'],
                        s=300, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
    
    # Add labels for each point
    for idx, row in comparison_df.iterrows():
        ax.annotate(row['student_size'], 
                   (row['auc_gain_pct'], row['time_ratio']),
                   xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('AUROC Gain from KD (%)', fontsize=12)
    ax.set_ylabel('Training Time Overhead (x)', fontsize=12)
    ax.set_title('KD Efficiency: Performance Gain vs Time Cost', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No overhead')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}efficiency_plot.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Efficiency plot saved: efficiency_plot.png")
    plt.close()
    
    logger.info(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    
except Exception as e:
    logger.warning(f"Could not generate plots: {e}")

# =========================
# DETAILED REPORT
# =========================
report_path = f'{OUTPUT_DIR}comparison_report.txt'

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BASELINE vs DISTILLATION COMPARISON REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Generated: {pd.Timestamp.now()}\n\n")
    
    f.write("DETAILED RESULTS:\n")
    f.write("-"*80 + "\n")
    f.write(comparison_df.to_string())
    f.write("\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    
    if not comparison_df.empty:
        best_gain_idx = comparison_df['auc_gain'].idxmax()
        best_gain_size = comparison_df.loc[best_gain_idx, 'student_size']
        best_gain_val = comparison_df.loc[best_gain_idx, 'auc_gain']
        best_gain_pct = comparison_df.loc[best_gain_idx, 'auc_gain_pct']
        
        f.write(f"\n1. LARGEST KD GAIN:\n")
        f.write(f"   Student: {best_gain_size}\n")
        f.write(f"   Gain: {best_gain_val:+.4f} AUROC ({best_gain_pct:+.2f}%)\n")
        
        worst_gain_idx = comparison_df['auc_gain'].idxmin()
        worst_gain_size = comparison_df.loc[worst_gain_idx, 'student_size']
        worst_gain_val = comparison_df.loc[worst_gain_idx, 'auc_gain']
        worst_gain_pct = comparison_df.loc[worst_gain_idx, 'auc_gain_pct']
        
        f.write(f"\n2. SMALLEST KD GAIN:\n")
        f.write(f"   Student: {worst_gain_size}\n")
        f.write(f"   Gain: {worst_gain_val:+.4f} AUROC ({worst_gain_pct:+.2f}%)\n")
        
        f.write(f"\n3. AVERAGE STATISTICS:\n")
        f.write(f"   Average AUROC Gain: {comparison_df['auc_gain'].mean():+.4f}\n")
        f.write(f"   Average % Gain: {comparison_df['auc_gain_pct'].mean():+.2f}%\n")
        f.write(f"   Average Time Overhead: {comparison_df['time_ratio'].mean():.2f}x\n")
        f.write(f"   Average Baseline Time: {comparison_df['baseline_time_sec'].mean()/60:.2f} min\n")
        f.write(f"   Average Distilled Time: {comparison_df['distilled_time_sec'].mean()/60:.2f} min\n")
        
        f.write(f"\n4. RECOMMENDATIONS:\n")
        best_efficiency_idx = (comparison_df['auc_gain'] / comparison_df['time_ratio']).idxmax()
        best_efficiency_size = comparison_df.loc[best_efficiency_idx, 'student_size']
        f.write(f"   Best efficiency (gain/time): {best_efficiency_size}\n")
        
        if comparison_df['auc_gain'].mean() > 0.03:
            f.write(f"   KD is highly beneficial (avg {comparison_df['auc_gain'].mean():.4f} gain)\n")
        else:
            f.write(f"   KD provides modest improvements\n")

logger.info(f"✓ Report saved to: {report_path}")

# =========================
# FINAL SUMMARY
# =========================
logger.info("\n" + "="*80)
logger.info("COMPARISON COMPLETE")
logger.info("="*80)
logger.info(f"\nOutput files saved to: {OUTPUT_DIR}")
logger.info("  - baseline_vs_distilled.csv (detailed metrics)")
logger.info("  - auroc_comparison.png (performance comparison)")
logger.info("  - time_comparison.png (training time analysis)")
logger.info("  - efficiency_plot.png (gain vs cost analysis)")
logger.info("  - comparison_report.txt (detailed report)")