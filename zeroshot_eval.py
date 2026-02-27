# =========================
# Zero-Shot Classification Script
# =========================
# This script evaluates the pretrained ECGFounder model on a test dataset
# WITHOUT any training or fine-tuning (zero-shot inference)

import os
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from dataset import LVEF_12lead_cls_Dataset

# =========================
# ARGUMENT PARSING
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot classification with pretrained ECGFounder")
    
    # Model Parameters
    parser.add_argument("--num_lead", type=int, default=12)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Paths
    parser.add_argument("--test_csv", type=str, 
                       default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv',
                       help="Path to test CSV file")
    parser.add_argument("--ecg_path", type=str, default='', help="Base path for ECG data files")
    parser.add_argument("--pth", type=str, default='12_lead_ECGFounder.pth', 
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default='./res/zeroshot/',
                       help="Directory to save results")
    parser.add_argument("--log_file", type=str, default='logging/g12ec_zeroshot.log')
    
    # Evaluation options
    parser.add_argument("--num_workers", type=int, default=24, help="Number of dataloader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logging', exist_ok=True)

    # =========================
    # LOGGING
    # =========================
    file_handler = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Force flush after each log
    def flush_handlers():
        for handler in logger.handlers:
            handler.flush()
    
    original_info = logger.info
    def info_with_flush(msg, *args, **kwargs):
        original_info(msg, *args, **kwargs)
        flush_handlers()
    logger.info = info_with_flush

    logger.info("=" * 60)
    logger.info("ZERO-SHOT CLASSIFICATION WITH PRETRAINED ECGFounder")
    logger.info("=" * 60)

    # =========================
    # DEVICE
    # =========================
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # =========================
    # DATA
    # =========================
    test_df = pd.read_csv(args.test_csv)
    labels = test_df.columns[4:].tolist()
    n_classes = len(labels)

    test_dataset = LVEF_12lead_cls_Dataset(args.ecg_path, test_df)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)

    logger.info(f"Test dataset: {args.test_csv}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class labels: {labels}")

    # =========================
    # LOAD PRETRAINED MODEL
    # =========================
    logger.info(f"Loading pretrained model from: {args.pth}")
    
    model = ft_12lead_ECGFounder(
        device=device,
        pth=args.pth,
        n_classes=n_classes,
        linear_prob=False  # Load full model, but we won't train
    ).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info("Model loaded successfully - Running zero-shot inference...")

    # =========================
    # ZERO-SHOT INFERENCE
    # =========================
    criterion = nn.BCEWithLogitsLoss()
    
    all_gt = []
    all_pred = []
    all_logits = []
    test_loss = 0.0
    corrupted_batches = 0
    corrupted_samples = 0

    logger.info("Starting inference...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(testloader, desc="Zero-shot Inference")):
            x, y = x.to(device), y.to(device)
            
            # Validate input data
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                corrupted_batches += 1
                corrupted_samples += x.size(0)
                logger.warning(f"Batch {batch_idx} contains NaN/Inf in input data. Skipping.")
                continue
            
            # Forward pass
            logits = model(x)
            
            # Validate output
            if not torch.isfinite(logits).all():
                corrupted_batches += 1
                corrupted_samples += x.size(0)
                logger.warning(f"Batch {batch_idx} produced NaN/Inf outputs. Skipping.")
                continue
            
            # Compute loss
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                corrupted_batches += 1
                corrupted_samples += x.size(0)
                logger.warning(f"Batch {batch_idx} produced NaN/Inf loss. Skipping.")
                continue
            
            test_loss += loss.item()
            
            # Get predictions (apply sigmoid to logits)
            pred = torch.sigmoid(logits)
            
            # Validate predictions
            if not torch.isfinite(pred).all():
                corrupted_batches += 1
                corrupted_samples += x.size(0)
                logger.warning(f"Batch {batch_idx} produced NaN/Inf predictions. Skipping.")
                continue
            
            all_gt.append(y.cpu())
            all_pred.append(pred.cpu())
            all_logits.append(logits.cpu())

    # Log corruption statistics
    if corrupted_batches > 0:
        logger.warning(
            f"Corrupted batches: {corrupted_batches}/{len(testloader)} "
            f"({100*corrupted_batches/len(testloader):.2f}%) | "
            f"Corrupted samples: {corrupted_samples}"
        )

    # =========================
    # COMPUTE METRICS
    # =========================
    if len(all_gt) == 0:
        logger.error("No valid predictions! All batches were corrupted.")
        return

    avg_test_loss = test_loss / max(len(testloader) - corrupted_batches, 1)
    all_gt = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)
    all_logits = torch.cat(all_logits)

    logger.info("Computing metrics...")
    
    (
        test_macro_avg_prec,
        test_micro_avg_prec,
        test_macro_auroc,
        test_micro_auroc,
        test_challenge_metric
    ) = cal_multilabel_metrics(all_gt, all_pred, np.array(labels), args.threshold)

    # =========================
    # RESULTS
    # =========================
    logger.info("=" * 60)
    logger.info("ZERO-SHOT CLASSIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Macro AUROC: {test_macro_auroc:.4f}")
    logger.info(f"Test Micro AUROC: {test_micro_auroc:.4f}")
    logger.info(f"Test Macro Average Precision: {test_macro_avg_prec:.4f}")
    logger.info(f"Test Micro Average Precision: {test_micro_avg_prec:.4f}")
    logger.info(f"PhysioNet Challenge Metric: {test_challenge_metric:.4f}")
    logger.info("=" * 60)

    # =========================
    # SAVE RESULTS
    # =========================
    results = {
        'model': args.pth,
        'test_dataset': args.test_csv,
        'num_samples': len(test_dataset),
        'num_classes': n_classes,
        'threshold': args.threshold,
        'test_loss': float(avg_test_loss),
        'macro_auroc': float(test_macro_auroc),
        'micro_auroc': float(test_micro_auroc),
        'macro_avg_prec': float(test_macro_avg_prec),
        'micro_avg_prec': float(test_micro_avg_prec),
        'challenge_metric': float(test_challenge_metric),
        'corrupted_batches': corrupted_batches,
        'corrupted_samples': corrupted_samples
    }
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, 'zeroshot_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to: {results_file}")
    
    # Save predictions (optional - can be large)
    predictions_file = os.path.join(args.output_dir, 'zeroshot_predictions.npz')
    np.savez_compressed(
        predictions_file,
        ground_truth=all_gt.numpy(),
        predictions=all_pred.numpy(),
        logits=all_logits.numpy(),
        labels=labels
    )
    logger.info(f"Predictions saved to: {predictions_file}")
    
    logger.info("Zero-shot evaluation completed successfully!")

if __name__ == "__main__":
    main()
