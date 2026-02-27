# =========================
# Imports
# =========================
import os
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch_pruning as tp

from util import save_checkpoint
from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from dataset import LVEF_12lead_cls_Dataset

# =========================
# ARGUMENT PARSING
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Pruning for ECG Foundation Model")
    
    # Training Parameters
    parser.add_argument("--linear_probe", action="store_true", help="True = Linear Probe | False = Full Fine-tuning")
    parser.add_argument("--num_lead", type=int, default=12)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lp_lr", type=float, default=1e-3, help="Linear probe learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs_per_iteration", type=int, default=5, help="Epochs between pruning steps")
    parser.add_argument("--early_stop_lr", type=float, default=1e-6)
    
    # Pruning Parameters
    parser.add_argument("--pruning_method", type=str, default="taylor", 
                       choices=["magnitude", "taylor", "groupnorm"], 
                       help="Pruning importance criterion")
    parser.add_argument("--target_pruning_ratio", type=float, default=0.5, 
                       help="Target channel pruning ratio (0.5 = remove 50% channels)")
    parser.add_argument("--iterative_steps", type=int, default=5, 
                       help="Number of iterative pruning steps")
    parser.add_argument("--global_pruning", action="store_true", 
                       help="Use global pruning across all layers")
    parser.add_argument("--isomorphic", action="store_true",
                       help="Use isomorphic pruning (recommended for foundation models)")
    parser.add_argument("--round_to", type=int, default=8, 
                       help="Round channels to multiples of this number (4 or 8 for GPU efficiency)")
    
    # Sparse Training (for GroupNorm/BN pruners)
    parser.add_argument("--reg", type=float, default=1e-4, 
                       help="Regularization coefficient for sparse training")
    parser.add_argument("--use_sparse_training", action="store_true",
                       help="Enable sparse training (only for groupnorm method)")
    
    # Paths
    parser.add_argument("--train_csv", type=str, 
                       default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv')
    parser.add_argument("--test_csv", type=str, 
                       default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv')
    parser.add_argument("--ecg_path", type=str, default='')
    parser.add_argument("--pth", type=str, default='12_lead_ECGFounder.pth')
    parser.add_argument("--saved_dir", type=str, default='./res/pruned/')
    parser.add_argument("--log_file", type=str, default='logging/ecg_pruning.log')
    
    # Checkpointing
    parser.add_argument("--save_all_iterations", action="store_true", 
                       help="Save model at each pruning iteration")
    
    return parser.parse_args()

# =========================
# LOGGING SETUP
# =========================
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
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
    
    return logger

# =========================
# TRAINING FUNCTION
# =========================
def train_one_epoch(model, trainloader, criterion, optimizer, pruner, device, logger, use_sparse_training=False):
    """Train for one epoch with optional sparse training"""
    model.train()
    train_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc="Training", leave=False)):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        
        # Apply sparse training regularization if enabled
        if use_sparse_training and pruner is not None:
            pruner.regularize(model)
        
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(trainloader)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, testloader, criterion, device, labels):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    all_gt, all_pred = [], []
    
    with torch.no_grad():
        for x, y in tqdm(testloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            
            pred = torch.sigmoid(logits)
            all_gt.append(y.cpu())
            all_pred.append(pred.cpu())
    
    all_gt = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)
    
    # Calculate metrics
    test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc, test_challenge_metric = \
        cal_multilabel_metrics(all_gt, all_pred, np.array(labels), 0.5)
    
    avg_test_loss = test_loss / len(testloader)
    
    return avg_test_loss, test_macro_auroc, test_micro_auroc

# =========================
# MAIN FUNCTION
# =========================
def main():
    args = parse_args()
    
    os.makedirs(args.saved_dir, exist_ok=True)
    logger = setup_logger(args.log_file)
    
    # =========================
    # DEVICE
    # =========================
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # =========================
    # DATA
    # =========================
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    labels = train_df.columns[4:].tolist()
    n_classes = len(labels)
    
    train_dataset = LVEF_12lead_cls_Dataset(args.ecg_path, train_df)
    test_dataset = LVEF_12lead_cls_Dataset(args.ecg_path, test_df)
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {n_classes}")
    
    # =========================
    # MODEL
    # =========================
    logger.info("Loading pretrained model...")
    model = ft_12lead_ECGFounder(
        device=device,
        pth=args.pth,
        n_classes=n_classes,
        linear_prob=args.linear_probe
    ).to(device)
    
    # =========================
    # EXAMPLE INPUTS (required for dependency graph)
    # =========================
    # Get sample input shape from dataset
    sample_x, _ = train_dataset[0]
    example_inputs = torch.randn(1, *sample_x.shape).to(device)
    
    # =========================
    # CONFIGURE PRUNER
    # =========================
    # Set up ignored layers (don't prune final classifier)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == n_classes:
            ignored_layers.append(m)
            logger.info(f"Ignoring final classifier layer with {m.out_features} outputs")
    
    # Choose importance criterion
    if args.pruning_method == "magnitude":
        logger.info("Using GroupMagnitudeImportance (L2 norm)")
        importance = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
    elif args.pruning_method == "taylor":
        logger.info("Using GroupTaylorImportance (first-order Taylor expansion)")
        importance = tp.importance.GroupTaylorImportance()
        pruner_class = tp.pruner.MagnitudePruner  # Can use base pruner with Taylor importance
    elif args.pruning_method == "groupnorm":
        logger.info("Using GroupNormPruner with sparse training")
        importance = tp.importance.GroupNormImportance(p=2)
        pruner_class = tp.pruner.GroupNormPruner
    else:
        raise ValueError(f"Unknown pruning method: {args.pruning_method}")
    
    # Initialize pruner
    logger.info("=" * 60)
    logger.info("PRUNING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Method: {args.pruning_method}")
    logger.info(f"Target pruning ratio: {args.target_pruning_ratio}")
    logger.info(f"Iterative steps: {args.iterative_steps}")
    logger.info(f"Global pruning: {args.global_pruning}")
    logger.info(f"Isomorphic pruning: {args.isomorphic}")
    logger.info(f"Round channels to: {args.round_to}")
    logger.info(f"Sparse training: {args.use_sparse_training}")
    logger.info("=" * 60)
    
    pruner_kwargs = {
        "model": model,
        "example_inputs": example_inputs,
        "importance": importance,
        "iterative_steps": args.iterative_steps,
        "pruning_ratio": args.target_pruning_ratio,
        "global_pruning": args.global_pruning,
        "ignored_layers": ignored_layers,
        "round_to": args.round_to,
    }
    
    # Add isomorphic if supported
    if args.isomorphic:
        pruner_kwargs["isomorphic"] = True
    
    # Add regularization for GroupNormPruner
    if args.pruning_method == "groupnorm":
        pruner_kwargs["reg"] = args.reg
    
    pruner = pruner_class(**pruner_kwargs)
    
    # =========================
    # CRITERION
    # =========================
    criterion = nn.BCEWithLogitsLoss()
    
    # =========================
    # COMPUTE INITIAL GRADIENTS (Required for Taylor importance)
    # =========================
    if args.pruning_method == "taylor":
        logger.info("Computing initial gradients for Taylor importance...")
        model.train()
        optimizer_temp = optim.Adam(model.parameters(), lr=1e-4)
        
        # Run one batch to compute gradients
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer_temp.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            break  # Only need one batch for gradient computation
        
        logger.info("Gradients computed successfully")
    
    # =========================
    # BASELINE PERFORMANCE
    # =========================
    logger.info("=" * 60)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("=" * 60)
    
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Baseline MACs: {base_macs/1e9:.4f} G")
    logger.info(f"Baseline Params: {base_params/1e6:.4f} M")
    
    baseline_loss, baseline_macro_auroc, baseline_micro_auroc = evaluate(
        model, testloader, criterion, device, labels
    )
    logger.info(f"Baseline Test Loss: {baseline_loss:.4f}")
    logger.info(f"Baseline Macro AUROC: {baseline_macro_auroc:.4f}")
    logger.info(f"Baseline Micro AUROC: {baseline_micro_auroc:.4f}")
    
    # =========================
    # ITERATIVE PRUNING LOOP
    # =========================
    best_auroc = baseline_macro_auroc
    best_model_state = None
    
    for iteration in range(args.iterative_steps):
        logger.info("=" * 60)
        logger.info(f"PRUNING ITERATION {iteration + 1}/{args.iterative_steps}")
        logger.info("=" * 60)
        
        # Compute fresh gradients before each pruning step (for Taylor importance)
        if args.pruning_method == "taylor":
            logger.info("Computing gradients before pruning...")
            model.train()
            optimizer_temp = optim.Adam(model.parameters(), lr=1e-4)
            
            # Run a few batches to get stable gradients
            for batch_idx, (x, y) in enumerate(trainloader):
                if batch_idx >= 3:  # Use 3 batches for more stable gradient estimation
                    break
                x, y = x.to(device), y.to(device)
                optimizer_temp.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
            
            logger.info("Gradients updated")
        
        # Perform one pruning step
        logger.info("Performing pruning step...")
        pruner.step()
        
        # Count model size after pruning
        current_macs, current_params = tp.utils.count_ops_and_params(model, example_inputs)
        macs_reduction = (1 - current_macs / base_macs) * 100
        params_reduction = (1 - current_params / base_params) * 100
        
        logger.info(f"After pruning iteration {iteration + 1}:")
        logger.info(f"  MACs: {current_macs/1e9:.4f} G ({macs_reduction:.2f}% reduced)")
        logger.info(f"  Params: {current_params/1e6:.4f} M ({params_reduction:.2f}% reduced)")
        
        # Re-initialize optimizer for pruned model
        learning_rate = args.lp_lr if args.linear_probe else args.lr
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=args.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, mode='max'
        )
        
        # Fine-tune the pruned model
        logger.info(f"Fine-tuning for {args.epochs_per_iteration} epochs...")
        best_iter_auroc = 0.0
        
        for epoch in range(args.epochs_per_iteration):
            # Training
            train_loss = train_one_epoch(
                model, trainloader, criterion, optimizer, 
                pruner if args.use_sparse_training else None,
                device, logger, args.use_sparse_training
            )
            
            # Evaluation
            test_loss, test_macro_auroc, test_micro_auroc = evaluate(
                model, testloader, criterion, device, labels
            )
            
            logger.info(
                f"Iter {iteration + 1}, Epoch {epoch + 1}/{args.epochs_per_iteration} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Macro AUROC: {test_macro_auroc:.4f} | "
                f"Micro AUROC: {test_micro_auroc:.4f}"
            )
            
            # Update learning rate and log if changed
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(test_macro_auroc)
            new_lr = optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                logger.info(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Track best in this iteration
            if test_macro_auroc > best_iter_auroc:
                best_iter_auroc = test_macro_auroc
            
            # Early stopping
            if optimizer.param_groups[0]['lr'] < args.early_stop_lr:
                logger.info(f"Early stop: LR {optimizer.param_groups[0]['lr']:.2e} < {args.early_stop_lr:.2e}")
                break
        
        # Save iteration checkpoint if requested
        if args.save_all_iterations:
            iter_checkpoint_path = os.path.join(
                args.saved_dir, 
                f"iter_{iteration + 1}_ratio_{args.target_pruning_ratio}_auroc_{best_iter_auroc:.4f}.pth"
            )
            model.zero_grad()
            torch.save({
                'iteration': iteration + 1,
                'model': model,
                'auroc': best_iter_auroc,
                'macs': current_macs,
                'params': current_params,
                'config': vars(args)
            }, iter_checkpoint_path)
            logger.info(f"Saved iteration checkpoint: {iter_checkpoint_path}")
        
        # Track global best
        if best_iter_auroc > best_auroc:
            best_auroc = best_iter_auroc
            # Deep copy the entire model, not just state_dict (since structure changes during pruning)
            best_model_state = copy.deepcopy(model)
            logger.info(f"New best AUROC: {best_auroc:.4f}")
    
    # =========================
    # FINAL RESULTS
    # =========================
    logger.info("=" * 60)
    logger.info("PRUNING COMPLETED")
    logger.info("=" * 60)
    
    final_macs, final_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    logger.info(f"Baseline -> Final:")
    logger.info(f"  MACs: {base_macs/1e9:.4f} G -> {final_macs/1e9:.4f} G ({(1-final_macs/base_macs)*100:.2f}% reduction)")
    logger.info(f"  Params: {base_params/1e6:.4f} M -> {final_params/1e6:.4f} M ({(1-final_params/base_params)*100:.2f}% reduction)")
    logger.info(f"  Macro AUROC: {baseline_macro_auroc:.4f} -> {best_auroc:.4f} (Δ {best_auroc - baseline_macro_auroc:+.4f})")
    
    # Save final best model
    final_model_path = os.path.join(
        args.saved_dir,
        f"pruned_{args.pruning_method}_ratio_{args.target_pruning_ratio}_auroc_{best_auroc:.4f}.pth"
    )
    
    # Use the best model if we have it, otherwise use current model
    if best_model_state is not None:
        save_model = best_model_state
    else:
        save_model = model
    
    save_model.zero_grad()
    torch.save({
        'model': save_model,
        'auroc': best_auroc,
        'baseline_auroc': baseline_macro_auroc,
        'macs': final_macs,
        'params': final_params,
        'baseline_macs': base_macs,
        'baseline_params': base_params,
        'config': vars(args)
    }, final_model_path)
    
    logger.info(f"Final pruned model saved: {final_model_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()