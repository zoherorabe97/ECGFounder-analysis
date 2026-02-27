# =========================
# Imports
# =========================
import os
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from util import save_checkpoint
from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from dataset import LVEF_12lead_cls_Dataset

# =========================
# BATCHNORM CONVERSION UTILITIES
# =========================
def convert_batchnorm_to_groupnorm(module, num_groups=32):
    """
    Recursively convert BatchNorm layers to GroupNorm while preserving weights.
    This is more controlled than ModuleValidator.fix() for pretrained models.
    """
    module_output = module
    if isinstance(module, nn.BatchNorm1d):
        num_channels = module.num_features
        # Use fewer groups if num_channels < num_groups
        groups = min(num_groups, num_channels)
        # Ensure num_channels is divisible by groups
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        
        module_output = nn.GroupNorm(
            num_groups=groups,
            num_channels=num_channels,
            eps=module.eps,
            affine=module.affine
        )
        
        # Copy weights and biases if they exist
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
    
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm_to_groupnorm(child, num_groups))
    
    del module
    return module_output

def freeze_normalization_layers(model):
    """
    Freeze all normalization layers (GroupNorm, LayerNorm, etc.)
    This prevents them from being updated during training.
    """
    frozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
            for param in module.parameters():
                param.requires_grad = False
            # Set to eval mode to use running stats
            module.eval()
            frozen_count += 1
    return model, frozen_count

# =========================
# ARGUMENT PARSING
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning with Differential Privacy using Opacus")
    
    # Training Parameters
    parser.add_argument("--linear_probe", action="store_true", help="True = Linear Probe | False = Full Fine-tuning")
    parser.add_argument("--num_lead", type=int, default=12)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (full FT or private)")
    parser.add_argument("--lp_lr", type=float, default=1e-3, help="Linear probe learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early_stop_lr", type=float, default=1e-5)
    
    # Privacy Parameters
    parser.add_argument("--privacy_budget", type=float, default=8.0, help="Target epsilon for DP")
    parser.add_argument("--max_clipping", type=float, default=1.0, help="Max grad norm clipping for DP")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta (usually 1/N)")
    parser.add_argument("--accountant", type=str, default="rdp", choices=["rdp", "gdp", "prv"], help="Opacus accountant type")
    parser.add_argument("--poisson_sampling", action="store_true", help="Use poisson sampling for DP")
    
    # Paths
    parser.add_argument("--train_csv", type=str, default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv')
    parser.add_argument("--test_csv", type=str, default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv')
    parser.add_argument("--ecg_path", type=str, default='')
    parser.add_argument("--pth", type=str, default='12_lead_ECGFounder.pth')
    parser.add_argument("--saved_dir", type=str, default='./res/eval_dp/')
    parser.add_argument("--log_file", type=str, default='logging/g12ec_dp_ft.log')
    
    # Data validation
    parser.add_argument("--validate_data", action="store_true", help="Validate dataset before training")
    
    # Normalization layer handling
    parser.add_argument("--freeze_norm", action="store_true", help="Freeze normalization layers (recommended for DP)")
    parser.add_argument("--num_groups", type=int, default=32, help="Number of groups for GroupNorm conversion")
    
    return parser.parse_args()

def validate_dataset(dataset, logger, max_samples=None):
    """
    Validate dataset for NaN/Inf values before training.
    This is done BEFORE training to avoid privacy leakage.
    """
    logger.info("Validating dataset for corrupted samples...")
    corrupted_indices = []
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    for idx in tqdm(range(num_samples), desc="Validating data"):
        try:
            x, y = dataset[idx]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                corrupted_indices.append(idx)
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            corrupted_indices.append(idx)
    
    if corrupted_indices:
        logger.warning(f"Found {len(corrupted_indices)} corrupted samples: {corrupted_indices[:10]}...")
        logger.warning("Consider cleaning your dataset before training with DP")
    else:
        logger.info("Dataset validation passed - no corrupted samples found")
    
    return corrupted_indices

def main():
    args = parse_args()

    os.makedirs(args.saved_dir, exist_ok=True)
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

    # =========================
    # DEVICE
    # =========================
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # =========================
    # DATA
    # =========================
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    labels = train_df.columns[4:].tolist()
    n_classes = len(labels)

    train_dataset = LVEF_12lead_cls_Dataset(args.ecg_path, train_df)
    test_dataset  = LVEF_12lead_cls_Dataset(args.ecg_path, test_df)

    # Validate datasets if requested (recommended for DP training)
    if args.validate_data:
        validate_dataset(train_dataset, logger, max_samples=1000)
        validate_dataset(test_dataset, logger, max_samples=500)

    # Use standard DataLoader here; Opacus will wrap it later
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24)
    testloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {n_classes}")

    # =========================
    # MODEL LOADING
    # =========================
    logger.info("Loading pretrained model...")
    model = ft_12lead_ECGFounder(
        device=device,
        pth=args.pth,
        n_classes=n_classes,
        linear_prob=args.linear_probe
    ).to(device)

    # =========================
    # CONVERT BATCHNORM TO GROUPNORM
    # =========================
    logger.info("Converting BatchNorm to GroupNorm for DP compatibility...")
    logger.info(f"Using {args.num_groups} groups for GroupNorm")
    model = convert_batchnorm_to_groupnorm(model, num_groups=args.num_groups)
    model = model.to(device)
    
    # Verify the model is now compatible
    logger.info("Validating model compatibility with Opacus...")
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Remaining compatibility issues: {errors}")
        logger.info("Applying ModuleValidator.fix() as backup...")
        model = ModuleValidator.fix(model)
        model = model.to(device)
    else:
        logger.info("✓ Model is fully DP-compatible!")
    
    # =========================
    # FREEZE NORMALIZATION LAYERS (OPTIONAL)
    # =========================
    if args.freeze_norm:
        logger.info("Freezing normalization layers...")
        model, frozen_count = freeze_normalization_layers(model)
        logger.info(f"✓ Frozen {frozen_count} normalization layers")
    
    # =========================
    # FREEZE / UNFREEZE FOR LINEAR PROBE OR FULL FT
    # =========================
    if args.linear_probe:
        logger.info("Mode: Linear Probing (backbone frozen)")
        logger.info("Note: Privacy guarantees apply only to trainable parameters")
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                # Don't override normalization layer freezing
                if not args.freeze_norm:
                    param.requires_grad = False
    else:
        logger.info("Mode: Full Fine-tuning with DP")
        for name, param in model.named_parameters():
            # Check if this parameter belongs to a normalization layer
            is_norm_param = False
            if args.freeze_norm and '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                try:
                    parent_module = model.get_submodule(parent_name)
                    if isinstance(parent_module, (nn.GroupNorm, nn.LayerNorm)):
                        is_norm_param = True
                except:
                    pass
            
            # Only set requires_grad if not a frozen norm parameter
            if not is_norm_param:
                param.requires_grad = True

    # =========================
    # OPTIMIZER (created AFTER model modifications)
    # =========================
    learning_rate = args.lp_lr if args.linear_probe else args.lr
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=args.weight_decay
    )

    # =========================
    # OPACUS PRIVACY ENGINE
    # =========================
    logger.info(f"Initializing Privacy Engine (Budget={args.privacy_budget}, Delta={args.delta}, Max Clipping={args.max_clipping})")
    privacy_engine = PrivacyEngine(accountant=args.accountant)

    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=args.privacy_budget,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_clipping,
        poisson_sampling=args.poisson_sampling,
    )

    # =========================
    # SCHEDULER & CRITERION
    # =========================
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.1, mode='max'
    )
    criterion = nn.BCEWithLogitsLoss()

    # =========================
    # LOG PARAMS & DP CONFIGURATION
    # =========================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    logger.info(f"Fine-tuning mode: {'Linear Probe' if args.linear_probe else 'Full Fine-tuning'}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")
    logger.info(f"Frozen params: {total_params - trainable_params:,} ({100 - trainable_percentage:.2f}%)")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Get actual DP parameters from Opacus
    logger.info("=" * 60)
    logger.info("DIFFERENTIAL PRIVACY CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Privacy Budget (ε): {args.privacy_budget}")
    logger.info(f"Delta (δ): {args.delta}")
    logger.info(f"Max Gradient Clipping (C): {optimizer.max_grad_norm}")
    logger.info(f"Noise Multiplier (σ): {optimizer.noise_multiplier:.6f}")
    logger.info(f"Accountant: {args.accountant}")
    logger.info(f"Poisson Sampling: {args.poisson_sampling}")
    logger.info(f"Expected Gradient Noise Scale: {optimizer.noise_multiplier * optimizer.max_grad_norm:.6f}")
    logger.info(f"Normalization layers frozen: {args.freeze_norm}")
    logger.info("=" * 60)

    # =========================
    # TRAINING + TESTING
    # =========================
    best_test_auroc = 0.0
    global_step = 0
    best_checkpoint_path = None

    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.epochs}")
        
        # -------- TRAIN --------
        model.train()
        
        # Keep normalization layers in eval mode if frozen
        if args.freeze_norm:
            for module in model.modules():
                if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                    module.eval()
        
        train_loss = 0.0
        all_per_sample_norms = []
        
        for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} Training")):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            
            # Collect per-sample gradient norms BEFORE optimizer step
            try:
                per_sample_norms = []
                for p in model.parameters():
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None and p.requires_grad:
                        # Flatten each sample's gradient and compute L2 norm
                        batch_size = p.grad_sample.size(0)
                        flat_grads = p.grad_sample.reshape(batch_size, -1)
                        norms_squared = (flat_grads ** 2).sum(dim=1)
                        per_sample_norms.append(norms_squared)
                
                if per_sample_norms:
                    # Sum squared norms across all parameters, then take sqrt
                    total_norms_squared = torch.stack(per_sample_norms).sum(dim=0)
                    total_norms = torch.sqrt(total_norms_squared)
                    all_per_sample_norms.extend(total_norms.cpu().numpy().tolist())
            except (AttributeError, RuntimeError) as e:
                # Silently skip if gradient tracking is not available
                pass
            
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            # Log gradient statistics every 100 batches
            if batch_idx > 0 and batch_idx % 100 == 0 and all_per_sample_norms:
                recent_norms = all_per_sample_norms[-min(100 * args.batch_size, len(all_per_sample_norms)):]
                grad_mean = np.mean(recent_norms)
                grad_std = np.std(recent_norms)
                grad_max = np.max(recent_norms)
                grad_min = np.min(recent_norms)
                
                # Calculate clipping rate
                clipped = sum(1 for norm in recent_norms if norm > optimizer.max_grad_norm)
                clip_rate = 100 * clipped / len(recent_norms)
                
                logger.info(
                    f"Batch {batch_idx}/{len(trainloader)} | "
                    f"Per-sample gradient norms - Mean: {grad_mean:.4f}, Std: {grad_std:.4f}, "
                    f"Max: {grad_max:.4f}, Min: {grad_min:.4f} | "
                    f"Clipping rate: {clip_rate:.1f}%"
                )
        
        avg_train_loss = train_loss / len(trainloader)
        
        # Log epoch-level gradient statistics
        if all_per_sample_norms:
            epoch_grad_mean = np.mean(all_per_sample_norms)
            epoch_grad_std = np.std(all_per_sample_norms)
            epoch_grad_max = np.max(all_per_sample_norms)
            epoch_grad_min = np.min(all_per_sample_norms)
            epoch_grad_median = np.median(all_per_sample_norms)
            
            # Calculate clipping statistics
            clipped_count = sum(1 for norm in all_per_sample_norms if norm > optimizer.max_grad_norm)
            clipping_rate = 100 * clipped_count / len(all_per_sample_norms)
            
            logger.info("=" * 60)
            logger.info(f"Epoch {epoch + 1} Gradient Statistics (before clipping & noise)")
            logger.info(f"Mean: {epoch_grad_mean:.4f}, Std: {epoch_grad_std:.4f}, Median: {epoch_grad_median:.4f}")
            logger.info(f"Max: {epoch_grad_max:.4f}, Min: {epoch_grad_min:.4f}")
            logger.info(f"Clipping rate: {clipping_rate:.2f}% (gradients > {optimizer.max_grad_norm})")
            logger.info("=" * 60)
        
        # Get privacy spent
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        logger.info(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Privacy: (ε = {epsilon:.2f}, δ = {args.delta})")

        # -------- TEST --------
        model.eval()
        all_gt, all_pred = [], []
        test_loss = 0.0

        with torch.no_grad():
            for x, y in tqdm(testloader, desc=f"Epoch {epoch + 1} Testing", leave=False):
                x, y = x.to(device), y.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item()
                
                pred = torch.sigmoid(logits)
                all_gt.append(y.cpu())
                all_pred.append(pred.cpu())

        avg_test_loss = test_loss / len(testloader)
        all_gt = torch.cat(all_gt)
        all_pred = torch.cat(all_pred)

        # Calculate metrics
        (
            test_macro_avg_prec,
            test_micro_avg_prec,
            test_macro_auroc,
            test_micro_auroc,
            test_challenge_metric
        ) = cal_multilabel_metrics(all_gt, all_pred, np.array(labels), 0.5)

        logger.info(
            f"Epoch {epoch + 1} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Test AUROC (Macro): {test_macro_auroc:.4f} | "
            f"Test AUROC (Micro): {test_micro_auroc:.4f}"
        )

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_macro_auroc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            logger.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")

        # -------- SAVE BEST --------
        if test_macro_auroc > best_test_auroc:
            # Delete previous best checkpoint if it exists
            if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                try:
                    os.remove(best_checkpoint_path)
                    logger.info(f"Removed previous checkpoint: {best_checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Could not remove old checkpoint: {e}")
            
            best_test_auroc = test_macro_auroc
            logger.info(f"New best model! AUROC: {best_test_auroc:.4f}")
            
            # Create descriptive filename
            tuning_type = "linear_probe" if args.linear_probe else "full_ft"
            norm_status = "frozen_norm" if args.freeze_norm else "trainable_norm"
            checkpoint_filename = f'best_{tuning_type}_{norm_status}_eps{args.privacy_budget}_clip{args.max_clipping}_auroc{test_macro_auroc:.4f}.pth'
            best_checkpoint_path = os.path.join(args.saved_dir, checkpoint_filename)
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auroc': test_macro_auroc,
                'epsilon': epsilon,
                'delta': args.delta,
                'noise_multiplier': optimizer.noise_multiplier,
                'max_grad_norm': optimizer.max_grad_norm,
                'freeze_norm': args.freeze_norm,
                'config': vars(args)
            }
            
            torch.save(checkpoint_state, best_checkpoint_path)
            logger.info(f"Best checkpoint saved: {best_checkpoint_path}")

        # -------- EARLY STOPPING --------
        if optimizer.param_groups[0]['lr'] < args.early_stop_lr:
            logger.info(f"Early stop: LR {optimizer.param_groups[0]['lr']:.2e} < {args.early_stop_lr:.2e}")
            break

    # Final privacy accounting
    final_epsilon = privacy_engine.get_epsilon(delta=args.delta)
    
    logger.info("=" * 60)
    logger.info("DP Training completed")
    logger.info(f"Best Test Macro AUROC: {best_test_auroc:.4f}")
    logger.info(f"Final Privacy Budget: (ε = {final_epsilon:.2f}, δ = {args.delta})")
    logger.info(f"Privacy Parameters: σ = {optimizer.noise_multiplier:.6f}, C = {optimizer.max_grad_norm}")
    logger.info(f"Normalization layers were: {'FROZEN' if args.freeze_norm else 'TRAINABLE'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()