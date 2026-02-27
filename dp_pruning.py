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
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from util import save_checkpoint
from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from dataset import LVEF_12lead_cls_Dataset

# =========================
# BATCHNORM CONVERSION (for DP compatibility)
# =========================
def convert_batchnorm_to_groupnorm(module, num_groups=32):
    """
    Recursively convert BatchNorm layers to GroupNorm while preserving weights.
    Required for both DP and pruning compatibility.
    """
    module_output = module
    if isinstance(module, nn.BatchNorm1d):
        num_channels = module.num_features
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        
        module_output = nn.GroupNorm(
            num_groups=groups,
            num_channels=num_channels,
            eps=module.eps,
            affine=module.affine
        )
        
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
    
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm_to_groupnorm(child, num_groups))
    
    del module
    return module_output

def freeze_normalization_layers(model):
    """Freeze all normalization layers"""
    frozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
            frozen_count += 1
    return model, frozen_count

# =========================
# ARGUMENT PARSING
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Pruning with Differential Privacy")
    
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
    parser.add_argument("--pruning_method", type=str, default="magnitude", 
                       choices=["magnitude", "taylor", "groupnorm"], 
                       help="Pruning importance criterion (magnitude recommended with DP)")
    parser.add_argument("--target_pruning_ratio", type=float, default=0.3, 
                       help="Target channel pruning ratio")
    parser.add_argument("--iterative_steps", type=int, default=5, 
                       help="Number of iterative pruning steps")
    parser.add_argument("--global_pruning", action="store_true", 
                       help="Use global pruning across all layers")
    parser.add_argument("--isomorphic", action="store_true",
                       help="Use isomorphic pruning")
    parser.add_argument("--round_to", type=int, default=8, 
                       help="Round channels to multiples of this number")
    
    # Differential Privacy Parameters
    parser.add_argument("--enable_dp", action="store_true",
                       help="Enable differential privacy during training")
    parser.add_argument("--privacy_budget", type=float, default=8.0, 
                       help="Target epsilon for DP (only if --enable_dp)")
    parser.add_argument("--max_clipping", type=float, default=1.0, 
                       help="Max grad norm clipping for DP")
    parser.add_argument("--delta", type=float, default=1e-5, 
                       help="Target delta (usually 1/N)")
    parser.add_argument("--accountant", type=str, default="rdp", 
                       choices=["rdp", "gdp", "prv"], 
                       help="Opacus accountant type")
    parser.add_argument("--poisson_sampling", action="store_true", 
                       help="Use poisson sampling for DP")
    parser.add_argument("--freeze_norm", action="store_true",
                       help="Freeze normalization layers (recommended with DP)")
    
    # Paths
    parser.add_argument("--train_csv", type=str, 
                       default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv')
    parser.add_argument("--test_csv", type=str, 
                       default='/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv')
    parser.add_argument("--ecg_path", type=str, default='')
    parser.add_argument("--pth", type=str, default='12_lead_ECGFounder.pth')
    parser.add_argument("--saved_dir", type=str, default='./res/dp_pruned/')
    parser.add_argument("--log_file", type=str, default='logging/dp_pruning.log')
    
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
def train_one_epoch(model, trainloader, criterion, optimizer, device, logger, 
                   enable_dp=False, privacy_engine=None, freeze_norm=False):
    """Train for one epoch with optional DP"""
    model.train()
    
    # Keep frozen normalization layers in eval mode
    if freeze_norm:
        for module in model.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                module.eval()
    
    train_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc="Training", leave=False)):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Log privacy spent if DP is enabled
    if enable_dp and privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=optimizer.target_delta if hasattr(optimizer, 'target_delta') else 1e-5)
        logger.info(f"  Current privacy: ε = {epsilon:.2f}")
    
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
    
    # Initial dataloaders (will be replaced by Opacus if DP is enabled)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {n_classes}")
    
    # =========================
    # MODEL LOADING & CONVERSION
    # =========================
    logger.info("Loading pretrained model...")
    model = ft_12lead_ECGFounder(
        device=device,
        pth=args.pth,
        n_classes=n_classes,
        linear_prob=args.linear_probe
    ).to(device)
    
    # Convert BatchNorm to GroupNorm (required for both pruning and DP)
    logger.info("Converting BatchNorm to GroupNorm for compatibility...")
    model = convert_batchnorm_to_groupnorm(model, num_groups=32)
    model = model.to(device)
    
    # Validate compatibility
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Compatibility issues found: {errors}")
        model = ModuleValidator.fix(model)
        model = model.to(device)
    
    # Freeze normalization layers if requested
    if args.freeze_norm:
        logger.info("Freezing normalization layers...")
        model, frozen_count = freeze_normalization_layers(model)
        logger.info(f"Frozen {frozen_count} normalization layers")
    
    # =========================
    # FREEZE/UNFREEZE FOR LINEAR PROBE
    # =========================
    if args.linear_probe:
        logger.info("Mode: Linear Probing (backbone frozen)")
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name or "dense" in name:
                param.requires_grad = True
            elif not args.freeze_norm:
                param.requires_grad = False
    else:
        logger.info("Mode: Full Fine-tuning")
        for name, param in model.named_parameters():
            is_norm_param = False
            if args.freeze_norm and '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                try:
                    parent_module = model.get_submodule(parent_name)
                    if isinstance(parent_module, (nn.GroupNorm, nn.LayerNorm)):
                        is_norm_param = True
                except:
                    pass
            if not is_norm_param:
                param.requires_grad = True
    
    # =========================
    # EXAMPLE INPUTS (required for dependency graph)
    # =========================
    sample_x, _ = train_dataset[0]
    example_inputs = torch.randn(1, *sample_x.shape).to(device)
    
    # =========================
    # CONFIGURE PRUNER
    # =========================
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == n_classes:
            ignored_layers.append(m)
            logger.info(f"Ignoring final classifier layer with {m.out_features} outputs")
    
    # Choose importance criterion
    if args.pruning_method == "magnitude":
        logger.info("Using GroupMagnitudeImportance (recommended with DP)")
        importance = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
    elif args.pruning_method == "taylor":
        logger.info("Using GroupTaylorImportance (requires gradients)")
        importance = tp.importance.GroupTaylorImportance()
        pruner_class = tp.pruner.MagnitudePruner
    elif args.pruning_method == "groupnorm":
        logger.info("Using GroupNormPruner")
        importance = tp.importance.GroupNormImportance(p=2)
        pruner_class = tp.pruner.GroupNormPruner
    
    # Initialize pruner
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Pruning method: {args.pruning_method}")
    logger.info(f"Target pruning ratio: {args.target_pruning_ratio}")
    logger.info(f"Iterative steps: {args.iterative_steps}")
    logger.info(f"Global pruning: {args.global_pruning}")
    logger.info(f"Isomorphic pruning: {args.isomorphic}")
    logger.info(f"Differential Privacy: {args.enable_dp}")
    if args.enable_dp:
        logger.info(f"  Privacy budget (ε): {args.privacy_budget}")
        logger.info(f"  Delta (δ): {args.delta}")
        logger.info(f"  Max clipping: {args.max_clipping}")
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
    
    if args.isomorphic:
        pruner_kwargs["isomorphic"] = True
    
    pruner = pruner_class(**pruner_kwargs)
    
    # =========================
    # CRITERION
    # =========================
    criterion = nn.BCEWithLogitsLoss()
    
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
    privacy_engine = None
    
    for iteration in range(args.iterative_steps):
        logger.info("=" * 60)
        logger.info(f"PRUNING ITERATION {iteration + 1}/{args.iterative_steps}")
        logger.info("=" * 60)
        
        # Compute gradients for Taylor importance
        if args.pruning_method == "taylor":
            logger.info("Computing gradients for Taylor importance...")
            model.train()
            optimizer_temp = optim.Adam(model.parameters(), lr=1e-4)
            
            for batch_idx, (x, y) in enumerate(trainloader):
                if batch_idx >= 3:
                    break
                x, y = x.to(device), y.to(device)
                optimizer_temp.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
            
            logger.info("Gradients updated")
        
        # Perform pruning
        logger.info("Performing pruning step...")
        pruner.step()
        
        # Count model size after pruning
        current_macs, current_params = tp.utils.count_ops_and_params(model, example_inputs)
        macs_reduction = (1 - current_macs / base_macs) * 100
        params_reduction = (1 - current_params / base_params) * 100
        
        logger.info(f"After pruning iteration {iteration + 1}:")
        logger.info(f"  MACs: {current_macs/1e9:.4f} G ({macs_reduction:.2f}% reduced)")
        logger.info(f"  Params: {current_params/1e6:.4f} M ({params_reduction:.2f}% reduced)")
        
        # =========================
        # PREPARE MODEL FOR NEW DP ITERATION
        # =========================
        if args.enable_dp and iteration > 0:
            logger.info("Preparing model for new DP iteration...")
            
            # Unwrap from Opacus if wrapped (get the actual model)
            if hasattr(model, 'remove_hooks'):
                model.remove_hooks()
            if hasattr(model, '_module'):
                model = model._module
            
            logger.info("Model unwrapped from DP wrapper, keeping pruned architecture")
            
            # Clear gradients and reset training state
            model.train()
            model.zero_grad()
            
            # Reapply gradient configuration for next iteration
            if args.linear_probe:
                for name, param in model.named_parameters():
                    if "classifier" in name or "fc" in name or "dense" in name:
                        param.requires_grad = True
                    elif not args.freeze_norm:
                        param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    is_norm_param = False
                    if args.freeze_norm and '.' in name:
                        parent_name = name.rsplit('.', 1)[0]
                        try:
                            parent_module = model.get_submodule(parent_name)
                            if isinstance(parent_module, (nn.GroupNorm, nn.LayerNorm)):
                                is_norm_param = True
                        except:
                            pass
                    if not is_norm_param:
                        param.requires_grad = True
            
            # Freeze normalization layers if needed
            if args.freeze_norm:
                logger.info("Freezing normalization layers...")
                model, _ = freeze_normalization_layers(model)
        
        # =========================
        # SETUP OPTIMIZER
        # =========================
        learning_rate = args.lp_lr if args.linear_probe else args.lr
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=args.weight_decay
        )
        
        # =========================
        # APPLY DIFFERENTIAL PRIVACY (if enabled)
        # =========================
        if args.enable_dp:
            logger.info("Initializing Privacy Engine for this iteration...")
            
            # Set model to training mode before Opacus validation
            model.train()
            
            # Keep frozen normalization layers in eval mode
            if args.freeze_norm:
                for module in model.modules():
                    if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                        module.eval()
            
            privacy_engine = PrivacyEngine(accountant=args.accountant)
            
            # Create fresh dataloader for this iteration
            trainloader_dp = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=24
            )
            
            model, optimizer, trainloader_current = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=trainloader_dp,
                target_epsilon=args.privacy_budget,
                target_delta=args.delta,
                epochs=args.epochs_per_iteration,
                max_grad_norm=args.max_clipping,
                poisson_sampling=args.poisson_sampling,
            )
            
            # Store target_delta for logging
            optimizer.target_delta = args.delta
            
            logger.info(f"DP enabled: ε={args.privacy_budget}, σ={optimizer.noise_multiplier:.4f}, C={optimizer.max_grad_norm}")
        else:
            trainloader_current = trainloader
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, mode='max'
        )
        
        # =========================
        # FINE-TUNE PRUNED MODEL
        # =========================
        logger.info(f"Fine-tuning for {args.epochs_per_iteration} epochs...")
        best_iter_auroc = 0.0
        
        for epoch in range(args.epochs_per_iteration):
            # Training
            train_loss = train_one_epoch(
                model, trainloader_current, criterion, optimizer, device, logger,
                enable_dp=args.enable_dp, privacy_engine=privacy_engine, freeze_norm=args.freeze_norm
            )
            
            # Evaluation
            test_loss, test_macro_auroc, test_micro_auroc = evaluate(
                model, testloader, criterion, device, labels
            )
            
            logger.info(
                f"Iter {iteration + 1}, Epoch {epoch + 1}/{args.epochs_per_iteration} | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                f"Macro AUROC: {test_macro_auroc:.4f} | Micro AUROC: {test_micro_auroc:.4f}"
            )
            
            # Update learning rate
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
            dp_suffix = ""
            if args.enable_dp and privacy_engine is not None:
                current_eps = privacy_engine.get_epsilon(delta=args.delta)
                dp_suffix = f"_eps_{current_eps:.2f}_clip_{args.max_clipping}"
            
            iter_checkpoint_path = os.path.join(
                args.saved_dir, 
                f"iter_{iteration + 1}_{'dp' if args.enable_dp else ''}{dp_suffix}_auroc_{best_iter_auroc:.4f}.pth"
            )
            model.zero_grad()
            
            checkpoint_dict = {
                'iteration': iteration + 1,
                'model': model,
                'auroc': best_iter_auroc,
                'macs': current_macs,
                'params': current_params,
                'config': vars(args)
            }
            
            if args.enable_dp and privacy_engine is not None:
                checkpoint_dict['epsilon'] = privacy_engine.get_epsilon(delta=args.delta)
                checkpoint_dict['delta'] = args.delta
            
            torch.save(checkpoint_dict, iter_checkpoint_path)
            logger.info(f"Saved iteration checkpoint: {iter_checkpoint_path}")
        
        # Track global best
        if best_iter_auroc > best_auroc:
            best_auroc = best_iter_auroc
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
    
    logger.info(f"  Macro AUROC: {baseline_macro_auroc:.4f} -> {best_auroc:.4f} (Δ {best_auroc - baseline_macro_auroc:+.4f})")
    
    # Save final model
    dp_suffix = ""
    final_epsilon = None
    if args.enable_dp and privacy_engine is not None:
        final_epsilon = privacy_engine.get_epsilon(delta=args.delta)
        dp_suffix = f"_eps_{final_epsilon:.2f}_clip_{args.max_clipping}"
        logger.info(f"Final Privacy Budget: ε = {final_epsilon:.2f}, δ = {args.delta}")

    final_model_path = os.path.join(
        args.saved_dir,
        f"pruned_{'dp' if args.enable_dp else ''}{dp_suffix}_{args.pruning_method}_ratio_{args.target_pruning_ratio}_auroc_{best_auroc:.4f}.pth"
    )
    
    save_model = best_model_state if best_model_state is not None else model
    save_model.zero_grad()
    
    checkpoint_dict = {
        'model': save_model,
        'auroc': best_auroc,
        'baseline_auroc': baseline_macro_auroc,
        'macs': final_macs,
        'params': final_params,
        'baseline_macs': base_macs,
        'baseline_params': base_params,
        'config': vars(args)
    }
    
    if final_epsilon is not None:
        checkpoint_dict['epsilon'] = final_epsilon
        checkpoint_dict['delta'] = args.delta
    
    torch.save(checkpoint_dict, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()