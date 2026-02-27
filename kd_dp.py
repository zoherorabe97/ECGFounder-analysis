# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from net1d import Net1D
from dataset import LVEF_12lead_cls_Dataset

# =========================
# CONFIG
# =========================
num_lead = 12
gpu_id = 1
batch_size = 512
lr = 1e-4  # Adam learning rate (appropriate for Adam optimizer)
weight_decay = 0  # Disabled for DP (frozen norm layers already regularize)
early_stop_lr = 1e-6
Epochs = 10 # Increased epochs
temperature = 3.0  # Reduced from 4.0 (lower = sharper targets)
alpha = 0.3  # Reduced from 0.5 (more weight on hard labels for DP stability)

# Differential Privacy Parameters
TARGET_EPSILON = 8.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0

# Paths
train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv'
test_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
pretrained_pth = '12_lead_ECGFounder.pth'
saved_dir = './res/distill_dp/'
results_csv = './res/distill_dp/distillation_dp_results.csv'

# DP Teacher Model Path
DP_TEACHER_PATH = '/home/zoorab/projects/ECGFounder/res/eval_dp/best_full_ft_frozen_norm_eps8.0_clip1.0_auroc0.9002.pth'

# Student configurations
STUDENT_CONFIGS = {
    'tiny': {
        'base_filters': 16,
        'filter_list': [16, 32, 32, 64, 64, 128, 128],
        'm_blocks_list': [1, 1, 1, 1, 1, 1, 1],
        'description': '~10x compression, fastest inference'
    },
    'small': {
        'base_filters': 32,
        'filter_list': [32, 64, 64, 128, 128, 256, 256],
        'm_blocks_list': [1, 1, 1, 2, 2, 2, 2],
        'description': '~5x compression, balanced performance'
    },
    'medium': {
        'base_filters': 48,
        'filter_list': [48, 96, 96, 192, 192, 384, 384],
        'm_blocks_list': [1, 2, 2, 2, 2, 3, 3],
        'description': '~3x compression, higher accuracy'
    },
    'large': {
        'base_filters': 64,
        'filter_list': [64, 128, 128, 256, 256, 512, 512],
        'm_blocks_list': [2, 2, 2, 3, 3, 4, 4],
        'description': '~2x compression, closest to teacher'
    }
}

os.makedirs(saved_dir, exist_ok=True)
os.makedirs('logging', exist_ok=True)

# =========================
# LOGGING
# =========================
log_file = f'logging/distillation_dp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

for handler in logger.handlers:
    handler.setLevel(logging.INFO)
    if isinstance(handler, logging.FileHandler):
        handler.flush()

# =========================
# DEVICE
# =========================
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# =========================
# DATA
# =========================
logger.info("Loading datasets...")
labels = pd.read_csv(train_csv_path).columns[4:].tolist()
n_classes = len(labels)

train_dataset = LVEF_12lead_cls_Dataset(ecg_path='', labels_df=pd.read_csv(train_csv_path))
test_dataset = LVEF_12lead_cls_Dataset(ecg_path='', labels_df=pd.read_csv(test_csv_path))

trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle=False)

logger.info(f"Train samples: {len(train_dataset):,}")
logger.info(f"Test samples: {len(test_dataset):,}")
logger.info(f"Number of classes: {n_classes}")

# =========================
# DP-COMPATIBLE MODEL WRAPPER
# =========================
class DPCompatibleNet1D(nn.Module):
    """Wrapper around Net1D to make it DP-compatible by using simpler layers"""
    def __init__(self, base_model, n_classes):
        super().__init__()
        self.base_model = base_model
        self.n_classes = n_classes
    
    def forward(self, x):
        return self.base_model(x)

# =========================
# HELPER FUNCTIONS
# =========================
def load_dp_teacher_model():
    """Load DP-trained teacher model"""
    logger.info(f"Loading DP teacher from: {DP_TEACHER_PATH}")
    
    checkpoint = torch.load(DP_TEACHER_PATH, map_location=device, weights_only=False)
    
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")
    if 'config' in checkpoint:
        logger.info(f"Model config: {checkpoint['config']}")
    
    # Create teacher model
    teacher_model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=True,
        use_do=True,
        n_classes=n_classes
    ).to(device)
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    
    # Remove '_module.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_module.'):
            new_key = key[8:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    missing_keys, unexpected_keys = teacher_model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info("Successfully loaded DP teacher model")
    
    teacher_model.eval()
    
    # Freeze all BatchNorm layers
    for module in teacher_model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    # Freeze all parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    teacher_size_mb = (teacher_params * 4) / (1024 * 1024)
    
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Teacher size: {teacher_size_mb:.2f} MB")
    
    if 'epsilon' in checkpoint:
        logger.info(f"Teacher DP Training:")
        logger.info(f"  - Epsilon: {checkpoint['epsilon']:.2f}")
        logger.info(f"  - Delta: {checkpoint.get('delta', 'N/A')}")
    
    return teacher_model, teacher_params, teacher_size_mb, checkpoint

def create_student_model(student_size, student_config):
    """Create student model"""
    logger.info(f"Creating student: {student_size.upper()} - {student_config['description']}")
    
    student_model = Net1D(
        in_channels=12,
        base_filters=student_config['base_filters'],
        ratio=1,
        filter_list=student_config['filter_list'],
        m_blocks_list=student_config['m_blocks_list'],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    ).to(device)
    
    # Validate model
    logger.info("Validating model for Opacus compatibility...")
    errors = ModuleValidator.validate(student_model, strict=False)
    if errors:
        logger.warning(f"Validation errors found, attempting fix...")
        student_model = ModuleValidator.fix(student_model)
        logger.info("Model fixed for Opacus")
    else:
        logger.info("Model passes Opacus validation")
    
    # CRITICAL: Freeze GroupNorm parameters (created by ModuleValidator.fix)
    # Opacus cannot compute per-sample gradients for normalization layers
    logger.info("Freezing normalization layers for DP compatibility...")
    norm_params_frozen = 0
    for name, module in student_model.named_modules():
        if isinstance(module, (nn.GroupNorm, nn.InstanceNorm1d)):
            for param in module.parameters():
                param.requires_grad = False
                norm_params_frozen += 1
    
    logger.info(f"Froze {norm_params_frozen} normalization layer parameters")
    
    student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    student_size_mb = (student_params * 4) / (1024 * 1024)
    logger.info(f"Trainable student parameters: {student_params:,}")
    logger.info(f"Student model size: {student_size_mb:.2f} MB")
    
    return student_model, student_params, student_size_mb

def evaluate(model, loader, phase):
    """Evaluate model"""
    model.eval()
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {phase}", leave=False):
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x, y = batch['x'].to(device), batch['y'].to(device)
            
            if torch.isnan(x).any():
                continue
            
            logits = model(x)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue
            
            logits = torch.clamp(logits, min=-50, max=50)
            pred = torch.sigmoid(logits)
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                continue
            
            all_gt.append(y.cpu())
            all_pred.append(pred.cpu())
    
    if len(all_gt) == 0:
        logger.error("No valid batches found during evaluation!")
        return {
            'macro_auc': 0.0,
            'micro_auc': 0.0,
            'macro_prec': 0.0,
            'micro_prec': 0.0
        }
    
    all_gt = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)
    all_pred = torch.nan_to_num(all_pred, nan=0.5, posinf=1.0, neginf=0.0)
    
    macro_prec, micro_prec, macro_auc, micro_auc, challenge = cal_multilabel_metrics(
        all_gt, all_pred, np.array(labels), 0.5
    )
    
    return {
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'macro_prec': macro_prec,
        'micro_prec': micro_prec
    }

def save_results_to_csv(results_list):
    """Save results"""
    df = pd.DataFrame(results_list)
    
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to: {results_csv}")

# =========================
# MAIN DISTILLATION FUNCTION
# =========================
def train_dp_student(student_size, student_config, teacher_model, teacher_metrics):
    """Train student with DP using knowledge distillation"""
    
    logger.info("\n" + "-" * 80)
    logger.info(f"Training DP Student: {student_size.upper()}")
    logger.info("-" * 80)
    
    student_model, student_params, student_size_mb = create_student_model(student_size, student_config)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    teacher_size_mb = (teacher_params * 4) / (1024 * 1024)
    compression_ratio = teacher_params / student_params
    size_reduction_mb = teacher_size_mb - student_size_mb
    size_reduction_pct = (size_reduction_mb / teacher_size_mb) * 100
    
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Size reduction: {size_reduction_mb:.2f} MB ({size_reduction_pct:.1f}%)")
    
    # Loss functions
    criterion_ce = nn.BCEWithLogitsLoss(reduction='mean')
    
    # Optimizer - Adam with DP-compatible settings
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    logger.info(f"Using optimizer: Adam (lr={lr})")
    
    logger.info(f"Initializing Privacy Engine with target ε={TARGET_EPSILON}, δ={TARGET_DELTA}")
    privacy_engine = PrivacyEngine()
    
    try:
        student_model, optimizer, trainloader_private = privacy_engine.make_private_with_epsilon(
            module=student_model,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=Epochs,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        logger.info("✓ Privacy Engine initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Privacy Engine: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info(f"DP Training configured:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Target ε: {TARGET_EPSILON}")
    logger.info(f"  - Target δ: {TARGET_DELTA}")
    logger.info(f"  - Max grad norm: {MAX_GRAD_NORM}")
    logger.info(f"  - Optimizer: Adam")
    logger.info(f"  - Learning rate: {lr}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Alpha (KD weight): {alpha}")
    
    # Test DP
    logger.info("Testing DP forward/backward pass...")
    test_passed = False
    try:
        test_batch_iter = iter(trainloader_private)
        test_batch = next(test_batch_iter)
        
        if isinstance(test_batch, (list, tuple)):
            test_x, test_y = test_batch[0].to(device), test_batch[1].to(device)
        else:
            test_x = test_batch['x'].to(device)
            test_y = test_batch['y'].to(device)
        
        logger.info(f"Test batch shape - x: {test_x.shape}, y: {test_y.shape}")
        
        optimizer.zero_grad()
        test_out = student_model(test_x)
        
        if torch.isnan(test_out).any():
            logger.error("Model output contains NaN")
            raise ValueError("NaN in model output")
        
        test_loss = criterion_ce(test_out, test_y.float())
        
        if torch.isnan(test_loss):
            logger.error(f"Loss is NaN: {test_loss}")
            raise ValueError("Loss is NaN")
        
        logger.info(f"Loss: {test_loss.item():.4f}")
        
        test_loss.backward()
        
        # Check gradients - only for trainable parameters
        trainable_params = [p for p in student_model.parameters() if p.requires_grad]
        grad_count = sum(1 for p in trainable_params if p.grad is not None)
        logger.info(f"Gradients computed for {grad_count}/{len(trainable_params)} trainable parameters")
        
        # Check for per-sample gradients
        missing_grad_samples = []
        for name, param in student_model.named_parameters():
            if param.requires_grad and not hasattr(param, 'grad_sample'):
                missing_grad_samples.append(name)
        
        if missing_grad_samples:
            logger.error(f"Found {len(missing_grad_samples)} parameters missing grad_sample (causing the error):")
            for name in missing_grad_samples[:10]:
                logger.error(f"  - {name}")
            if len(missing_grad_samples) > 10:
                logger.error(f"  ... and {len(missing_grad_samples) - 10} more")
            raise ValueError(f"Per-sample gradients not initialized for {len(missing_grad_samples)} parameters")
        
        optimizer.step()
        logger.info("✓ DP forward/backward/step successful")
        test_passed = True
        
    except Exception as e:
        logger.error(f"✗ DP test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if not test_passed:
            raise
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, mode='max', verbose=False
    )
    
    # Training loop
    best_test_auc = 0.0
    best_epoch = 0
    best_epsilon = 0.0
    best_student_metrics = None
    training_start_time = datetime.now()
    
    for epoch in range(Epochs):
        student_model.train()
        train_loss = 0.0
        train_kd_loss = 0.0
        train_ce_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, batch in enumerate(tqdm(trainloader_private, desc=f"Epoch {epoch + 1}/{Epochs}", leave=False)):
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x = batch['x'].to(device)
                    y = batch['y'].to(device)
                
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                
                optimizer.zero_grad()
                
                # Teacher inference
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                    teacher_logits = torch.clamp(teacher_logits, min=-50, max=50)
                    # Use soft targets from teacher
                    teacher_soft_targets = torch.sigmoid(teacher_logits / temperature)
                    teacher_soft_targets = teacher_soft_targets.detach()
                
                # Student inference
                student_logits = student_model(x)
                
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    continue
                
                student_logits = torch.clamp(student_logits, min=-50, max=50)
                
                # Hard targets (ground truth)
                hard_targets = y.float()
                
                # KD Loss: Using soft teacher targets with temperature
                # Scale student logits by temperature for softer probability distribution
                student_soft_logits = student_logits / temperature
                
                # BCE loss between soft targets and soft student predictions
                kd_loss_val = criterion_ce(student_soft_logits, teacher_soft_targets)
                
                # CE Loss: Student vs hard labels
                ce_loss_val = criterion_ce(student_logits, hard_targets)
                
                # Combined loss: balance between KD and CE
                # Use lower alpha since CE already provides signal
                loss = alpha * kd_loss_val + (1 - alpha) * ce_loss_val
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_kd_loss += kd_loss_val.item()
                train_ce_loss += ce_loss_val.item()
                num_batches += 1
        
        except Exception as e:
            logger.error(f"Error during epoch {epoch + 1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            break
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        avg_kd_loss = train_kd_loss / num_batches if num_batches > 0 else 0.0
        avg_ce_loss = train_ce_loss / num_batches if num_batches > 0 else 0.0
        
        current_epsilon = privacy_engine.get_epsilon(TARGET_DELTA)
        
        # Evaluate
        student_metrics = evaluate(student_model, testloader, "Student")
        
        logger.info(
            f"Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} (KD: {avg_kd_loss:.4f}, CE: {avg_ce_loss:.4f}) | "
            f"ε: {current_epsilon:.2f} | "
            f"Macro AUROC: {student_metrics['macro_auc']:.4f}"
        )
        
        # Debug: Log if loss or metrics look suspicious
        if avg_train_loss > 10:
            logger.warning(f"⚠ High training loss detected: {avg_train_loss:.4f}")
        if student_metrics['macro_auc'] < 0.5:
            logger.warning(f"⚠ Low AUROC: {student_metrics['macro_auc']:.4f} (random baseline is 0.5)")
        
        if student_metrics['macro_auc'] > best_test_auc:
            best_test_auc = student_metrics['macro_auc']
            best_epoch = epoch + 1
            best_epsilon = current_epsilon
            best_student_metrics = student_metrics.copy()
        
        scheduler.step(student_metrics['macro_auc'])
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < early_stop_lr:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if current_epsilon > TARGET_EPSILON * 1.1:
            logger.warning(f"Privacy budget exceeded! ε: {current_epsilon:.2f}")
            break
    
    training_duration = (datetime.now() - training_start_time).total_seconds()
    final_epsilon = privacy_engine.get_epsilon(TARGET_DELTA)
    
    if best_student_metrics is None:
        best_student_metrics = student_metrics
    
    # Results
    result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'student_size': student_size,
        'student_params': student_params,
        'student_size_mb': student_size_mb,
        'teacher_params': teacher_params,
        'teacher_size_mb': teacher_size_mb,
        'compression_ratio': compression_ratio,
        'param_reduction_pct': (1 - student_params/teacher_params) * 100,
        'size_reduction_mb': size_reduction_mb,
        'size_reduction_pct': size_reduction_pct,
        'teacher_macro_auroc': teacher_metrics['macro_auc'],
        'teacher_micro_auroc': teacher_metrics['micro_auc'],
        'teacher_macro_ap': teacher_metrics['macro_prec'],
        'teacher_micro_ap': teacher_metrics['micro_prec'],
        'best_epoch': best_epoch,
        'best_macro_auroc': best_test_auc,
        'best_micro_auroc': best_student_metrics['micro_auc'],
        'best_macro_ap': best_student_metrics['macro_prec'],
        'best_micro_ap': best_student_metrics['micro_prec'],
        'performance_gap': teacher_metrics['macro_auc'] - best_test_auc,
        'performance_retention_pct': (best_test_auc / teacher_metrics['macro_auc']) * 100,
        'target_epsilon': TARGET_EPSILON,
        'target_delta': TARGET_DELTA,
        'final_epsilon': final_epsilon,
        'best_epsilon': best_epsilon,
        'max_grad_norm': MAX_GRAD_NORM,
        'training_duration_sec': training_duration,
        'temperature': temperature,
        'alpha': alpha,
        'learning_rate': lr,
        'batch_size': batch_size,
        'max_epochs': Epochs,
    }
    
    # Save checkpoint
    checkpoint_filename = f'student_dp_{student_size}_eps{final_epsilon:.1f}_auroc{best_test_auc:.4f}.pth'
    checkpoint_path = os.path.join(saved_dir, checkpoint_filename)
    
    torch.save({
        'experiment_config': result,
        'state_dict': student_model.state_dict(),
        'teacher_metrics': teacher_metrics,
        'student_metrics': best_student_metrics,
        'epsilon': final_epsilon,
        'delta': TARGET_DELTA,
        'best_epsilon': best_epsilon,
    }, checkpoint_path)
    
    logger.info(f"✓ Best AUROC: {best_test_auc:.4f} at epoch {best_epoch} (ε={best_epsilon:.2f})")
    logger.info(f"✓ Final ε: {final_epsilon:.2f}")
    logger.info(f"✓ Checkpoint saved: {checkpoint_filename}")
    
    del student_model, optimizer, privacy_engine
    torch.cuda.empty_cache()
    
    return result

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("STARTING DP KNOWLEDGE DISTILLATION EXPERIMENTS")
    logger.info(f"Total experiments: {len(STUDENT_CONFIGS)} student models")
    logger.info("=" * 80)
    
    teacher_model, teacher_params, teacher_size_mb, teacher_checkpoint = load_dp_teacher_model()
    
    logger.info("\nEvaluating DP Teacher Model...")
    teacher_metrics = evaluate(teacher_model, testloader, "Teacher")
    logger.info(
        f"Teacher Baseline | "
        f"Macro AUROC: {teacher_metrics['macro_auc']:.4f} | "
        f"Micro AUROC: {teacher_metrics['micro_auc']:.4f}"
    )
    
    all_results = []
    for student_size, student_config in STUDENT_CONFIGS.items():
        result = train_dp_student(student_size, student_config, teacher_model, teacher_metrics)
        all_results.append(result)
        save_results_to_csv([result])
    
    del teacher_model
    torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    
    results_df = pd.read_csv(results_csv)
    
    logger.info("\nRESULTS SUMMARY:")
    for _, row in results_df.iterrows():
        logger.info(
            f"\n{row['student_size'].upper()}: "
            f"AUROC {row['best_macro_auroc']:.4f} | "
            f"ε={row['final_epsilon']:.2f} | "
            f"{row['compression_ratio']:.1f}x compression"
        )
    
    best_row = results_df.loc[results_df['best_macro_auroc'].idxmax()]
    logger.info(
        f"\n{'='*80}\n"
        f"BEST STUDENT: {best_row['student_size'].upper()}\n"
        f"  AUROC: {best_row['best_macro_auroc']:.4f}\n"
        f"  Epsilon: {best_row['final_epsilon']:.2f}\n"
        f"  Compression: {best_row['compression_ratio']:.1f}x\n"
        f"  Performance Retention: {best_row['performance_retention_pct']:.1f}%\n"
        f"{'='*80}"
    )
    
    logger.info(f"\nDetailed results saved to: {results_csv}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)