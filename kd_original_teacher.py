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

from metrics import cal_multilabel_metrics
from net1d import Net1D
from dataset import LVEF_12lead_cls_Dataset

# =========================
# CONFIG
# =========================
num_lead = 12
gpu_id = 1
batch_size = 125
weight_decay = 1e-5
early_stop_lr = 1e-5
Epochs = 50


temperature = 8.0    # Instead of 4.0 (softer targets)
alpha = 0.7          # Instead of 0.5 (trust teacher more)
lr = 5e-4 # Instead of 1e-4 (faster learning)

# ECG Task Labels Mapping: teacher class index (0-149) -> student class label
# Maps specific teacher classes to your 17 student classes
ecg_task_labels = {
    3: 'sinus rhythm',
    4: 'sinus bradycardia',
    14: 't wave abnormal',
    34: 'sinus arrhythmia',
    18: 'incomplete right bundle branch block',
    6: 'sinus tachycardia',
    11: 'right bundle branch block',
    5: 'atrial fibrillation',
    23: 't wave inversion',
    37: 'right axis deviation',
    36: 'left anterior fascicular block',
    8: 'left axis deviation',
    32: 'atrial flutter',
    20: 'left bundle branch block',
    79: '1st degree av block',
    15: 'low qrs voltages',
    16: 'premature atrial contraction'
}

# Extract teacher class indices in sorted order (these are the indices to extract from 150 classes)
TEACHER_CLASS_INDICES = sorted(ecg_task_labels.keys())  # [3, 4, 5, 6, 8, 11, 14, 15, 16, 18, 20, 23, 32, 34, 36, 37, 79]

# Paths
train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_chapman_ptb.csv'
test_csv_path  = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
saved_dir = './res/distill_original_ecgfounder/'
results_csv = './res/distill_original_ecgfounder/distillation_results.csv'

# Teacher configurations to try
TEACHER_CONFIGS = [
    {
        'name': 'original_ecgfounder',
        'path': '/home/zoorab/projects/ECGFounder/12_lead_ECGFounder.pth',
        'description': 'Original ECGFounder',
        'type': 'original',  # NEW: distinguish original from finetuned
        'teacher_n_classes': 150  # NEW: original model has 150 labels
    }
]

# Student configurations
STUDENT_CONFIGS = {
    'large': {
        'base_filters': 64,
        'filter_list': [64, 128, 128, 256, 256, 512, 512],
        'm_blocks_list': [2, 2, 2, 3, 3, 4, 4],
        'description': '~2x compression, closest to teacher'
    },
    'medium': {
        'base_filters': 48,
        'filter_list': [48, 96, 96, 192, 192, 384, 384],
        'm_blocks_list': [1, 2, 2, 2, 2, 3, 3],
        'description': '~3x compression, higher accuracy'
    },
    'small': {
        'base_filters': 32,
        'filter_list': [32, 64, 64, 128, 128, 256, 256],
        'm_blocks_list': [1, 1, 1, 2, 2, 2, 2],
        'description': '~5x compression, balanced performance'
    },
    'tiny': {
        'base_filters': 16,
        'filter_list': [16, 32, 32, 64, 64, 128, 128],
        'm_blocks_list': [1, 1, 1, 1, 1, 1, 1],
        'description': '~10x compression, fastest inference'
    },

}

os.makedirs(saved_dir, exist_ok=True)
os.makedirs('logging', exist_ok=True)

# =========================
# LOGGING
# =========================
log_file = f'logging/distill_original_ecgfounder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log the teacher to student class mapping
logger.info(f"Teacher class indices mapping (150 -> 17):")
for student_idx, teacher_idx in enumerate(TEACHER_CLASS_INDICES):
    logger.info(f"  Student class {student_idx} <- Teacher class {teacher_idx}: {ecg_task_labels[teacher_idx]}")

# =========================
# DEVICE
# =========================
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# =========================
# DATA (Load once, use for all experiments)
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
logger.info(f"Number of student classes: {n_classes}")
logger.info(f"Expected student classes from labels: {n_classes} (must match len(TEACHER_CLASS_INDICES)={len(TEACHER_CLASS_INDICES)})")

if n_classes != len(TEACHER_CLASS_INDICES):
    logger.warning(f"⚠️  MISMATCH: Student has {n_classes} classes but mapping has {len(TEACHER_CLASS_INDICES)} classes!")
else:
    logger.info(f"✓ Class count matches: {n_classes} classes")

# =========================
# HELPER FUNCTIONS
# =========================
def load_teacher_model_original(teacher_config):
    """
    Load original ECGFounder teacher model (150 classes) using Net1D.
    This is the foundation model before any finetuning.
    """
    logger.info(f"Loading teacher: {teacher_config['description']}")
    logger.info(f"Teacher architecture: Net1D with 150 output classes")
    logger.info(f"Teacher has {teacher_config['teacher_n_classes']} classes (different from student's {n_classes})")
    
    # Create teacher model with original architecture and 150 classes
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
        use_bn=False,
        use_do=False,
        n_classes=teacher_config['teacher_n_classes']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(teacher_config['path'], map_location=device)
    state_dict = checkpoint['state_dict']
    log = teacher_model.load_state_dict(state_dict, strict=False)
    
    # Log loading details
    logger.info(f"State dict loading log: {log}")
    
    # Freeze all parameters
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False 
    
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Note: Using Net1D directly (no finetuning wrapper)")
    
    return teacher_model, teacher_params, checkpoint

def load_teacher_model_finetuned(teacher_config):
    """
    Load finetuned teacher model using Net1D.
    For future use if you want to support finetuned models as teacher.
    """
    logger.info(f"Loading teacher: {teacher_config['description']}")
    logger.info(f"Teacher architecture: Net1D with {n_classes} output classes")
    
    # Create teacher model with matching number of classes
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
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(teacher_config['path'], map_location=device)
    state_dict = checkpoint['state_dict']
    log = teacher_model.load_state_dict(state_dict, strict=False)
    
    logger.info(f"State dict loading log: {log}")
    
    # Freeze parameters
    for param in teacher_model.named_parameters():
        param.requires_grad = False
    
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Teacher parameters: {teacher_params:,}")
    
    return teacher_model, teacher_params, checkpoint

def load_teacher_model(teacher_config):
    """Wrapper to load appropriate teacher model based on type"""
    if teacher_config.get('type') == 'original':
        return load_teacher_model_original(teacher_config)
    else:
        return load_teacher_model_finetuned(teacher_config)

def create_student_model(student_size, student_config):
    """Create student model based on configuration"""
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
    
    student_params = sum(p.numel() for p in student_model.parameters())
    # Calculate model size in MB (assuming float32 = 4 bytes per parameter)
    student_size_mb = (student_params * 4) / (1024 * 1024)
    logger.info(f"Student parameters: {student_params:,}")
    logger.info(f"Student model size: {student_size_mb:.2f} MB")
    logger.info(f"Student output classes: {n_classes}")
    
    return student_model, student_params, student_size_mb

def get_teacher_features(model, x):
    """
    Extract features from teacher model before classification head.
    Works for both original and finetuned models.
    """
    # This extracts intermediate features before the final classification layer
    # Modify based on your ft_12lead_ECGFounder architecture
    with torch.no_grad():
        features = model.encoder(x) if hasattr(model, 'encoder') else model.features(x)
    return features

def get_student_features(model, x):
    """Extract features from student model before classification head"""
    # Modify based on your Net1D architecture
    features = model.features(x) if hasattr(model, 'features') else x
    return features

def evaluate(model, loader, phase, n_eval_classes=None):
    """
    Evaluate model on given dataset.
    n_eval_classes: if specified, only evaluate on first n_eval_classes outputs
                    (useful when teacher has different number of classes)
    """
    if n_eval_classes is None:
        n_eval_classes = n_classes
    
    model.eval()
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating {phase}", leave=False):
            x, y = x.to(device), y.to(device)
            
            if torch.isnan(x).any():
                continue
            
            logits = model(x)
            
            # For teacher with different number of classes, take only matching classes
            if logits.shape[1] > n_eval_classes:
                logits = logits[:, :n_eval_classes]
            
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
    """Save or append results to CSV"""
    df = pd.DataFrame(results_list)
    
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to: {results_csv}")

# =========================
# MAIN EXPERIMENT LOOP
# =========================
logger.info("=" * 80)
logger.info("STARTING DISTILLATION EXPERIMENTS")
logger.info(f"Total experiments: {len(TEACHER_CONFIGS)} teachers × {len(STUDENT_CONFIGS)} students = {len(TEACHER_CONFIGS) * len(STUDENT_CONFIGS)}")
logger.info("=" * 80)

all_results = []
experiment_num = 0

for teacher_config in TEACHER_CONFIGS:
    # Load teacher model
    logger.info("\n" + "=" * 80)
    logger.info(f"TEACHER: {teacher_config['description']}")
    logger.info("=" * 80)
    
    teacher_model, teacher_params, teacher_checkpoint = load_teacher_model(teacher_config)
    
    # Calculate teacher size in MB
    teacher_size_mb = (teacher_params * 4) / (1024 * 1024)
    
    # For original ECGFounder, skip teacher evaluation (different label space)
    if teacher_config.get('type') == 'original':
        logger.info("=" * 80)
        logger.info("TEACHER CLASS MAPPING (150 -> 17)")
        logger.info("=" * 80)
        for student_idx, teacher_idx in enumerate(TEACHER_CLASS_INDICES):
            logger.info(f"  Student class {student_idx:2d} <- Teacher class {teacher_idx:3d}: {ecg_task_labels[teacher_idx]}")
        logger.info("=" * 80)
        logger.info("Skipping teacher baseline evaluation (original model has different label space)")
        logger.info(f"Teacher model size: {teacher_size_mb:.2f} MB")
        
        # Use placeholder metrics for teacher
        teacher_metrics = {
            'macro_auc': -1.0,
            'micro_auc': -1.0,
            'macro_prec': -1.0,
            'micro_prec': -1.0
        }
    else:
        # For finetuned models, evaluate teacher baseline
        teacher_metrics = evaluate(teacher_model, testloader, "Teacher")
        logger.info(
            f"Teacher Baseline | "
            f"Macro AUROC: {teacher_metrics['macro_auc']:.4f} | "
            f"Micro AUROC: {teacher_metrics['micro_auc']:.4f} | "
            f"Size: {teacher_size_mb:.2f} MB"
        )
    
    for student_size, student_config in STUDENT_CONFIGS.items():
        experiment_num += 1
        logger.info("\n" + "-" * 80)
        logger.info(f"EXPERIMENT {experiment_num}/{len(TEACHER_CONFIGS) * len(STUDENT_CONFIGS)}")
        logger.info(f"Teacher: {teacher_config['name']} | Student: {student_size}")
        logger.info("-" * 80)
        
        # Create student model
        student_model, student_params, student_size_mb = create_student_model(student_size, student_config)
        compression_ratio = teacher_params / student_params
        size_reduction_mb = teacher_size_mb - student_size_mb
        size_reduction_pct = (size_reduction_mb / teacher_size_mb) * 100
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Size reduction: {size_reduction_mb:.2f} MB ({size_reduction_pct:.1f}%)")
        
        # Setup training
        criterion_ce = nn.BCEWithLogitsLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max')
        
        # Training loop
        best_test_auc = 0.0
        best_epoch = 0
        global_step = 0
        training_start_time = datetime.now()
        
        for epoch in range(Epochs):
            # Train
            student_model.train()
            train_loss = 0.0
            train_kd_loss = 0.0
            train_ce_loss = 0.0
            num_batches = 0
            
            for x, y in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{Epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                
                # Get teacher logits (without gradient)
                with torch.no_grad():
                    teacher_logits_full = teacher_model(x)  # [B, 150]
                    
                    # Extract only the relevant classes based on ecg_task_labels
                    # TEACHER_CLASS_INDICES = [3, 4, 5, 6, 8, 11, 14, 15, 16, 18, 20, 23, 32, 34, 36, 37, 79]
                    teacher_logits_indices = torch.tensor(
                        TEACHER_CLASS_INDICES, 
                        dtype=torch.long, 
                        device=device
                    )
                    teacher_logits = teacher_logits_full.index_select(1, teacher_logits_indices)  # [B, 17]
                    
                    teacher_logits = torch.clamp(teacher_logits, min=-50, max=50)
                
                # Get student logits
                student_logits = student_model(x)
                
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    continue
                
                student_logits = torch.clamp(student_logits, min=-50, max=50)
                
                # Knowledge Distillation Loss (adapted for matching logit dimensions)
                try:
                    kd_loss_val = criterion_kl(
                        F.log_softmax(student_logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1)
                    ) * (temperature * temperature)
                except Exception as e:
                    logger.warning(f"KD loss calculation error: {e}")
                    continue
                
                # Cross-entropy loss with ground truth
                ce_loss_val = criterion_ce(student_logits, y)
                
                # Combined loss
                loss = alpha * kd_loss_val + (1 - alpha) * ce_loss_val
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_kd_loss += kd_loss_val.item()
                train_ce_loss += ce_loss_val.item()
                num_batches += 1
                global_step += 1
            
            if num_batches == 0:
                logger.warning(f"Epoch {epoch + 1}: No valid batches!")
                continue
            
            avg_train_loss = train_loss / num_batches
            avg_kd_loss = train_kd_loss / num_batches
            avg_ce_loss = train_ce_loss / num_batches
            
            # Evaluate on test set
            student_metrics = evaluate(student_model, testloader, "Student")
            
            logger.info(
                f"Epoch {epoch + 1} | "
                f"Total Loss: {avg_train_loss:.4f} | "
                f"KD Loss: {avg_kd_loss:.4f} | "
                f"CE Loss: {avg_ce_loss:.4f} | "
                f"Macro AUROC: {student_metrics['macro_auc']:.4f} | "
                f"Micro AUROC: {student_metrics['micro_auc']:.4f}"
            )
            
            # Track best
            if student_metrics['macro_auc'] > best_test_auc:
                best_test_auc = student_metrics['macro_auc']
                best_epoch = epoch + 1
                best_student_metrics = student_metrics.copy()
            
            scheduler.step(student_metrics['macro_auc'])
            
            if optimizer.param_groups[0]['lr'] < early_stop_lr:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        training_duration = (datetime.now() - training_start_time).total_seconds()
        
        # Save experiment results
        result = {
            'experiment_num': experiment_num,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'teacher_type': teacher_config['name'],
            'teacher_path': teacher_config['path'],
            'teacher_n_classes': teacher_config.get('teacher_n_classes', n_classes),
            'teacher_macro_auroc': teacher_metrics['macro_auc'],
            'teacher_micro_auroc': teacher_metrics['micro_auc'],
            'teacher_macro_ap': teacher_metrics['macro_prec'],
            'teacher_micro_ap': teacher_metrics['micro_prec'],
            'teacher_params': teacher_params,
            'teacher_size_mb': teacher_size_mb,
            'student_size': student_size,
            'student_n_classes': n_classes,
            'student_params': student_params,
            'student_size_mb': student_size_mb,
            'compression_ratio': compression_ratio,
            'param_reduction_pct': (1 - student_params/teacher_params) * 100,
            'size_reduction_mb': size_reduction_mb,
            'size_reduction_pct': size_reduction_pct,
            'best_epoch': best_epoch,
            'best_macro_auroc': best_test_auc,
            'best_micro_auroc': best_student_metrics['micro_auc'],
            'best_macro_ap': best_student_metrics['macro_prec'],
            'best_micro_ap': best_student_metrics['micro_prec'],
            'performance_gap': teacher_metrics['macro_auc'] - best_test_auc if teacher_metrics['macro_auc'] > 0 else 'N/A',
            'performance_retention_pct': (best_test_auc / teacher_metrics['macro_auc']) * 100 if teacher_metrics['macro_auc'] > 0 else 'N/A',
            'training_duration_sec': training_duration,
            'temperature': temperature,
            'alpha': alpha,
            'learning_rate': lr,
            'batch_size': batch_size,
            'max_epochs': Epochs,
            'teacher_type_code': teacher_config.get('type', 'finetuned'),
            'notes': f'Original teacher: selected 17 classes from 150 using indices {TEACHER_CLASS_INDICES}'
        }
        
        all_results.append(result)
        
        # Save checkpoint with descriptive name
        checkpoint_filename = f'student_{teacher_config["name"]}_{student_size}_auroc{best_test_auc:.4f}.pth'
        checkpoint_path = os.path.join(saved_dir, checkpoint_filename)
        
        torch.save({
            'experiment_config': result,
            'state_dict': student_model.state_dict(),
            'teacher_config': teacher_config,
            'student_config': student_config,
            'teacher_metrics': teacher_metrics,
            'student_metrics': best_student_metrics,
            'n_classes': n_classes,
            'teacher_n_classes': teacher_config.get('teacher_n_classes', n_classes),
        }, checkpoint_path)
        
        logger.info(f"✓ Best AUROC: {best_test_auc:.4f} at epoch {best_epoch}")
        if isinstance(result['performance_gap'], str):
            logger.info(f"✓ Note: Teacher evaluation skipped (original model)")
        else:
            logger.info(f"✓ Performance gap: {result['performance_gap']:.4f}")
        logger.info(f"✓ Checkpoint saved: {checkpoint_filename}")
        
        # Save results after each experiment (incremental save)
        save_results_to_csv([result])
        
        # Clean up for next experiment
        del student_model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Clean up teacher model
    del teacher_model
    torch.cuda.empty_cache()

# =========================
# FINAL SUMMARY
# =========================
logger.info("\n" + "=" * 80)
logger.info("ALL EXPERIMENTS COMPLETED")
logger.info("=" * 80)

results_df = pd.read_csv(results_csv)

logger.info("\nBEST RESULTS BY TEACHER:")
for teacher_name in results_df['teacher_type'].unique():
    teacher_results = results_df[results_df['teacher_type'] == teacher_name]
    best_row = teacher_results.loc[teacher_results['best_macro_auroc'].idxmax()]
    
    teacher_type_info = f" ({best_row.get('teacher_type_code', 'unknown')} model)" if 'teacher_type_code' in best_row else ""
    
    logger.info(
        f"\n{teacher_name}{teacher_type_info}: Best is {best_row['student_size']} student with "
        f"AUROC {best_row['best_macro_auroc']:.4f} "
        f"({best_row['compression_ratio']:.1f}x compression)"
    )

logger.info(f"\nDetailed results saved to: {results_csv}")
logger.info("=" * 80)