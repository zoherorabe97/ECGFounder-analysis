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
from finetune_model import ft_12lead_ECGFounder
from net1d import Net1D
from dataset import LVEF_12lead_cls_Dataset

# =========================
# CONFIG
# =========================
num_lead = 12
gpu_id = 1
batch_size = 64
lr = 1e-4
weight_decay = 1e-5
early_stop_lr = 1e-5
Epochs = 50
temperature = 4.0
alpha = 0.5

# Paths
train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv'
test_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
pretrained_pth = '12_lead_ECGFounder.pth'
saved_dir = './res/distill/'
results_csv = './res/distill/distillation_results.csv'

# Teacher configurations to try
TEACHER_CONFIGS = [
    {
        'name': 'full_ft',
        'path': '/home/zoorab/projects/ECGFounder/res/eval/checkpoint_full_ft_step3327_auroc0.9263.pth',
        'description': 'Full Fine-tuned Teacher'
    },
    # {
    #     'name': 'linear_probe',
    #     'path': '/home/zoorab/projects/ECGFounder/res/eval/checkpoint_linear_probe_step4416_auroc0.9011.pth',
    #     'description': 'Linear Probe Teacher'
    # }
]

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
log_file = f'logging/distillation_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
logger.info(f"Number of classes: {n_classes}")

# =========================
# HELPER FUNCTIONS
# =========================
def load_teacher_model(teacher_config):
    """Load teacher model with fine-tuned weights"""
    logger.info(f"Loading teacher: {teacher_config['description']}")
    
    teacher_model = ft_12lead_ECGFounder(
        device=device,
        pth=pretrained_pth,
        n_classes=n_classes,
        linear_prob=False
    ).to(device)
    
    checkpoint = torch.load(teacher_config['path'], map_location=device)
    teacher_model.load_state_dict(checkpoint['state_dict'])
    teacher_model.eval()
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Teacher parameters: {teacher_params:,}")
    
    return teacher_model, teacher_params, checkpoint

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
    
    return student_model, student_params, student_size_mb

def evaluate(model, loader, phase):
    """Evaluate model on given dataset"""
    model.eval()
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating {phase}", leave=False):
            x, y = x.to(device), y.to(device)
            
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
    
    # Evaluate teacher baseline
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
            
            for x, y in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{Epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                    teacher_logits = torch.clamp(teacher_logits, min=-50, max=50)
                
                student_logits = student_model(x)
                
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    continue
                
                student_logits = torch.clamp(student_logits, min=-50, max=50)
                
                kd_loss_val = criterion_kl(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature * temperature)
                
                ce_loss_val = criterion_ce(student_logits, y)
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
                global_step += 1
            
            avg_train_loss = train_loss / len(trainloader)
            
            # Evaluate
            student_metrics = evaluate(student_model, testloader, "Student")
            
            logger.info(
                f"Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | "
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
                logger.info("Early stopping triggered")
                break
        
        training_duration = (datetime.now() - training_start_time).total_seconds()
        
        # Save experiment results
        result = {
            'experiment_num': experiment_num,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'teacher_type': teacher_config['name'],
            'teacher_path': teacher_config['path'],
            'teacher_macro_auroc': teacher_metrics['macro_auc'],
            'teacher_micro_auroc': teacher_metrics['micro_auc'],
            'teacher_macro_ap': teacher_metrics['macro_prec'],
            'teacher_micro_ap': teacher_metrics['micro_prec'],
            'teacher_params': teacher_params,
            'teacher_size_mb': teacher_size_mb,
            'student_size': student_size,
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
            'performance_gap': teacher_metrics['macro_auc'] - best_test_auc,
            'performance_retention_pct': (best_test_auc / teacher_metrics['macro_auc']) * 100,
            'training_duration_sec': training_duration,
            'temperature': temperature,
            'alpha': alpha,
            'learning_rate': lr,
            'batch_size': batch_size,
            'max_epochs': Epochs,
        }
        
        all_results.append(result)
        
        # Save checkpoint with descriptive name
        checkpoint_filename = f'student_{teacher_config["name"]}_{student_size}_auroc{best_test_auc:.4f}.pth'
        checkpoint_path = os.path.join(saved_dir, checkpoint_filename)
        
        torch.save({
            'experiment_config': result,
            'state_dict': student_model.state_dict(),
            'teacher_metrics': teacher_metrics,
            'student_metrics': best_student_metrics,
        }, checkpoint_path)
        
        logger.info(f"✓ Best AUROC: {best_test_auc:.4f} at epoch {best_epoch}")
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
    logger.info(
        f"\n{teacher_name}: Best is {best_row['student_size']} student with "
        f"AUROC {best_row['best_macro_auroc']:.4f} "
        f"({best_row['compression_ratio']:.1f}x compression)"
    )

logger.info(f"\nDetailed results saved to: {results_csv}")
logger.info("=" * 80)