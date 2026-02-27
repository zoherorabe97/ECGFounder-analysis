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
from torch.utils.data import DataLoader

from metrics import cal_multilabel_metrics
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
patience = 15

# Paths
train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_chapman_ptb.csv'
test_csv_path  = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
saved_dir = './res/baseline_student/'
results_csv = './res/baseline_student/baseline_training_results.csv'

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
log_file = f'logging/baseline_student_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    total_loss = 0.0
    batch_count = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
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
            
            # Calculate loss
            loss = criterion(logits, y)
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                batch_count += 1
            
            all_gt.append(y.cpu())
            all_pred.append(pred.cpu())
    
    if len(all_gt) == 0:
        logger.error("No valid batches found during evaluation!")
        return {
            'macro_auc': 0.0,
            'micro_auc': 0.0,
            'macro_prec': 0.0,
            'micro_prec': 0.0,
            'loss': 0.0
        }
    
    all_gt = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)
    all_pred = torch.nan_to_num(all_pred, nan=0.5, posinf=1.0, neginf=0.0)
    
    macro_prec, micro_prec, macro_auc, micro_auc, challenge = cal_multilabel_metrics(
        all_gt, all_pred, np.array(labels), 0.5
    )
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    
    return {
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'macro_prec': macro_prec,
        'micro_prec': micro_prec,
        'loss': avg_loss
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
# MAIN TRAINING LOOP
# =========================
logger.info("=" * 80)
logger.info("STARTING BASELINE STUDENT TRAINING")
logger.info(f"Total experiments: {len(STUDENT_CONFIGS)}")
logger.info("=" * 80)

all_results = []
experiment_num = 0

for student_size, student_config in STUDENT_CONFIGS.items():
    experiment_num += 1
    
    logger.info("\n" + "=" * 80)
    logger.info(f"EXPERIMENT {experiment_num}/{len(STUDENT_CONFIGS)}")
    logger.info(f"Student: {student_size.upper()}")
    logger.info("=" * 80)
    
    # Create student model
    student_model, student_params, student_size_mb = create_student_model(student_size, student_config)
    
    logger.info(f"Model size: {student_size_mb:.2f} MB")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=patience, factor=0.1, verbose=False
    )
    
    # Training variables
    best_test_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    training_start_time = datetime.now()
    
    # Track metrics over epochs
    epoch_metrics = {
        'epoch': [],
        'train_loss': [],
        'test_macro_auc': [],
        'test_micro_auc': [],
        'test_macro_ap': [],
        'test_micro_ap': [],
        'test_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(Epochs):
        # Train phase
        student_model.train()
        train_loss = 0.0
        batch_count = 0
        
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{Epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            if torch.isnan(x).any() or torch.isnan(y).any():
                continue
            
            student_logits = student_model(x)
            
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                continue
            
            student_logits = torch.clamp(student_logits, min=-50, max=50)
            
            loss = criterion(student_logits, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            global_step += 1
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0.0
        
        # Evaluation phase
        student_metrics = evaluate(student_model, testloader, "Student")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch + 1:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {student_metrics['loss']:.4f} | "
            f"Macro AUROC: {student_metrics['macro_auc']:.4f} | "
            f"Micro AUROC: {student_metrics['micro_auc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Track metrics
        epoch_metrics['epoch'].append(epoch + 1)
        epoch_metrics['train_loss'].append(avg_train_loss)
        epoch_metrics['test_macro_auc'].append(student_metrics['macro_auc'])
        epoch_metrics['test_micro_auc'].append(student_metrics['micro_auc'])
        epoch_metrics['test_macro_ap'].append(student_metrics['macro_prec'])
        epoch_metrics['test_micro_ap'].append(student_metrics['micro_prec'])
        epoch_metrics['test_loss'].append(student_metrics['loss'])
        epoch_metrics['learning_rate'].append(current_lr)
        
        # Track best model
        if student_metrics['macro_auc'] > best_test_auc:
            best_test_auc = student_metrics['macro_auc']
            best_epoch = epoch + 1
            best_student_metrics = student_metrics.copy()
            epochs_without_improvement = 0
            
            logger.info(f"✓ New best AUROC: {best_test_auc:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Learning rate scheduler
        scheduler.step(student_metrics['macro_auc'])
        
        # Early stopping conditions
        if current_lr < early_stop_lr:
            logger.info("Early stopping triggered: learning rate too low")
            break
        
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered: no improvement for {patience} epochs")
            break
    
    training_duration = (datetime.now() - training_start_time).total_seconds()
    
    # Save experiment results
    result = {
        'experiment_num': experiment_num,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'student_size': student_size,
        'student_params': student_params,
        'student_size_mb': student_size_mb,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'best_macro_auroc': best_test_auc,
        'best_micro_auroc': best_student_metrics['micro_auc'],
        'best_macro_ap': best_student_metrics['macro_prec'],
        'best_micro_ap': best_student_metrics['micro_prec'],
        'best_test_loss': best_student_metrics['loss'],
        'training_duration_sec': training_duration,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'max_epochs': Epochs,
        'early_stop_lr': early_stop_lr,
        'scheduler_patience': patience,
    }
    
    all_results.append(result)
    
    # Save checkpoint with descriptive name
    checkpoint_filename = f'student_{student_size}_auroc{best_test_auc:.4f}_epoch{best_epoch}.pth'
    checkpoint_path = os.path.join(saved_dir, checkpoint_filename)
    
    torch.save({
        'experiment_config': result,
        'state_dict': student_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'student_metrics': best_student_metrics,
        'epoch_metrics': epoch_metrics,
    }, checkpoint_path)
    
    logger.info(f"✓ Best AUROC: {best_test_auc:.4f} at epoch {best_epoch}")
    logger.info(f"✓ Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    logger.info(f"✓ Checkpoint saved: {checkpoint_filename}")
    
    # Save results after each experiment (incremental save)
    save_results_to_csv([result])
    
    # Save epoch metrics to CSV for detailed analysis
    epoch_metrics_df = pd.DataFrame(epoch_metrics)
    epoch_metrics_csv = os.path.join(saved_dir, f'epoch_metrics_{student_size}.csv')
    epoch_metrics_df.to_csv(epoch_metrics_csv, index=False)
    logger.info(f"✓ Epoch metrics saved: epoch_metrics_{student_size}.csv")
    
    # Clean up for next experiment
    del student_model, optimizer, scheduler
    torch.cuda.empty_cache()

# =========================
# FINAL SUMMARY
# =========================
logger.info("\n" + "=" * 80)
logger.info("ALL EXPERIMENTS COMPLETED")
logger.info("=" * 80)

results_df = pd.read_csv(results_csv)

logger.info("\nFINAL RESULTS SUMMARY:")
logger.info("-" * 80)

for idx, row in results_df.iterrows():
    logger.info(
        f"{row['student_size']:8s} | "
        f"AUROC: {row['best_macro_auroc']:.4f} | "
        f"Epoch: {int(row['best_epoch']):2d}/{int(row['total_epochs']):2d} | "
        f"Time: {row['training_duration_sec']/60:6.2f} min | "
        f"Size: {row['student_size_mb']:6.2f} MB"
    )

logger.info("-" * 80)

best_row = results_df.loc[results_df['best_macro_auroc'].idxmax()]
logger.info(f"\nBEST OVERALL: {best_row['student_size']} with AUROC {best_row['best_macro_auroc']:.4f}")

logger.info(f"\nDetailed results saved to: {results_csv}")
logger.info("=" * 80)