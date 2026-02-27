# =========================
# Teacher Model Training from Scratch
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
batch_size = 64
lr = 1e-4
weight_decay = 1e-5
early_stop_lr = 1e-5
Epochs = 50
patience = 15

# Paths
train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_sph_chapman_ptb_v2.csv'
test_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
teacher_checkpoint_path = './res/teacher/teacher_best.pth'
teacher_latest_checkpoint_path = './res/teacher/teacher_latest.pth'
results_csv = './res/teacher/teacher_training_log.csv'

# Teacher configurations
TEACHER_CONFIG = {
    'in_channels': 12,
    'base_filters': 64,
    'ratio': 1,
    'filter_list': [64, 160, 160, 400, 400, 1024, 1024],
    'm_blocks_list': [2, 2, 2, 3, 3, 4, 4],
    'kernel_size': 16,
    'stride': 2,
    'groups_width': 16,
    'verbose': False,
    'use_bn': False,
    'use_do': False,
}

os.makedirs('./res/teacher', exist_ok=True)
os.makedirs('logging', exist_ok=True)

# =========================
# LOGGING
# =========================
log_file = f'logging/teacher_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
# DATA LOADING
# =========================
logger.info("Loading datasets...")
labels_df = pd.read_csv(train_csv_path)
labels = labels_df.columns[4:].tolist()
n_classes = len(labels)

train_dataset = LVEF_12lead_cls_Dataset(ecg_path='', labels_df=labels_df)
test_dataset = LVEF_12lead_cls_Dataset(ecg_path='', labels_df=pd.read_csv(test_csv_path))

trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle=False)

logger.info(f"Train samples: {len(train_dataset):,}")
logger.info(f"Test samples: {len(test_dataset):,}")
logger.info(f"Number of classes: {n_classes}")

# =========================
# HELPER FUNCTIONS
# =========================
def create_teacher_model(n_classes):
    """Create teacher model"""
    logger.info("Creating teacher model...")
    
    model = Net1D(
        in_channels=TEACHER_CONFIG['in_channels'],
        base_filters=TEACHER_CONFIG['base_filters'],
        ratio=TEACHER_CONFIG['ratio'],
        filter_list=TEACHER_CONFIG['filter_list'],
        m_blocks_list=TEACHER_CONFIG['m_blocks_list'],
        kernel_size=TEACHER_CONFIG['kernel_size'],
        stride=TEACHER_CONFIG['stride'],
        groups_width=TEACHER_CONFIG['groups_width'],
        verbose=TEACHER_CONFIG['verbose'],
        use_bn=TEACHER_CONFIG['use_bn'],
        use_do=TEACHER_CONFIG['use_do'],
        n_classes=n_classes
    ).to(device)
    
    teacher_params = sum(p.numel() for p in model.parameters())
    teacher_size_mb = (teacher_params * 4) / (1024 * 1024)
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Teacher model size: {teacher_size_mb:.2f} MB")
    
    return model, teacher_params, teacher_size_mb

def check_checkpoint_exists(checkpoint_path):
    """Check if training checkpoint exists"""
    return os.path.exists(checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and return starting epoch"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_auc = checkpoint['best_auc']
    patience_counter = checkpoint['patience_counter']
    
    logger.info(f"Resumed from epoch {start_epoch} with best AUROC: {best_auc:.4f}")
    
    return start_epoch, best_auc, patience_counter

def save_checkpoint(model, optimizer, scheduler, epoch, best_auc, patience_counter, checkpoint_path, is_best=False):
    """Save checkpoint"""
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_auc': best_auc,
        'patience_counter': patience_counter,
    }, checkpoint_path)
    
    if is_best:
        logger.info(f"✓ Best checkpoint saved: {checkpoint_path}")
    else:
        logger.info(f"✓ Latest checkpoint saved: {checkpoint_path}")

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

def save_training_log(epoch, train_loss, train_metrics, test_metrics, learning_rate):
    """Save training log to CSV"""
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_macro_auc': train_metrics.get('macro_auc', 0.0),
        'train_micro_auc': train_metrics.get('micro_auc', 0.0),
        'test_macro_auc': test_metrics['macro_auc'],
        'test_micro_auc': test_metrics['micro_auc'],
        'test_macro_ap': test_metrics['macro_prec'],
        'test_micro_ap': test_metrics['micro_prec'],
        'learning_rate': learning_rate,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(results_csv, index=False)

# =========================
# MAIN TRAINING
# =========================
def main():
    logger.info("=" * 80)
    logger.info("TEACHER MODEL TRAINING")
    logger.info("=" * 80)
    
    # Create model
    teacher_model, teacher_params, teacher_size_mb = create_teacher_model(n_classes)
    
    # Check if checkpoint exists
    checkpoint_exists = check_checkpoint_exists(teacher_latest_checkpoint_path)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=patience, 
        factor=0.1, 
        mode='max',
        verbose=True
    )
    
    # Initialize training variables
    start_epoch = 0
    best_auc = 0.0
    patience_counter = 0
    
    # Load checkpoint if exists
    if checkpoint_exists:
        logger.info(f"Checkpoint found! Resuming training...")
        start_epoch, best_auc, patience_counter = load_checkpoint(
            teacher_model, 
            optimizer, 
            scheduler,
            teacher_latest_checkpoint_path
        )
    else:
        logger.info("No checkpoint found. Starting training from scratch...")
    
    # Training loop
    training_start_time = datetime.now()
    
    try:
        for epoch in range(start_epoch, Epochs):
            # Train phase
            teacher_model.train()
            train_loss = 0.0
            
            for x, y in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{Epochs} [Train]", leave=False):
                x, y = x.to(device), y.to(device)
                
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                
                logits = teacher_model(x)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                
                logits = torch.clamp(logits, min=-50, max=50)
                loss = criterion(logits, y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(trainloader)
            
            # Evaluate phase
            test_metrics = evaluate(teacher_model, testloader, "Teacher")
            current_auc = test_metrics['macro_auc']
            
            logger.info(
                f"Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | "
                f"Test Macro AUROC: {current_auc:.4f} | "
                f"Test Micro AUROC: {test_metrics['micro_auc']:.4f}"
            )
            
            # Save training log
            save_training_log(epoch + 1, avg_train_loss, {}, test_metrics, optimizer.param_groups[0]['lr'])
            
            # Track best model
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                
                # Save best checkpoint
                save_checkpoint(
                    teacher_model, 
                    optimizer, 
                    scheduler, 
                    epoch, 
                    best_auc, 
                    patience_counter,
                    teacher_checkpoint_path,
                    is_best=True
                )
                logger.info(f"✓ New best AUROC: {best_auc:.4f}")
            else:
                patience_counter += 1
            
            # Save latest checkpoint
            save_checkpoint(
                teacher_model, 
                optimizer, 
                scheduler, 
                epoch, 
                best_auc, 
                patience_counter,
                teacher_latest_checkpoint_path,
                is_best=False
            )
            
            # Learning rate scheduling
            scheduler.step(current_auc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Early stopping
            if current_lr < early_stop_lr:
                logger.info(f"Early stopping triggered (lr: {current_lr:.2e} < {early_stop_lr:.2e})")
                break
            
            if patience_counter >= patience:
                logger.info(f"Patience exceeded ({patience_counter}/{patience}). Early stopping...")
                break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Latest checkpoint saved.")
    
    training_duration = (datetime.now() - training_start_time).total_seconds()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Training duration: {training_duration / 3600:.2f} hours")
    logger.info(f"Best AUROC: {best_auc:.4f}")
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Teacher size: {teacher_size_mb:.2f} MB")
    logger.info(f"Best checkpoint: {teacher_checkpoint_path}")
    logger.info(f"Training log: {results_csv}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()