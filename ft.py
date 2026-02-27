# =========================
# Imports
# =========================
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util import save_checkpoint
from metrics import cal_multilabel_metrics
from finetune_model import ft_12lead_ECGFounder
from dataset import LVEF_12lead_cls_Dataset

# =========================
# CONFIG
# =========================
LINEAR_PROBE = False          # True = Linear Probe | False = Full Fine-tuning
num_lead = 12
gpu_id = 0
batch_size = 64
lr = 1e-4                   # full FT LR
lp_lr = 1e-3                # linear probe LR
weight_decay = 1e-5
Epochs = 10
early_stop_lr = 1e-5

train_csv_path = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/clean_cpsc_chapman_ptb.csv'
test_csv_path  = '/home/zoorab/projects/12-lead-ecg-classifier/data/split_csvs/G12EC/clean_all_g12ec.csv'
ecg_path = ''
pth = '12_lead_ECGFounder.pth'
saved_dir = './res/eval/'

os.makedirs(saved_dir, exist_ok=True)
os.makedirs('logging', exist_ok=True)

# =========================
# LOGGING
# =========================
log_file = 'logging/g12ec_ft.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =========================
# DEVICE
# =========================
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# =========================
# DATA
# =========================
train_df = pd.read_csv(train_csv_path)
test_df  = pd.read_csv(test_csv_path)

labels = train_df.columns[4:].tolist()
n_classes = len(labels)

train_dataset = LVEF_12lead_cls_Dataset(ecg_path, train_df)
test_dataset  = LVEF_12lead_cls_Dataset(ecg_path, test_df)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
testloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=24)

logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")
logger.info(f"Classes: {n_classes}")

# =========================
# MODEL
# =========================
model = ft_12lead_ECGFounder(
    device=device,
    pth=pth,
    n_classes=n_classes,
    linear_prob=LINEAR_PROBE
).to(device)

# =========================
# FREEZE / UNFREEZE
# =========================
def configure_finetuning(model, linear_probe):
    if linear_probe:
        logger.info("Mode: Linear Probing (backbone frozen)")
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        logger.info("Mode: Full Fine-tuning")
        for param in model.parameters():
            param.requires_grad = True

configure_finetuning(model, LINEAR_PROBE)

# =========================
# OPTIMIZER & SCHEDULER
# =========================
if LINEAR_PROBE:
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lp_lr,
        weight_decay=weight_decay
    )
else:
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.1, mode='max'
)

criterion = nn.BCEWithLogitsLoss()

# =========================
# LOG PARAMS
# =========================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info(f"Fine-tuning mode: {'Linear Probe' if LINEAR_PROBE else 'Full Fine-tuning'}")
logger.info(f"Total params: {total_params:,}")
logger.info(f"Trainable params: {trainable_params:,}")
logger.info(f"Learning rate: {lp_lr if LINEAR_PROBE else lr}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Epochs: {Epochs}")

# =========================
# TRAINING + TESTING
# =========================
best_test_auroc = 0.0
global_step = 0

for epoch in range(Epochs):
    logger.info(f"Starting Epoch {epoch + 1}/{Epochs}")
    
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} Training")):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        global_step += 1
    
    avg_train_loss = train_loss / len(trainloader)
    logger.info(f"Epoch {epoch + 1} | Average Training Loss: {avg_train_loss:.4f}")

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
        f"Test Macro AUROC: {test_macro_auroc:.4f} | "
        f"Test Micro AUROC: {test_micro_auroc:.4f} | "
        f"Test Macro AP: {test_macro_avg_prec:.4f} | "
        f"Test Micro AP: {test_micro_avg_prec:.4f}"
    )

    # Update learning rate
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(test_macro_auroc)
    new_lr = optimizer.param_groups[0]['lr']
    
    if new_lr != current_lr:
        logger.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")

    # -------- SAVE BEST --------
    if test_macro_auroc > best_test_auroc:
        best_test_auroc = test_macro_auroc
        logger.info(f"New best model! Test Macro AUROC: {best_test_auroc:.4f}")
        
        # Determine checkpoint filename based on training mode
        mode_suffix = "linear_probe" if LINEAR_PROBE else "full_ft"
        checkpoint_filename = f'checkpoint_{mode_suffix}_step{global_step}_auroc{test_macro_auroc:.4f}.pth'
        checkpoint_path = os.path.join(saved_dir, checkpoint_filename)
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_auroc': test_macro_auroc,
            'test_macro_auroc': test_macro_auroc,
            'test_micro_auroc': test_micro_auroc,
            'test_macro_avg_prec': test_macro_avg_prec,
            'test_micro_avg_prec': test_micro_avg_prec,
            'config': {
                'linear_probe': LINEAR_PROBE,
                'n_classes': n_classes,
                'batch_size': batch_size,
                'lr': lp_lr if LINEAR_PROBE else lr,
            }
        }
        
        # Save with custom filename
        torch.save(checkpoint_state, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_filename}")

    # -------- EARLY STOPPING --------
    if optimizer.param_groups[0]['lr'] < early_stop_lr:
        logger.info(f"Early stopping triggered: LR ({optimizer.param_groups[0]['lr']:.2e}) < threshold ({early_stop_lr:.2e})")
        break

logger.info("=" * 60)
logger.info("Training completed")
logger.info(f"Best Test Macro AUROC: {best_test_auroc:.4f}")
logger.info("=" * 60)