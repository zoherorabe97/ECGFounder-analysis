 
# =========================
# Quantile-based Membership Inference Attack (QMIA) for ECG Models
# =========================
# Based on "Quantile Regression for Membership Inference Attacks"
# https://arxiv.org/pdf/2307.03694

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# =========================
# Configuration & Setup
# =========================

class QMIAConfig:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.dataset = 'ecg'
        
        # Model paths
        self.base_model_path = './res/teacher/teacher_best.pth'  # Your trained ECG model
        self.results_dir = './results/qmia/'
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints/')
        
        # Data parameters
        self.test_batch_size = 128
        self.member_subset_size = 1000  # Number of training samples to use as members
        self.nonmember_subset_size = 1000  # Number of non-training samples as non-members
        
        # Attack model training
        self.attack_batch_size = 32
        self.attack_epochs = 30
        self.attack_learning_rate = 1e-4
        self.attack_optimizer = 'adam'
        
        # Quantile regression parameters
        self.low_quantile = -3  # log10(0.001)
        self.high_quantile = 0  # log10(1.0)
        self.num_quantiles = 100
        
        # HPO parameters (optional)
        self.use_hpo = False
        self.num_hpo_trials = 20
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# =========================
# Utility Functions
# =========================

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pinball_loss_fn(predictions, targets, quantiles):
    """
    Pinball loss for quantile regression
    
    Args:
        predictions: Model predictions (batch_size, num_quantiles)
        targets: Target scores (batch_size, 1)
        quantiles: Quantile values (1, num_quantiles)
    
    Returns:
        Loss tensor (batch_size, num_quantiles)
    """
    targets = targets.reshape([-1, 1])
    delta = targets - predictions
    
    loss = torch.nn.functional.relu(delta) * quantiles + \
           torch.nn.functional.relu(-delta) * (1.0 - quantiles)
    
    return loss

def hinge_scoring_fn(logits, labels, num_classes):
    """
    Compute hinge scores for quantile binning membership inference.
    
    Args:
        logits: Model output logits, shape (batch_size, num_classes)
        labels: True labels, shape (batch_size,) or (batch_size, num_classes) if one-hot
        num_classes: Number of classes
    
    Returns:
        Hinge scores, shape (batch_size,)
    """
    batch_size = logits.shape[0]
    
    # FIX: Handle one-hot encoded labels
    if labels.dim() > 1:
        # Convert one-hot to class indices
        batch_labels = labels.argmax(dim=1).long()
    else:
        batch_labels = labels.long()
    
    # Get logit for true class
    true_class_logits = logits[range(batch_size), batch_labels]
    
    # Get max logit among other classes
    # Create a mask for non-true classes
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[range(batch_size), batch_labels] = False
    
    # Get max of other class logits
    other_class_logits = logits.clone()
    other_class_logits[~mask] = float('-inf')
    max_other_logits = other_class_logits.max(dim=1)[0]
    
    # Compute hinge score: logit(true) - max(logit(other))
    hinge_scores = true_class_logits - max_other_logits
    
    return hinge_scores

def load_model(model_path, num_classes, model_architecture='net1d', device='cpu'):
    """
    Load a trained ECG model
    
    Args:
        model_path: Path to model checkpoint
        num_classes: Number of classes
        model_architecture: Type of model ('net1d', 'cnn', 'transformer')
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if model_architecture == 'net1d':
        from net1d import Net1D
        model = Net1D(
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
            n_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def extract_features(model, data_loader, device, num_batches=None):
    """
    Extract logits from model
    
    Args:
        model: ECG model
        data_loader: DataLoader for ECG signals
        device: Device to compute on
        num_batches: Maximum number of batches to process (None = all)
    
    Returns:
        all_logits: Stacked logits
        all_labels: Stacked labels
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (ecg_signals, labels) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            ecg_signals = ecg_signals.to(device)
            labels = labels.to(device)
            
            logits = model(ecg_signals)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels

# =========================
# Attack Model
# =========================

class QuantileRegressionHead(nn.Module):
    """
    Quantile regression head for attack model
    Maps logits to quantile predictions
    """
    def __init__(self, input_dim, num_quantiles):
        super(QuantileRegressionHead, self).__init__()
        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        
        # Simple linear layer mapping logits to quantile predictions
        self.fc = nn.Linear(input_dim, num_quantiles)
        self.fc.weight.data.fill_(0.0)
        self.fc.bias.data.fill_(0.0)
    
    def forward(self, x):
        """Forward pass"""
        return self.fc(x)

# =========================
# Attack Trainer
# =========================

class QMIAAttacker:
    """Main QMIA attack class"""
    
    def __init__(self, config, num_classes):
        self.config = config
        self.num_classes = num_classes
        self.device = config.device
        
        # Create quantile tensor
        self.quantiles = torch.sort(
            1 - torch.logspace(
                config.low_quantile,
                config.high_quantile,
                config.num_quantiles,
                requires_grad=False
            )
        )[0].reshape([1, -1]).to(self.device)
        
        print(f"[QMIA] Quantiles created: {self.quantiles.shape}")
        print(f"[QMIA] Min quantile: {self.quantiles.min().item():.4f}, "
              f"Max quantile: {self.quantiles.max().item():.4f}")
    
    def create_attack_model(self, feature_dim):
        """Create attack model"""
        model = QuantileRegressionHead(
            input_dim=feature_dim,
            num_quantiles=self.config.num_quantiles
        )
        model.to(self.device)
        return model
    
    def compute_hinge_scores(self, logits, labels):
        """Compute hinge scores from logits and labels"""
        return hinge_scoring_fn(logits, labels, self.num_classes)
    
    def train_attack_model(self, attack_model, member_logits, member_labels):
        """
        Train attack model on member data
        
        Args:
            attack_model: Quantile regression model
            member_logits: Logits from member samples
            member_labels: Labels for member samples
        """
        print("\n[QMIA] Training attack model...")
        
        member_logits = member_logits.to(self.device)
        member_labels = member_labels.to(self.device)
        
        # Create data loader
        train_loader = DataLoader(
            TensorDataset(member_logits, member_labels),
            batch_size=self.config.attack_batch_size,
            shuffle=True
        )
        
        # Setup optimizer
        if self.config.attack_optimizer == 'adam':
            optimizer = optim.Adam(attack_model.parameters(), 
                                  lr=self.config.attack_learning_rate)
        else:
            optimizer = optim.SGD(attack_model.parameters(),
                                 lr=self.config.attack_learning_rate,
                                 momentum=0.9)
        
        attack_model.train()
        
        for epoch in range(self.config.attack_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_logits, batch_labels in train_loader:
                optimizer.zero_grad()
                
                # Compute hinge scores (target for attack model)
                target_scores = self.compute_hinge_scores(batch_logits, batch_labels)
                
                # Forward pass
                predictions = attack_model(batch_logits)
                
                # Compute loss
                loss = pinball_loss_fn(predictions, target_scores, self.quantiles).mean()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.config.attack_epochs}: Loss = {avg_loss:.6f}")
        
        return attack_model
    
    def evaluate_attack(self, attack_model, member_logits, member_labels, 
                       nonmember_logits, nonmember_labels):
        """
        Evaluate attack on member and non-member data
        
        Returns:
            Dictionary with attack metrics
        """
        print("\n[QMIA] Evaluating attack...")
        
        attack_model.eval()
        
        # Compute scores for members
        with torch.no_grad():
            member_logits = member_logits.to(self.device)
            member_labels = member_labels.to(self.device)
            
            member_predictions = attack_model(member_logits)
            member_target_scores = self.compute_hinge_scores(member_logits, member_labels)
            
            # Compute loss for members
            member_loss = pinball_loss_fn(member_predictions, member_target_scores, 
                                         self.quantiles).mean(dim=1)
        
        # Compute scores for non-members
        with torch.no_grad():
            nonmember_logits = nonmember_logits.to(self.device)
            nonmember_labels = nonmember_labels.to(self.device)
            
            nonmember_predictions = attack_model(nonmember_logits)
            nonmember_target_scores = self.compute_hinge_scores(nonmember_logits, nonmember_labels)
            
            # Compute loss for non-members
            nonmember_loss = pinball_loss_fn(nonmember_predictions, nonmember_target_scores,
                                            self.quantiles).mean(dim=1)
        
        # Compute ROC curve
        member_loss_np = member_loss.cpu().numpy()
        nonmember_loss_np = nonmember_loss.cpu().numpy()
        
        # Create labels: 1 for members, 0 for non-members
        true_labels = np.concatenate([np.ones(len(member_loss_np)), 
                                      np.zeros(len(nonmember_loss_np))])
        predictions = np.concatenate([member_loss_np, nonmember_loss_np])
        
        # For ROC curve, we want higher score = more likely member
        # So we negate the loss (lower loss = higher confidence of membership)
        predictions_negated = -predictions
        
        fpr, tpr, thresholds = roc_curve(true_labels, predictions_negated)
        attack_auc = auc(fpr, tpr)
        
        results = {
            'member_loss_mean': member_loss_np.mean(),
            'member_loss_std': member_loss_np.std(),
            'nonmember_loss_mean': nonmember_loss_np.mean(),
            'nonmember_loss_std': nonmember_loss_np.std(),
            'attack_auc': attack_auc,
            'fpr': fpr,
            'tpr': tpr,
            'member_losses': member_loss_np,
            'nonmember_losses': nonmember_loss_np
        }
        
        return results

# =========================
# Data Loading
# =========================

class ECGDataLoader:
    """Load ECG training and test data"""
    
    def __init__(self, config, dataset_name='G12EC'):
        self.config = config
        self.dataset_name = dataset_name
        self.device = config.device
    
    def load_data(self, csv_path, ecg_dir='', num_samples=None):
        """
        Load ECG data from CSV
        
        Args:
            csv_path: Path to CSV file with ECG data
            ecg_dir: Base directory for ECG files
            num_samples: Maximum number of samples to load (None = all)
        
        Returns:
            logits: Model logits
            labels: Class labels
        """
        import pandas as pd
        
        print(f"[QMIA] Loading ECG data from {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Extract features and labels (adjust based on your CSV format)
        # Assuming columns are: filename, label_1, label_2, ..., label_n
        
        if num_samples is not None:
            df = df.iloc[:num_samples]
        
        print(f"[QMIA] Loaded {len(df)} samples from {csv_path}")
        
        return df

# =========================
# Main QMIA Script
# =========================

class QMIARunner:
    """Main runner for QMIA attack"""
    
    def __init__(self, config):
        self.config = config
        self.config.create_directories()
        
        print("="*80)
        print("Quantile-based Membership Inference Attack (QMIA)")
        print("="*80)
        print(f"Device: {config.device}")
        print(f"Results directory: {config.results_dir}")
        print("="*80)
    
    def load_ecg_model(self):
        """Load trained ECG model"""
        print(f"\n[QMIA] Loading ECG model from {self.config.base_model_path}")
        
        if not os.path.exists(self.config.base_model_path):
            raise FileNotFoundError(f"Model not found at {self.config.base_model_path}")
        
        # Determine number of classes from your dataset
        # For ECG, this might be binary or multi-class
        num_classes = 17  # Adjust based on your ECG task
        
        model = load_model(
            self.config.base_model_path,
            num_classes=num_classes,
            model_architecture='net1d',
            device=self.config.device
        )
        
        return model, num_classes
    
    def create_synthetic_member_nonmember_data(self, model, num_classes):
        """
        Create synthetic member and non-member datasets
        
        This is a placeholder. In a real scenario, you would:
        1. Use actual training data for members
        2. Use actual holdout test data for non-members
        """
        print("\n[QMIA] Creating synthetic member/non-member data...")
        
        # Create dummy ECG signals (12 channels, 5000 time steps)
        ecg_shape = (self.config.member_subset_size, 12, 5000)
        member_signals = torch.randn(ecg_shape)
        member_labels = torch.randint(0, num_classes, 
                                     (self.config.member_subset_size,))
        
        # Non-member data
        nonmember_shape = (self.config.nonmember_subset_size, 12, 5000)
        nonmember_signals = torch.randn(nonmember_shape)
        nonmember_labels = torch.randint(0, num_classes,
                                        (self.config.nonmember_subset_size,))
        
        print(f"[QMIA] Created member data: {member_signals.shape}")
        print(f"[QMIA] Created non-member data: {nonmember_signals.shape}")
        
        return member_signals, member_labels, nonmember_signals, nonmember_labels
    
    def load_real_data(self, train_csv, test_csv, model):
        """
        Load real ECG data
        
        Args:
            train_csv: Path to training CSV
            test_csv: Path to test CSV
            model: Trained model
        
        Returns:
            member_logits, member_labels, nonmember_logits, nonmember_labels
        """
        from dataset import LVEF_12lead_cls_Dataset
        
        print("\n[QMIA] Loading real ECG data...")
        
        # Load training data (members)
        member_dataset = LVEF_12lead_cls_Dataset(
            ecg_path='',
            labels_df=pd.read_csv(train_csv)
        )
        
        # Load test data (non-members)
        nonmember_dataset = LVEF_12lead_cls_Dataset(
            ecg_path='',
            labels_df=pd.read_csv(test_csv)
        )
        
        # Create data loaders
        member_loader = DataLoader(
            member_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=8
        )
        
        nonmember_loader = DataLoader(
            nonmember_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=8
        )
        
        # Extract logits
        print("[QMIA] Extracting member logits...")
        member_logits, member_labels = extract_features(
            model, member_loader, self.config.device
        )
        
        print("[QMIA] Extracting non-member logits...")
        nonmember_logits, nonmember_labels = extract_features(
            model, nonmember_loader, self.config.device
        )
        
        # Subsample if needed
        if len(member_logits) > self.config.member_subset_size:
            indices = torch.randperm(len(member_logits))[:self.config.member_subset_size]
            member_logits = member_logits[indices]
            member_labels = member_labels[indices]
        
        if len(nonmember_logits) > self.config.nonmember_subset_size:
            indices = torch.randperm(len(nonmember_logits))[:self.config.nonmember_subset_size]
            nonmember_logits = nonmember_logits[indices]
            nonmember_labels = nonmember_labels[indices]
        
        print(f"[QMIA] Member logits: {member_logits.shape}")
        print(f"[QMIA] Non-member logits: {nonmember_logits.shape}")
        
        return member_logits, member_labels, nonmember_logits, nonmember_labels
    
    def run(self, use_real_data=False, train_csv=None, test_csv=None):
        """
        Run the complete QMIA attack
        
        Args:
            use_real_data: Whether to use real ECG data
            train_csv: Path to training CSV (if using real data)
            test_csv: Path to test CSV (if using real data)
        """
        # Set seeds
        set_seeds(self.config.seed)
        
        # Load ECG model
        model, num_classes = self.load_ecg_model()
        
        # Load data
        if use_real_data and train_csv and test_csv:
            member_logits, member_labels, nonmember_logits, nonmember_labels = \
                self.load_real_data(train_csv, test_csv, model)
        else:
            print("\n[WARNING] Using synthetic data. For real evaluation, provide CSV paths.")
            member_signals, member_labels, nonmember_signals, nonmember_labels = \
                self.create_synthetic_member_nonmember_data(model, num_classes)
            
            # Extract logits from signals
            member_loader = DataLoader(
                TensorDataset(member_signals, member_labels),
                batch_size=self.config.test_batch_size,
                shuffle=False
            )
            nonmember_loader = DataLoader(
                TensorDataset(nonmember_signals, nonmember_labels),
                batch_size=self.config.test_batch_size,
                shuffle=False
            )
            
            member_logits, member_labels = extract_features(
                model, member_loader, self.config.device
            )
            nonmember_logits, nonmember_labels = extract_features(
                model, nonmember_loader, self.config.device
            )
        
        # Create attacker
        attacker = QMIAAttacker(self.config, num_classes)
        
        # Create and train attack model
        attack_model = attacker.create_attack_model(feature_dim=num_classes)
        attack_model = attacker.train_attack_model(attack_model, nonmember_logits, nonmember_labels)
        
        # Evaluate attack
        results = attacker.evaluate_attack(
            attack_model, member_logits, member_labels,
            nonmember_logits, nonmember_labels
        )
        
        # Print results
        self.print_results(results)
        
        # Save results
        self.save_results(results, attack_model)
        
        # Plot ROC curve
        self.plot_roc_curve(results)
        
        return results
    
    def print_results(self, results):
        """Print attack results"""
        print("\n" + "="*80)
        print("QMIA Attack Results")
        print("="*80)
        print(f"Member Loss - Mean: {results['member_loss_mean']:.6f} ± {results['member_loss_std']:.6f}")
        print(f"Non-member Loss - Mean: {results['nonmember_loss_mean']:.6f} ± {results['nonmember_loss_std']:.6f}")
        print(f"Attack AUC: {results['attack_auc']:.4f}")
        print("="*80)
        
        # Interpretation
        if results['attack_auc'] > 0.9:
            print("⚠️  CRITICAL: Model is highly vulnerable to membership inference attacks!")
        elif results['attack_auc'] > 0.7:
            print("⚠️  WARNING: Model shows significant vulnerability to membership inference attacks")
        elif results['attack_auc'] > 0.6:
            print("⚠️  CAUTION: Model shows moderate vulnerability to membership inference attacks")
        else:
            print("✓ Model appears resistant to membership inference attacks")
        print("="*80)
    
    def save_results(self, results, attack_model):
        """Save results to disk"""
        # Save metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'member_loss_mean': float(results['member_loss_mean']),
            'member_loss_std': float(results['member_loss_std']),
            'nonmember_loss_mean': float(results['nonmember_loss_mean']),
            'nonmember_loss_std': float(results['nonmember_loss_std']),
            'attack_auc': float(results['attack_auc']),
            'num_quantiles': self.config.num_quantiles,
            'member_samples': len(results['member_losses']),
            'nonmember_samples': len(results['nonmember_losses'])
        }
        
        metrics_path = os.path.join(self.config.results_dir, 'qmia_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[QMIA] Metrics saved to {metrics_path}")
        
        # Save raw predictions
        predictions = {
            'member_losses': results['member_losses'],
            'nonmember_losses': results['nonmember_losses'],
            'fpr': results['fpr'],
            'tpr': results['tpr']
        }
        
        predictions_path = os.path.join(self.config.results_dir, 'qmia_predictions.pkl')
        with open(predictions_path, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"[QMIA] Predictions saved to {predictions_path}")
        
        # Save model
        model_path = os.path.join(self.config.checkpoint_dir, 'attack_model.pth')
        torch.save(attack_model.state_dict(), model_path)
        print(f"[QMIA] Attack model saved to {model_path}")
    
    def plot_roc_curve(self, results):
        """Plot and save ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(results['fpr'], results['tpr'], 
                label=f"QMIA (AUC = {results['attack_auc']:.4f})",
                linewidth=2.5)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('QMIA Attack ROC Curve on ECG Model', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.config.results_dir, 'qmia_roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"\n[QMIA] ROC curve saved to {roc_path}")
        plt.close()
    
    def plot_loss_distributions(self, results):
        """Plot member vs non-member loss distributions"""
        plt.figure(figsize=(12, 5))
        
        # Loss distributions
        plt.subplot(1, 2, 1)
        plt.hist(results['member_losses'], bins=30, alpha=0.6, label='Members', color='red')
        plt.hist(results['nonmember_losses'], bins=30, alpha=0.6, label='Non-members', color='blue')
        plt.xlabel('Loss', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.title('Member vs Non-member Loss Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [results['member_losses'], results['nonmember_losses']]
        plt.boxplot(data_to_plot, labels=['Members', 'Non-members'])
        plt.ylabel('Loss', fontsize=11)
        plt.title('Loss Distribution Comparison', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        dist_path = os.path.join(self.config.results_dir, 'qmia_loss_distributions.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        print(f"[QMIA] Loss distributions saved to {dist_path}")
        plt.close()

# =========================
# Command Line Interface
# =========================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Quantile-based Membership Inference Attack on ECG Models'
    )
    
    # Model paths
    parser.add_argument('--model_path', type=str,
                       default='./res/teacher/teacher_best.pth',
                       help='Path to trained ECG model')
    
    # Data paths
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real ECG data instead of synthetic')
    parser.add_argument('--train_csv', type=str, default=None,
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV file')
    
    # Attack parameters
    parser.add_argument('--attack_epochs', type=int, default=30,
                       help='Number of attack training epochs')
    parser.add_argument('--attack_lr', type=float, default=1e-4,
                       help='Attack model learning rate')
    parser.add_argument('--attack_batch_size', type=int, default=32,
                       help='Attack training batch size')
    
    # Quantile parameters
    parser.add_argument('--num_quantiles', type=int, default=100,
                       help='Number of quantiles for regression')
    parser.add_argument('--low_quantile', type=float, default=-3,
                       help='Log10 of lowest quantile')
    parser.add_argument('--high_quantile', type=float, default=0,
                       help='Log10 of highest quantile')
    
    # Data size
    parser.add_argument('--member_samples', type=int, default=1000,
                       help='Number of member samples')
    parser.add_argument('--nonmember_samples', type=int, default=1000,
                       help='Number of non-member samples')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--results_dir', type=str, default='./results/qmia/',
                       help='Directory to save results')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create config
    config = QMIAConfig()
    
    # Update config with command line arguments
    config.base_model_path = args.model_path
    config.results_dir = args.results_dir
    config.checkpoint_dir = os.path.join(args.results_dir, 'checkpoints/')
    config.attack_epochs = args.attack_epochs
    config.attack_learning_rate = args.attack_lr
    config.attack_batch_size = args.attack_batch_size
    config.num_quantiles = args.num_quantiles
    config.low_quantile = args.low_quantile
    config.high_quantile = args.high_quantile
    config.member_subset_size = args.member_samples
    config.nonmember_subset_size = args.nonmember_samples
    config.seed = args.seed
    
    # Create and run QMIA
    runner = QMIARunner(config)
    results = runner.run(
        use_real_data=args.use_real_data,
        train_csv=args.train_csv,
        test_csv=args.test_csv
    )
    
    # Plot additional visualizations
    runner.plot_loss_distributions(results)
    
    print("\n" + "="*80)
    print("QMIA Attack Complete!")
    print(f"Results saved to: {config.results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
