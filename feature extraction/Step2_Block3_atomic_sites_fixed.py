#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 STEP 2 - BLOCK 3: ATOMIC SITES SELF-SUPERVISED LEARNING")
print("SpatialGraphNet: Fractional Coordinates → Dense Geometric Features")
print("=" * 80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  Using CPU")

print(f"🔧 Device: {device}")

class AtomicSitesDataset(Dataset):
    def __init__(self, X_coords, y_synth=None, k_neighbors=8, augment=True):
        """
        Dataset for Atomic Sites Self-Supervised Learning
        
        Args:
            X_coords: (N, 100, 3) fractional coordinates 
            y_synth: Synthesis labels (for final evaluation only)
            k_neighbors: Number of neighbors for graph construction
            augment: Whether to apply coordinate augmentation
        """
        self.X_coords = torch.FloatTensor(X_coords)
        self.y_synth = torch.LongTensor(y_synth) if y_synth is not None else None
        self.k_neighbors = k_neighbors
        self.augment = augment
        
        print(f"🔬 Dataset created: {len(self.X_coords)} samples")
        
        # Pre-analyze coordinate patterns
        self._analyze_coordinates()
        
    def _analyze_coordinates(self):
        """Analyze coordinate patterns for better understanding"""
        coords_flat = self.X_coords.reshape(-1, 3)
        
        # Find non-zero coordinates (remove padding)
        non_zero_mask = (coords_flat.abs().sum(dim=1) > 1e-6)
        active_coords = coords_flat[non_zero_mask]
        
        self.coord_stats = {
            'total_sites': len(coords_flat),
            'active_sites': len(active_coords),
            'sparsity': (1 - len(active_coords) / len(coords_flat)) * 100,
            'coord_mean': active_coords.mean(dim=0),
            'coord_std': active_coords.std(dim=0)
        }
        
        print(f"📊 Coordinate Analysis:")
        print(f"   Active sites: {self.coord_stats['active_sites']:,} / {self.coord_stats['total_sites']:,}")
        print(f"   Sparsity: {self.coord_stats['sparsity']:.1f}%")
        print(f"   Coord range: [{active_coords.min():.3f}, {active_coords.max():.3f}]")
        
    def __len__(self):
        return len(self.X_coords)
    
    def __getitem__(self, idx):
        coords = self.X_coords[idx]  # (100, 3)
        
        # Remove padding sites
        site_mask = (coords.abs().sum(dim=1) > 1e-6)
        active_coords = coords[site_mask]  # (n_active, 3)
        
        if len(active_coords) < 3:
            # Handle edge case with very few sites - pad with small values
            n_needed = 3 - len(active_coords)
            padding = torch.randn(n_needed, 3) * 0.01
            active_coords = torch.cat([active_coords, padding], dim=0)
            
        # Limit max sites to prevent memory issues
        if len(active_coords) > 50:
            # Sample 50 sites randomly
            indices = torch.randperm(len(active_coords))[:50]
            active_coords = active_coords[indices]
            
        # Apply augmentation during training
        if self.augment:
            active_coords = self._augment_coordinates(active_coords)
            
        # Create self-supervised tasks
        mcm_data = self._create_masked_coordinate_task(active_coords)
        env_data = self._create_environment_task(active_coords)
        
        return {
            'coords': active_coords,
            'mcm': mcm_data,
            'env': env_data,
            'synth_label': self.y_synth[idx] if self.y_synth is not None else -1,
            'n_sites': len(active_coords)
        }
    
    def _augment_coordinates(self, coords):
        """Apply coordinate augmentation"""
        # Small random noise (±0.02)
        noise = torch.randn_like(coords) * 0.02
        coords = coords + noise
        
        # Random rotation around z-axis (preserving crystal structure)
        if torch.rand(1) < 0.3:
            angle = torch.rand(1) * 2 * np.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=coords.dtype)
            coords = torch.matmul(coords, rot_matrix.T)
            
        return coords
    
    def _create_masked_coordinate_task(self, coords):
        """Create masked coordinate modeling task"""
        n_sites = len(coords)
        
        # Mask 15% of coordinates
        n_mask = max(1, int(0.15 * n_sites))
        mask_indices = torch.randperm(n_sites)[:n_mask]
        
        # Create masked coordinates
        masked_coords = coords.clone()
        original_values = coords[mask_indices].clone()
        masked_coords[mask_indices] = 0.0  # Mask with zeros
        
        return {
            'masked_coords': masked_coords,
            'mask_indices': mask_indices,
            'target_values': original_values
        }
    
    def _create_environment_task(self, coords):
        """Create local environment classification task"""
        n_sites = len(coords)
        
        # Calculate coordination numbers using pairwise distances
        distances = torch.cdist(coords, coords)  # (n_sites, n_sites)
        
        coord_numbers = []
        env_labels = []
        
        for i in range(n_sites):
            # Count neighbors within 0.3 cutoff (excluding self)
            neighbors = (distances[i] < 0.3) & (distances[i] > 1e-6)
            coord_num = neighbors.sum().item()
            coord_numbers.append(coord_num)
            
            # Classify environment
            if coord_num <= 2:
                env_labels.append(0)  # Isolated
            elif coord_num <= 5:
                env_labels.append(1)  # Under-coordinated
            elif coord_num <= 8:
                env_labels.append(2)  # Well-coordinated
            else:
                env_labels.append(3)  # Over-coordinated
                
        return {
            'coord_numbers': torch.tensor(coord_numbers, dtype=torch.float),
            'env_labels': torch.tensor(env_labels, dtype=torch.long)
        }

class SpatialGraphNet(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input embedding
        self.coord_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Self-attention layers for graph processing
        self.attention1 = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.attention2 = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.attention3 = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)
        
        # Feedforward networks
        self.ff1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        self.ff3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Global pooling
        self.global_attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, 128))
        
        # Task-specific heads
        self.coord_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        
        self.env_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)  # 4 environment classes
        )
        
        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, coords_list, return_features=False):
        batch_size = len(coords_list)
        batch_features = []
        coord_predictions = []
        env_predictions = []
        
        for i, coords in enumerate(coords_list):
            # Coordinate embedding
            x = self.coord_embed(coords)  # (n_sites, 128)
            x = x.unsqueeze(0)  # (1, n_sites, 128)
            
            # Multi-layer self-attention
            # Layer 1
            x_att, _ = self.attention1(x, x, x)
            x = self.norm1(x + x_att)
            x = x + self.ff1(x)
            
            # Layer 2
            x_att, _ = self.attention2(x, x, x)
            x = self.norm2(x + x_att)
            x = x + self.ff2(x)
            
            # Layer 3
            x_att, _ = self.attention3(x, x, x)
            x = self.norm3(x + x_att)
            x = x + self.ff3(x)
            
            # Global pooling via attention
            query = self.global_query.expand(1, -1, -1)  # (1, 1, 128)
            global_feat, _ = self.global_attention(query, x, x)
            global_feat = global_feat.squeeze(0).squeeze(0)  # (128,)
            
            batch_features.append(global_feat)
            
            if not return_features:
                # Task predictions
                x_nodes = x.squeeze(0)  # (n_sites, 128)
                coord_pred = self.coord_head(x_nodes)  # (n_sites, 3)
                env_pred = self.env_head(x_nodes)      # (n_sites, 4)
                
                coord_predictions.append(coord_pred)
                env_predictions.append(env_pred)
        
        # Stack batch features
        batch_features = torch.stack(batch_features)  # (batch_size, 128)
        
        if return_features:
            return self.feature_proj(batch_features)
        
        return {
            'coord_pred': coord_predictions,
            'env_pred': env_predictions,
            'features': self.feature_proj(batch_features)
        }

def custom_collate(batch):
    """Custom collate function for variable-length sequences"""
    coords_list = [item['coords'] for item in batch]
    mcm_data = [item['mcm'] for item in batch]
    env_data = [item['env'] for item in batch]
    synth_labels = torch.tensor([item['synth_label'] for item in batch])
    n_sites = torch.tensor([item['n_sites'] for item in batch])
    
    return {
        'coords': coords_list,
        'mcm': mcm_data,
        'env': env_data,
        'synth_labels': synth_labels,
        'n_sites': n_sites
    }

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    mcm_losses = []
    env_losses = []
    env_accs = []
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Move coordinates to device
        coords_list = [coords.to(device) for coords in batch['coords']]
        n_sites = batch['n_sites']
        
        # Forward pass
        outputs = model(coords_list)
        
        # Calculate losses
        mcm_loss = 0
        env_loss = 0
        total_env_correct = 0
        total_env_samples = 0
        
        batch_size = len(batch['mcm'])
        
        for i in range(batch_size):
            # Masked Coordinate Modeling Loss
            mcm_data = batch['mcm'][i]
            if len(mcm_data['mask_indices']) > 0:
                pred_coords = outputs['coord_pred'][i]
                target_coords = mcm_data['target_values'].to(device)
                mask_indices = mcm_data['mask_indices']
                
                mcm_loss += F.mse_loss(pred_coords[mask_indices], target_coords)
            
            # Environment Classification Loss
            env_data = batch['env'][i]
            env_pred = outputs['env_pred'][i]
            env_target = env_data['env_labels'].to(device)
            
            env_loss += F.cross_entropy(env_pred, env_target)
            
            # Environment accuracy
            env_pred_classes = env_pred.argmax(dim=1)
            total_env_correct += (env_pred_classes == env_target).sum().item()
            total_env_samples += len(env_target)
        
        # Combine losses (60% MCM, 40% ENV)
        mcm_loss = mcm_loss / batch_size
        env_loss = env_loss / batch_size
        total_loss_batch = 0.6 * mcm_loss + 0.4 * env_loss
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += total_loss_batch.item()
        mcm_losses.append(mcm_loss.item())
        env_losses.append(env_loss.item())
        env_accs.append(total_env_correct / total_env_samples if total_env_samples > 0 else 0)
        
        # Progress monitoring
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx:3d}/{len(dataloader)} | "
                  f"Loss: {total_loss_batch.item():.4f} | "
                  f"MCM: {mcm_loss.item():.4f} | "
                  f"ENV: {env_loss.item():.4f} | "
                  f"ENV_Acc: {total_env_correct/total_env_samples:.3f}")
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mcm_loss': np.mean(mcm_losses),
        'env_loss': np.mean(env_losses),
        'env_acc': np.mean(env_accs)
    }

def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    mcm_losses = []
    env_losses = []
    env_accs = []
    
    with torch.no_grad():
        for batch in dataloader:
            coords_list = [coords.to(device) for coords in batch['coords']]
            outputs = model(coords_list)
            
            mcm_loss = 0
            env_loss = 0
            total_env_correct = 0
            total_env_samples = 0
            
            batch_size = len(batch['mcm'])
            
            for i in range(batch_size):
                # MCM Loss
                mcm_data = batch['mcm'][i]
                if len(mcm_data['mask_indices']) > 0:
                    pred_coords = outputs['coord_pred'][i]
                    target_coords = mcm_data['target_values'].to(device)
                    mask_indices = mcm_data['mask_indices']
                    mcm_loss += F.mse_loss(pred_coords[mask_indices], target_coords)
                
                # ENV Loss
                env_data = batch['env'][i]
                env_pred = outputs['env_pred'][i]
                env_target = env_data['env_labels'].to(device)
                env_loss += F.cross_entropy(env_pred, env_target)
                
                env_pred_classes = env_pred.argmax(dim=1)
                total_env_correct += (env_pred_classes == env_target).sum().item()
                total_env_samples += len(env_target)
            
            mcm_loss = mcm_loss / batch_size
            env_loss = env_loss / batch_size
            total_loss_batch = 0.6 * mcm_loss + 0.4 * env_loss
            
            total_loss += total_loss_batch.item()
            mcm_losses.append(mcm_loss.item())
            env_losses.append(env_loss.item())
            env_accs.append(total_env_correct / total_env_samples if total_env_samples > 0 else 0)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mcm_loss': np.mean(mcm_losses),
        'env_loss': np.mean(env_losses),
        'env_acc': np.mean(env_accs)
    }

# Load datasets
print("\n📂 Loading datasets...")
X_train = np.load('X_train_final.npy')
X_val = np.load('X_validation_final.npy')
X_test = np.load('X_test_final.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

print(f"✅ Datasets loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# Extract atomic coordinates: FTCP[105:205, 0:3]
print("\n🔬 Extracting atomic coordinates (FTCP[105:205, 0:3])...")
X_coords_train = X_train[:, 105:205, 0:3]
X_coords_val = X_val[:, 105:205, 0:3]
X_coords_test = X_test[:, 105:205, 0:3]

print(f"Coordinate shapes: Train {X_coords_train.shape}, Val {X_coords_val.shape}, Test {X_coords_test.shape}")

# Create datasets (reduced size for efficiency)
sample_size_train = min(20000, len(X_coords_train))
sample_size_val = min(8000, len(X_coords_val))
sample_size_test = min(3000, len(X_coords_test))

print(f"\n🎯 Using sample sizes: Train {sample_size_train}, Val {sample_size_val}, Test {sample_size_test}")

train_dataset = AtomicSitesDataset(X_coords_train[:sample_size_train], y_train[:sample_size_train], augment=True)
val_dataset = AtomicSitesDataset(X_coords_val[:sample_size_val], y_val[:sample_size_val], augment=False)
test_dataset = AtomicSitesDataset(X_coords_test[:sample_size_test], y_test[:sample_size_test], augment=False)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         collate_fn=custom_collate, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       collate_fn=custom_collate, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=custom_collate, num_workers=0)

print(f"✅ Dataloaders created - Batch size: {batch_size}")

# Initialize model
model = SpatialGraphNet(hidden_dim=256, output_dim=256).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

print(f"\n🏋️  Model Configuration:")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Hidden dim: 256, Output dim: 256")
print(f"   Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")

# Training loop
print("\n🚀 Starting Block 3 Training...")
history = {
    'train_total_loss': [], 'val_total_loss': [], 'train_mcm_loss': [], 'val_mcm_loss': [],
    'train_env_loss': [], 'val_env_loss': [], 'train_env_acc': [], 'val_env_acc': []
}

best_val_loss = float('inf')
patience = 6
patience_counter = 0
num_epochs = 20

for epoch in range(num_epochs):
    print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
    start_time = time.time()
    
    # Training
    train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
    
    # Validation
    val_metrics = validate_model(model, val_loader, device)
    
    # Scheduler step
    scheduler.step()
    
    # Save metrics
    for key in train_metrics:
        history[f'train_{key}'].append(train_metrics[key])
        history[f'val_{key}'].append(val_metrics[key])
    
    # Early stopping
    if val_metrics['total_loss'] < best_val_loss:
        best_val_loss = val_metrics['total_loss']
        patience_counter = 0
        torch.save(model.state_dict(), 'Step2_Block3_atomic_sites_best.pth')
        print("  💾 New best model saved!")
    else:
        patience_counter += 1
    
    epoch_time = time.time() - start_time
    
    # Progress report
    print(f"  🎯 Results: Loss {val_metrics['total_loss']:.4f} | "
          f"MCM {val_metrics['mcm_loss']:.4f} | "
          f"ENV {val_metrics['env_loss']:.4f} | "
          f"ENV_Acc {val_metrics['env_acc']:.3f} | "
          f"Time: {epoch_time:.1f}s")
    
    if patience_counter >= patience:
        print(f"⏰ Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('Step2_Block3_atomic_sites_best.pth'))

# Final evaluation
print("\n📊 FINAL EVALUATION:")
train_final = validate_model(model, train_loader, device)
val_final = validate_model(model, val_loader, device)
test_final = validate_model(model, test_loader, device)

print(f"🎯 Self-Supervised Results:")
print(f"   MCM Loss      - Train: {train_final['mcm_loss']:.4f}, Val: {val_final['mcm_loss']:.4f}, Test: {test_final['mcm_loss']:.4f}")
print(f"   ENV Accuracy  - Train: {train_final['env_acc']:.3f}, Val: {val_final['env_acc']:.3f}, Test: {test_final['env_acc']:.3f}")

# Extract features for synthesis prediction
print("\n💾 Extracting features for synthesis evaluation...")
model.eval()

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            coords_list = [coords.to(device) for coords in batch['coords']]
            batch_features = model(coords_list, return_features=True)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch['synth_labels'].numpy())
    return np.vstack(features), np.array(labels)

h3_train, y_train_eval = extract_features(train_loader)
h3_val, y_val_eval = extract_features(val_loader)
h3_test, y_test_eval = extract_features(test_loader)

print(f"✅ Features extracted: Train {h3_train.shape}, Val {h3_val.shape}, Test {h3_test.shape}")

# Synthesis prediction evaluation
print("\n🎯 SYNTHESIS PREDICTION EVALUATION:")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(h3_train, y_train_eval)

# Predictions
train_pred = clf.predict_proba(h3_train)[:, 1]
val_pred = clf.predict_proba(h3_val)[:, 1]
test_pred = clf.predict_proba(h3_test)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train_eval, train_pred)
val_auc = roc_auc_score(y_val_eval, val_pred)
test_auc = roc_auc_score(y_test_eval, test_pred)

print(f"🎯 Synthesis AUC Results:")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Val AUC: {val_auc:.4f}")
print(f"   Test AUC: {test_auc:.4f}")

# Save features
np.save('Step2_Block3_features_train.npy', h3_train)
np.save('Step2_Block3_features_val.npy', h3_val)
np.save('Step2_Block3_features_test.npy', h3_test)

print(f"\n✅ Block 3 features saved!")

# Create visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Block 3: Atomic Sites Self-Supervised Learning Results', fontsize=16, fontweight='bold')

# Training curves
ax1 = axes[0, 0]
epochs = range(1, len(history['train_total_loss']) + 1)
ax1.plot(epochs, history['train_total_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs, history['val_total_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Total Loss', fontweight='bold')
ax1.set_title('Training Loss Curves', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Environment accuracy
ax2 = axes[0, 1]
ax2.plot(epochs, history['train_env_acc'], 'g-', label='Train ENV Acc', linewidth=2)
ax2.plot(epochs, history['val_env_acc'], 'orange', label='Val ENV Acc', linewidth=2)
ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Environment Accuracy', fontweight='bold')
ax2.set_title('Environment Classification Accuracy', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Task losses
ax3 = axes[1, 0]
ax3.plot(epochs, history['train_mcm_loss'], 'b-', label='Train MCM', linewidth=2)
ax3.plot(epochs, history['val_mcm_loss'], 'r-', label='Val MCM', linewidth=2)
ax3.axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Target (0.02)')
ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('MCM Loss', fontweight='bold')
ax3.set_title('Masked Coordinate Modeling Loss', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# AUC comparison
ax4 = axes[1, 1]
auc_scores = [train_auc, val_auc, test_auc]
datasets = ['Train', 'Validation', 'Test']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax4.bar(datasets, auc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax4.set_ylabel('AUC Score', fontweight='bold')
ax4.set_title('Synthesis Prediction AUC', fontweight='bold')
ax4.axhline(y=0.73, color='red', linestyle='--', linewidth=2, label='Target (0.73)')
ax4.set_ylim(0.5, 1.0)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('Step2_Block3_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("🎯 BLOCK 3 COMPLETED SUCCESSFULLY!")
print(f"✨ Target ENV Accuracy: >80% → Achieved: {test_final['env_acc']:.3f}")
print(f"✨ Target MCM Loss: <0.02 → Achieved: {test_final['mcm_loss']:.4f}")
print(f"✨ Target Synthesis AUC: >0.73 → Achieved: {test_auc:.4f}")
print("✨ Dense Geometric Representations Ready!")
print("=" * 80) 