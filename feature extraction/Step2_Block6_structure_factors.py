#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🌌 BLOCK 6: STRUCTURE FACTORS IN RECIPROCAL SPACE")
print("Most Information-Rich Block: Diffraction Patterns → Dense Features")
print("Input: (92, 63) Structure Factors → Output: h₆ ∈ ℝ⁷⁶⁸")
print("=" * 80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

class StructureFactorDataset(Dataset):
    def __init__(self, structure_factors, y_synth=None, augment=True):
        """
        Dataset for Structure Factor Self-Supervised Learning
        
        Args:
            structure_factors: (N, 92, 63) structure factors in reciprocal space
            y_synth: Synthesis labels (for final evaluation only)
            augment: Whether to apply data augmentation
        """
        self.structure_factors = torch.FloatTensor(structure_factors)
        self.y_synth = torch.LongTensor(y_synth) if y_synth is not None else None
        self.augment = augment
        
        print(f"🔬 Dataset created: {len(self.structure_factors)} samples")
        
        # Analyze structure factor patterns
        self._analyze_structure_factors()
        
    def _analyze_structure_factors(self):
        """Analyze structure factor patterns for better understanding"""
        sf_flat = self.structure_factors.reshape(-1)
        
        # Find non-zero structure factors
        non_zero_mask = torch.abs(sf_flat) > 1e-6
        active_sf = sf_flat[non_zero_mask]
        
        # Analyze real-space vs reciprocal-space
        real_space = self.structure_factors[:, :, 0:4]  # Columns 0-3
        reciprocal_space = self.structure_factors[:, :, 4:63]  # Columns 4-62
        
        real_active = torch.abs(real_space) > 1e-6
        recip_active = torch.abs(reciprocal_space) > 1e-6
        
        self.sf_stats = {
            'total_values': len(sf_flat),
            'active_values': len(active_sf),
            'sparsity_pct': (1 - len(active_sf) / len(sf_flat)) * 100,
            'sf_range': [active_sf.min().item(), active_sf.max().item()] if len(active_sf) > 0 else [0, 0],
            'sf_mean': active_sf.mean().item() if len(active_sf) > 0 else 0,
            'sf_std': active_sf.std().item() if len(active_sf) > 0 else 0,
            'real_space_active_pct': real_active.float().mean().item() * 100,
            'reciprocal_active_pct': recip_active.float().mean().item() * 100
        }
        
        print(f"📊 Structure Factor Analysis:")
        print(f"   Total SF values: {self.sf_stats['total_values']:,}")
        print(f"   Active SF values: {self.sf_stats['active_values']:,}")
        print(f"   Sparsity: {self.sf_stats['sparsity_pct']:.2f}%")
        print(f"   SF range: [{self.sf_stats['sf_range'][0]:.4f}, {self.sf_stats['sf_range'][1]:.4f}]")
        print(f"   Mean SF: {self.sf_stats['sf_mean']:.4f} ± {self.sf_stats['sf_std']:.4f}")
        print(f"   Real-space active: {self.sf_stats['real_space_active_pct']:.1f}%")
        print(f"   Reciprocal-space active: {self.sf_stats['reciprocal_active_pct']:.1f}%")
        
    def __len__(self):
        return len(self.structure_factors)
    
    def __getitem__(self, idx):
        sf_matrix = self.structure_factors[idx]  # (92, 63)
        
        # Apply augmentation during training
        if self.augment:
            sf_matrix = self._augment_structure_factors(sf_matrix)
            
        # Create self-supervised tasks
        kp_interp_data = self._create_kpoint_interpolation_task(sf_matrix)
        real_recip_data = self._create_real_reciprocal_consistency_task(sf_matrix)
        element_recon_data = self._create_element_aware_reconstruction_task(sf_matrix)
        
        return {
            'structure_factors': sf_matrix,
            'kp_interp': kp_interp_data,
            'real_recip': real_recip_data,
            'element_recon': element_recon_data,
            'synth_label': self.y_synth[idx] if self.y_synth is not None else -1
        }
    
    def _augment_structure_factors(self, sf_matrix):
        """Apply structure factor augmentation"""
        # Small multiplicative noise for non-zero values
        if torch.rand(1) < 0.3:
            noise_factor = 1.0 + torch.randn_like(sf_matrix) * 0.01  # ±1% noise
            non_zero_mask = torch.abs(sf_matrix) > 1e-6
            sf_matrix[non_zero_mask] = sf_matrix[non_zero_mask] * noise_factor[non_zero_mask]
            
        # Phase rotation for complex structure factors
        if torch.rand(1) < 0.2:
            # Simulate phase shift (assuming magnitude + phase representation)
            phase_shift = torch.randn(1) * 0.1
            sf_matrix = sf_matrix * torch.cos(phase_shift) + sf_matrix * torch.sin(phase_shift) * 0.1
            
        return sf_matrix
    
    def _create_kpoint_interpolation_task(self, sf_matrix):
        """Create K-point Interpolation task"""
        # Mask 15% of k-points (reciprocal space columns 4-62)
        n_k_points = 59  # Columns 4-62
        n_mask = max(1, int(0.15 * n_k_points))
        
        # Select random k-points to mask
        mask_k_indices = torch.randperm(n_k_points)[:n_mask] + 4  # Add 4 to get actual column indices
        
        # Create masked matrix
        masked_sf = sf_matrix.clone()
        original_values = sf_matrix[:, mask_k_indices].clone()  # (92, n_mask)
        masked_sf[:, mask_k_indices] = 0.0
        
        return {
            'masked_sf': masked_sf,
            'mask_k_indices': mask_k_indices,
            'target_values': original_values
        }
    
    def _create_real_reciprocal_consistency_task(self, sf_matrix):
        """Create Real↔Reciprocal Consistency task"""
        real_space = sf_matrix[:, 0:4]  # (92, 4)
        reciprocal_space = sf_matrix[:, 4:63]  # (92, 59)
        
        # Summary statistics for consistency learning
        real_stats = torch.stack([
            real_space.mean(dim=1),
            real_space.std(dim=1),
            real_space.max(dim=1)[0],
            real_space.min(dim=1)[0]
        ], dim=1)  # (92, 4)
        
        recip_stats = torch.stack([
            reciprocal_space.mean(dim=1),
            reciprocal_space.std(dim=1),
            reciprocal_space.max(dim=1)[0],
            reciprocal_space.min(dim=1)[0]
        ], dim=1)  # (92, 4)
        
        return {
            'real_space': real_space,
            'reciprocal_space': reciprocal_space,
            'real_stats': real_stats,
            'recip_stats': recip_stats
        }
    
    def _create_element_aware_reconstruction_task(self, sf_matrix):
        """Create Element-Aware Reconstruction task"""
        # Determine number of active elements based on real-space columns
        real_space = sf_matrix[:, 0:4]
        element_activity = torch.abs(real_space).sum(dim=0) > 1e-6  # (4,)
        n_active_elements = element_activity.sum().item()
        
        # Create element mask for reconstruction
        active_columns = torch.zeros(63, dtype=torch.bool)
        active_columns[0:n_active_elements] = True  # Real-space active columns
        active_columns[4:63] = True  # All reciprocal-space columns
        
        # Mask inactive element columns
        masked_sf = sf_matrix.clone()
        masked_sf[:, ~active_columns] = 0.0
        
        return {
            'masked_sf': masked_sf,
            'active_columns': active_columns,
            'n_active_elements': n_active_elements,
            'element_activity': element_activity
        }

class Block6StructureFactorEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Element adapters for variable element counts (2-4 elements)
        self.element_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * 92, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256)
            ),  # 2 elements
            nn.Sequential(
                nn.Linear(3 * 92, 256),
                nn.ReLU(), 
                nn.Dropout(0.1),
                nn.Linear(256, 256)
            ),  # 3 elements
            nn.Sequential(
                nn.Linear(4 * 92, 256),
                nn.ReLU(),
                nn.Dropout(0.1), 
                nn.Linear(256, 256)
            ),  # 4 elements
        ])
        
        # 2D CNN for reciprocal space (treat as image: 92 properties × 59 k-points)
        self.reciprocal_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Adaptive pooling to handle variable k-points
            nn.AdaptiveAvgPool2d((8, 8))  # Output: (128, 8, 8)
        )
        
        # Cross-space attention for real↔reciprocal interactions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True, dropout=0.1
        )
        self.cross_norm1 = nn.LayerNorm(256)
        self.cross_norm2 = nn.LayerNorm(256)
        
        # Feed-forward for cross-attention
        self.cross_ff = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Task-specific heads for self-supervised learning
        self.kp_interp_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 92)  # Predict structure factors for one k-point
        )
        
        self.real_recip_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # Predict statistics
        )
        
        self.element_recon_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 92 * 4)  # Reconstruct real-space
        )
        
        # Final projection heads for 3×256D output
        self.real_space_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )
        
        self.reciprocal_proj = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.interaction_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )
        
    def forward(self, structure_factors, return_features=False):
        batch_size = structure_factors.shape[0]
        
        # Split into real-space and reciprocal-space components
        real_space = structure_factors[:, :, 0:4]  # (batch, 92, 4)
        reciprocal_space = structure_factors[:, :, 4:63]  # (batch, 92, 59)
        
        # Process real-space with element adapters
        # Detect number of active elements
        element_activity = torch.abs(real_space).sum(dim=1) > 1e-6  # (batch, 4)
        n_elements = element_activity.sum(dim=1)  # (batch,)
        
        real_features = []
        for i in range(batch_size):
            n_elem = n_elements[i].item()
            if n_elem >= 2 and n_elem <= 4:
                adapter_idx = n_elem - 2  # 0, 1, or 2
                active_real = real_space[i, :, :n_elem].flatten()  # (92*n_elem,)
                # Pad to expected size if needed
                if n_elem == 2:
                    padded = torch.zeros(2 * 92, device=structure_factors.device)
                    padded[:len(active_real)] = active_real
                elif n_elem == 3:
                    padded = torch.zeros(3 * 92, device=structure_factors.device)
                    padded[:len(active_real)] = active_real
                else:  # n_elem == 4
                    padded = active_real
                
                real_feat = self.element_adapters[adapter_idx](padded)
            else:
                # Fallback: use 4-element adapter with padding
                padded = torch.zeros(4 * 92, device=structure_factors.device)
                active_real = real_space[i].flatten()
                padded[:len(active_real)] = active_real
                real_feat = self.element_adapters[2](padded)
            
            real_features.append(real_feat)
        
        real_features = torch.stack(real_features)  # (batch, 256)
        
        # Process reciprocal-space with 2D CNN
        reciprocal_input = reciprocal_space.unsqueeze(1)  # (batch, 1, 92, 59)
        reciprocal_cnn_out = self.reciprocal_cnn(reciprocal_input)  # (batch, 128, 8, 8)
        reciprocal_features = reciprocal_cnn_out.view(batch_size, -1)  # (batch, 128*8*8)
        
        # Cross-space attention for interactions
        real_expanded = real_features.unsqueeze(1)  # (batch, 1, 256)
        recip_for_attention = reciprocal_cnn_out.mean(dim=(2, 3)).unsqueeze(1)  # (batch, 1, 128)
        
        # Pad reciprocal features to match real features dimension
        recip_padded = F.pad(recip_for_attention, (0, 128))  # (batch, 1, 256)
        
        # Cross-attention
        attn_out, _ = self.cross_attention(real_expanded, recip_padded, recip_padded)
        interaction_features = self.cross_norm1(real_expanded + attn_out)
        
        # Feed-forward
        ff_out = self.cross_ff(interaction_features)
        interaction_features = self.cross_norm2(interaction_features + ff_out)
        interaction_features = interaction_features.squeeze(1)  # (batch, 256)
        
        if return_features:
            # Generate final 768D features (3×256D)
            real_final = self.real_space_proj(real_features)
            reciprocal_final = self.reciprocal_proj(reciprocal_features)
            interaction_final = self.interaction_proj(interaction_features)
            
            return torch.cat([real_final, reciprocal_final, interaction_final], dim=1)
        
        # Self-supervised task predictions
        kp_interp_pred = self.kp_interp_head(interaction_features)
        real_recip_pred = self.real_recip_head(interaction_features)
        element_recon_pred = self.element_recon_head(real_features).view(batch_size, 92, 4)
        
        # Generate final features for training
        real_final = self.real_space_proj(real_features)
        reciprocal_final = self.reciprocal_proj(reciprocal_features)
        interaction_final = self.interaction_proj(interaction_features)
        final_features = torch.cat([real_final, reciprocal_final, interaction_final], dim=1)
        
        return {
            'kp_interp_pred': kp_interp_pred,
            'real_recip_pred': real_recip_pred,
            'element_recon_pred': element_recon_pred,
            'features': final_features
        }

def custom_collate(batch):
    """Custom collate function for batch processing"""
    structure_factors = torch.stack([item['structure_factors'] for item in batch])
    
    # Collate task data
    kp_interp_data = {
        'masked_sf': torch.stack([item['kp_interp']['masked_sf'] for item in batch]),
        'mask_k_indices': [item['kp_interp']['mask_k_indices'] for item in batch],
        'target_values': [item['kp_interp']['target_values'] for item in batch]
    }
    
    real_recip_data = {
        'real_stats': torch.stack([item['real_recip']['real_stats'] for item in batch]),
        'recip_stats': torch.stack([item['real_recip']['recip_stats'] for item in batch])
    }
    
    element_recon_data = {
        'active_columns': torch.stack([item['element_recon']['active_columns'] for item in batch]),
        'n_active_elements': torch.tensor([item['element_recon']['n_active_elements'] for item in batch])
    }
    
    synth_labels = torch.tensor([item['synth_label'] for item in batch])
    
    return {
        'structure_factors': structure_factors,
        'kp_interp': kp_interp_data,
        'real_recip': real_recip_data,
        'element_recon': element_recon_data,
        'synth_labels': synth_labels
    }

def compute_kp_interp_loss(kp_interp_pred, kp_interp_data):
    """Compute K-point Interpolation loss"""
    total_loss = 0.0
    count = 0
    
    for i, (mask_indices, target_values) in enumerate(zip(kp_interp_data['mask_k_indices'], kp_interp_data['target_values'])):
        if len(mask_indices) > 0:
            # Average prediction across masked k-points
            pred_avg = kp_interp_pred[i]  # (92,)
            target_avg = target_values.mean(dim=1).to(kp_interp_pred.device)  # (92,)
            
            loss = F.mse_loss(pred_avg, target_avg)
            total_loss += loss
            count += 1
            
    return total_loss / max(count, 1)

def compute_real_recip_loss(real_recip_pred, real_recip_data):
    """Compute Real↔Reciprocal Consistency loss"""
    real_stats = real_recip_data['real_stats'].to(real_recip_pred.device)
    
    # Predict reciprocal statistics from real-space
    target_stats = real_stats.mean(dim=1)  # Average across properties -> (batch, 4)
    
    return F.mse_loss(real_recip_pred, target_stats)

def compute_element_recon_loss(element_recon_pred, element_recon_data, structure_factors):
    """Compute Element-Aware Reconstruction loss"""
    real_space_target = structure_factors[:, :, 0:4].to(element_recon_pred.device)
    
    return F.mse_loss(element_recon_pred, real_space_target)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    kp_losses, rr_losses, er_losses = [], [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        structure_factors = batch['structure_factors'].to(device)
        
        # Forward pass
        outputs = model(structure_factors)
        
        # Compute individual losses
        kp_loss = compute_kp_interp_loss(outputs['kp_interp_pred'], batch['kp_interp'])
        rr_loss = compute_real_recip_loss(outputs['real_recip_pred'], batch['real_recip'])
        er_loss = compute_element_recon_loss(outputs['element_recon_pred'], batch['element_recon'], structure_factors)
        
        # Combined loss: 50% interpolation + 30% consistency + 20% reconstruction
        combined_loss = 0.5 * kp_loss + 0.3 * rr_loss + 0.2 * er_loss
        
        # Backward pass
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += combined_loss.item()
        kp_losses.append(kp_loss.item())
        rr_losses.append(rr_loss.item())
        er_losses.append(er_loss.item())
        
        pbar.set_postfix({
            'Loss': f'{combined_loss.item():.4f}',
            'KP': f'{kp_loss.item():.4f}',
            'RR': f'{rr_loss.item():.4f}',
            'ER': f'{er_loss.item():.4f}'
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        'kp_loss': np.mean(kp_losses),
        'rr_loss': np.mean(rr_losses),
        'er_loss': np.mean(er_losses)
    }

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    kp_losses, rr_losses, er_losses = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            structure_factors = batch['structure_factors'].to(device)
            
            # Forward pass
            outputs = model(structure_factors)
            
            # Compute individual losses
            kp_loss = compute_kp_interp_loss(outputs['kp_interp_pred'], batch['kp_interp'])
            rr_loss = compute_real_recip_loss(outputs['real_recip_pred'], batch['real_recip'])
            er_loss = compute_element_recon_loss(outputs['element_recon_pred'], batch['element_recon'], structure_factors)
            
            # Combined loss
            combined_loss = 0.5 * kp_loss + 0.3 * rr_loss + 0.2 * er_loss
            
            # Accumulate losses
            total_loss += combined_loss.item()
            kp_losses.append(kp_loss.item())
            rr_losses.append(rr_loss.item())
            er_losses.append(er_loss.item())
    
    return {
        'total_loss': total_loss / len(dataloader),
        'kp_loss': np.mean(kp_losses),
        'rr_loss': np.mean(rr_losses),
        'er_loss': np.mean(er_losses)
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

# Extract structure factors: FTCP[306:398, 0:63] -> (92, 63)
print("\n🌌 Extracting structure factors (Rows 306-397, Cols 0-62)...")
sf_train = X_train[:, 306:398, 0:63]  # (N, 92, 63)
sf_val = X_val[:, 306:398, 0:63]
sf_test = X_test[:, 306:398, 0:63]

print(f"Structure factor shapes: Train {sf_train.shape}, Val {sf_val.shape}, Test {sf_test.shape}")

# Create datasets
train_dataset = StructureFactorDataset(sf_train, y_train, augment=True)
val_dataset = StructureFactorDataset(sf_val, y_val, augment=False)
test_dataset = StructureFactorDataset(sf_test, y_test, augment=False)

# Create dataloaders with element-count grouping for efficiency
batch_size = 32  # Smaller batch size due to large model
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         collate_fn=custom_collate, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       collate_fn=custom_collate, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=custom_collate, num_workers=2)

print(f"✅ Dataloaders created - Batch size: {batch_size}")

# Initialize model
model = Block6StructureFactorEncoder(output_dim=768).to(device)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

print(f"\n🏋️  Model Configuration:")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Output dimension: 768D (3×256D)")
print(f"   Architecture: 2D CNN + Element Adapters + Cross-Attention")
print(f"   Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)")

# Training targets
print(f"\n🎯 Self-Supervised Learning Targets:")
print(f"   K-point Interpolation MSE: < 0.02")
print(f"   Real↔Reciprocal R²: > 0.85")
print(f"   Element-aware Accuracy: > 95%")
print(f"   Synthesis prediction (individual): AUC > 0.74")

# Training loop
num_epochs = 20
best_val_loss = float('inf')
train_history = []
val_history = []

print(f"\n🚀 Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    
    # Validate
    val_metrics = validate_epoch(model, val_loader, device)
    
    # Update scheduler
    scheduler.step()
    
    # Store metrics
    train_history.append(train_metrics)
    val_history.append(val_metrics)
    
    # Print metrics
    print(f"Train - Loss: {train_metrics['total_loss']:.4f}, KP: {train_metrics['kp_loss']:.4f}, "
          f"RR: {train_metrics['rr_loss']:.4f}, ER: {train_metrics['er_loss']:.4f}")
    print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, KP: {val_metrics['kp_loss']:.4f}, "
          f"RR: {val_metrics['rr_loss']:.4f}, ER: {val_metrics['er_loss']:.4f}")
    
    # Save best model
    if val_metrics['total_loss'] < best_val_loss:
        best_val_loss = val_metrics['total_loss']
        torch.save(model.state_dict(), 'Step2_Block6_structure_factors_best.pth')
        print(f"✅ New best model saved (val_loss: {best_val_loss:.4f})")

print(f"\n🎉 Training completed!")

# Load best model for evaluation
model.load_state_dict(torch.load('Step2_Block6_structure_factors_best.pth'))

# Extract features
print(f"\n🔍 Extracting features from best model...")

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            structure_factors = batch['structure_factors'].to(device)
            features = model(structure_factors, return_features=True)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(batch['synth_labels'].numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)

h6_train, y_train_used = extract_features(model, train_loader, device)
h6_val, y_val_used = extract_features(model, val_loader, device)
h6_test, y_test_used = extract_features(model, test_loader, device)

print(f"✅ Features extracted:")
print(f"   Train: {h6_train.shape}")
print(f"   Val: {h6_val.shape}")
print(f"   Test: {h6_test.shape}")

# Save features
np.save('Step2_Block6_features_train.npy', h6_train)
np.save('Step2_Block6_features_val.npy', h6_val)
np.save('Step2_Block6_features_test.npy', h6_test)
print(f"💾 Features saved to .npy files")

# Evaluate synthesis prediction
print(f"\n📊 Evaluating synthesis prediction performance...")

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(h6_train, y_train_used)

y_train_pred = clf.predict_proba(h6_train)[:, 1]
y_val_pred = clf.predict_proba(h6_val)[:, 1]
y_test_pred = clf.predict_proba(h6_test)[:, 1]

auc_train = roc_auc_score(y_train_used, y_train_pred)
auc_val = roc_auc_score(y_val_used, y_val_pred)
auc_test = roc_auc_score(y_test_used, y_test_pred)

print(f"\n🎯 SYNTHESIS PREDICTION RESULTS:")
print(f"   Train AUC: {auc_train:.4f}")
print(f"   Val AUC: {auc_val:.4f}")
print(f"   Test AUC: {auc_test:.4f}")
print(f"   Target: >0.74 | Gap: {auc_test - 0.74:.3f}")

# Evaluate self-supervised tasks
print(f"\n📈 SELF-SUPERVISED LEARNING EVALUATION:")

final_train = train_history[-1]
final_val = val_history[-1]

print(f"\n✅ FINAL PERFORMANCE:")
print(f"   K-point Interpolation: {final_val['kp_loss']:.6f} MSE (Target: <0.02)")
print(f"   Real↔Reciprocal: {final_val['rr_loss']:.6f} MSE (Target: <0.15)")
print(f"   Element Reconstruction: {final_val['er_loss']:.6f} MSE (Target: <0.05)")

kp_status = "✅" if final_val['kp_loss'] < 0.02 else "⚠️"
rr_status = "✅" if final_val['rr_loss'] < 0.15 else "⚠️"
er_status = "✅" if final_val['er_loss'] < 0.05 else "⚠️"
auc_status = "✅" if auc_test > 0.74 else "⚠️"

print(f"\n🎯 TARGET ACHIEVEMENT:")
print(f"   {kp_status} K-point Interpolation: {'ACHIEVED' if final_val['kp_loss'] < 0.02 else 'NEEDS IMPROVEMENT'}")
print(f"   {rr_status} Real↔Reciprocal: {'ACHIEVED' if final_val['rr_loss'] < 0.15 else 'NEEDS IMPROVEMENT'}")
print(f"   {er_status} Element Reconstruction: {'ACHIEVED' if final_val['er_loss'] < 0.05 else 'NEEDS IMPROVEMENT'}")
print(f"   {auc_status} Synthesis AUC: {'ACHIEVED' if auc_test > 0.74 else 'NEEDS IMPROVEMENT'}")

print(f"\n" + "=" * 80)
print(f"✅ BLOCK 6 STRUCTURE FACTORS PROCESSING COMPLETED!")
print(f"=" * 80)

print(f"""
🎯 SUMMARY:

SELF-SUPERVISED LEARNING PERFORMANCE:
├── K-point Interpolation: {final_val['kp_loss']:.6f} MSE {kp_status}
├── Real↔Reciprocal Consistency: {final_val['rr_loss']:.6f} MSE {rr_status}
├── Element Reconstruction: {final_val['er_loss']:.6f} MSE {er_status}
└── Combined: 3 tasks learning diffraction physics

SYNTHESIS PREDICTION:
├── Train AUC: {auc_train:.4f}
├── Val AUC: {auc_val:.4f}
├── Test AUC: {auc_test:.4f} {auc_status}
└── Individual performance: {'Above' if auc_test > 0.74 else 'Below'} target

FEATURES READY:
├── h₆ ∈ ℝ⁷⁶⁸ structure factor representations (RICHEST BLOCK!)
├── Components: 256D real-space + 256D reciprocal + 256D interactions
├── Samples: {len(h6_train):,} train, {len(h6_val):,} val, {len(h6_test):,} test
└── Multi-modal fusion ready: h₁+h₂+h₃+h₄+h₅+h₆ → 1,536D total!

NEXT STEP: Complete multi-block feature fusion for target AUC >0.73! 🚀
""") 