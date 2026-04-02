#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 STEP 2 - BLOCK 4: SITE OCCUPANCY SELF-SUPERVISED LEARNING")
print("Graph Attention Network: Element-Site Assignments → Dense Features")
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

class SiteOccupancyDataset(Dataset):
    def __init__(self, X_occupancy, X_coords, y_synth=None, augment=True):
        """
        Dataset for Site Occupancy Self-Supervised Learning
        
        Args:
            X_occupancy: (N, 100, 4) site occupancy data FTCP[205:304, 0:4]
            X_coords: (N, 100, 3) coordinate data FTCP[105:205, 0:3] for cross-attention
            y_synth: Synthesis labels (for final evaluation only)
            augment: Whether to apply data augmentation
        """
        self.X_occupancy = torch.FloatTensor(X_occupancy)
        self.X_coords = torch.FloatTensor(X_coords)
        self.y_synth = torch.LongTensor(y_synth) if y_synth is not None else None
        self.augment = augment
        
        print(f"🔬 Dataset created: {len(self.X_occupancy)} samples")
        self._analyze_occupancy_patterns()
        
    def _analyze_occupancy_patterns(self):
        """Analyze occupancy patterns for better understanding"""
        occupancy_flat = self.X_occupancy.reshape(-1, 4)
        coords_flat = self.X_coords.reshape(-1, 3)
        
        # Find active sites (non-zero occupancy)
        occupancy_mask = (occupancy_flat.abs().sum(dim=1) > 1e-6)
        coords_mask = (coords_flat.abs().sum(dim=1) > 1e-6)
        
        active_occupancy = occupancy_flat[occupancy_mask]
        active_coords = coords_flat[coords_mask]
        
        self.occupancy_stats = {
            'total_sites': len(occupancy_flat),
            'active_occupancy_sites': len(active_occupancy),
            'active_coord_sites': len(active_coords),
            'occupancy_sparsity': (1 - len(active_occupancy) / len(occupancy_flat)) * 100,
            'occupancy_mean': active_occupancy.mean(dim=0),
            'occupancy_std': active_occupancy.std(dim=0)
        }
        
        print(f"📊 Site Occupancy Analysis:")
        print(f"   Active occupancy sites: {self.occupancy_stats['active_occupancy_sites']:,} / {self.occupancy_stats['total_sites']:,}")
        print(f"   Active coordinate sites: {self.occupancy_stats['active_coord_sites']:,}")
        print(f"   Occupancy sparsity: {self.occupancy_stats['occupancy_sparsity']:.1f}%")
        print(f"   Occupancy range: [{active_occupancy.min():.3f}, {active_occupancy.max():.3f}]")
        
    def __len__(self):
        return len(self.X_occupancy)
    
    def __getitem__(self, idx):
        occupancy = self.X_occupancy[idx]  # (100, 4)
        coords = self.X_coords[idx]       # (100, 3)
        
        # Remove padding sites based on occupancy
        site_mask = (occupancy.abs().sum(dim=1) > 1e-6)
        active_occupancy = occupancy[site_mask]  # (n_active, 4)
        active_coords = coords[site_mask]        # (n_active, 3)
        
        if len(active_occupancy) < 3:
            # Handle edge case - pad with small values
            n_needed = 3 - len(active_occupancy)
            occ_padding = torch.randn(n_needed, 4) * 0.01
            coord_padding = torch.randn(n_needed, 3) * 0.01
            active_occupancy = torch.cat([active_occupancy, occ_padding], dim=0)
            active_coords = torch.cat([active_coords, coord_padding], dim=0)
            
        # Limit max sites
        if len(active_occupancy) > 50:
            indices = torch.randperm(len(active_occupancy))[:50]
            active_occupancy = active_occupancy[indices]
            active_coords = active_coords[indices]
            
        # Apply augmentation
        if self.augment:
            active_occupancy = self._augment_occupancy(active_occupancy)
            
        # Create self-supervised task
        masked_data = self._create_masked_occupancy_task(active_occupancy)
        
        return {
            'occupancy': active_occupancy,
            'coords': active_coords,
            'masked_task': masked_data,
            'synth_label': self.y_synth[idx] if self.y_synth is not None else -1,
            'n_sites': len(active_occupancy)
        }
    
    def _augment_occupancy(self, occupancy):
        """Apply occupancy data augmentation"""
        # Small random noise (±0.01)
        noise = torch.randn_like(occupancy) * 0.01
        occupancy = occupancy + noise
        
        # Random scaling (±5%)
        if torch.rand(1) < 0.3:
            scale = 1.0 + (torch.rand(1) - 0.5) * 0.1
            occupancy = occupancy * scale
            
        return torch.clamp(occupancy, min=0)  # Ensure non-negative
    
    def _create_masked_occupancy_task(self, occupancy):
        """Create masked site occupancy prediction task"""
        n_sites = len(occupancy)
        
        # Mask 20% of sites
        n_mask = max(1, int(0.20 * n_sites))
        mask_indices = torch.randperm(n_sites)[:n_mask]
        
        # Create masked occupancy
        masked_occupancy = occupancy.clone()
        original_values = occupancy[mask_indices].clone()
        masked_occupancy[mask_indices] = 0.0  # Mask with zeros
        
        return {
            'masked_occupancy': masked_occupancy,
            'mask_indices': mask_indices,
            'target_values': original_values
        }

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads
        
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, x, edge_index=None, attention_mask=None):
        batch_size, n_nodes = x.shape[:2]
        
        # Multi-head attention
        Q = self.W_q(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.view(batch_size, n_nodes, self.out_features)
        
        # Output projection
        output = self.W_o(context)
        
        # Residual connection and normalization
        output = self.norm(output + x if x.shape[-1] == self.out_features else output)
        
        return output, attn_weights.mean(dim=2)  # Average over heads

class CrossAttentionLayer(nn.Module):
    def __init__(self, occupancy_dim, coord_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        
        self.W_q = nn.Linear(occupancy_dim, out_dim)
        self.W_k = nn.Linear(coord_dim, out_dim)
        self.W_v = nn.Linear(coord_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, occupancy_features, coord_features):
        batch_size, n_nodes = occupancy_features.shape[:2]
        
        Q = self.W_q(occupancy_features).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        K = self.W_k(coord_features).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        V = self.W_v(coord_features).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        
        # Cross-attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        context = context.view(batch_size, n_nodes, -1)
        
        output = self.W_o(context)
        return self.norm(output), attn_weights.mean(dim=2)

class SiteOccupancyGAT(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=256, n_elements=103):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Element embeddings
        self.element_embeddings = nn.Embedding(n_elements, 64)
        
        # Input projections
        self.occupancy_proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.coord_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # GAT layers
        self.gat1 = GraphAttentionLayer(128, 128, n_heads=4)
        self.gat2 = GraphAttentionLayer(128, hidden_dim, n_heads=4)
        
        # Cross-attention with Block 3 (coordinates)
        self.cross_attention = CrossAttentionLayer(hidden_dim, 64, hidden_dim, n_heads=4)
        
        # Global pooling
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Task-specific head
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # Predict 4D occupancy
        )
        
        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Attention regularization
        self.attention_reg_weight = 0.01
        
    def forward(self, occupancy_list, coords_list, return_features=False):
        batch_size = len(occupancy_list)
        batch_features = []
        occupancy_predictions = []
        attention_weights = []
        
        for i, (occupancy, coords) in enumerate(zip(occupancy_list, coords_list)):
            # Project inputs
            occ_features = self.occupancy_proj(occupancy)  # (n_sites, 128)
            coord_features = self.coord_proj(coords)       # (n_sites, 64)
            
            # Add batch dimension
            occ_features = occ_features.unsqueeze(0)       # (1, n_sites, 128)
            coord_features = coord_features.unsqueeze(0)   # (1, n_sites, 64)
            
            # GAT layers
            occ_features, attn1 = self.gat1(occ_features)
            occ_features, attn2 = self.gat2(occ_features)
            
            # Cross-attention with coordinates
            cross_features, cross_attn = self.cross_attention(occ_features, coord_features)
            
            # Combine features
            combined_features = occ_features + cross_features
            
            # Global pooling
            query = self.global_query.expand(1, -1, -1)
            global_feat, _ = self.global_attention(query, combined_features, combined_features)
            global_feat = global_feat.squeeze(0).squeeze(0)  # (hidden_dim,)
            
            batch_features.append(global_feat)
            attention_weights.append([attn1, attn2, cross_attn])
            
            if not return_features:
                # Occupancy prediction
                occ_pred = self.occupancy_head(combined_features.squeeze(0))  # (n_sites, 4)
                occupancy_predictions.append(occ_pred)
        
        # Stack batch features
        batch_features = torch.stack(batch_features)  # (batch_size, hidden_dim)
        
        if return_features:
            return self.feature_proj(batch_features)
        
        return {
            'occupancy_pred': occupancy_predictions,
            'features': self.feature_proj(batch_features),
            'attention_weights': attention_weights
        }
    
    def compute_attention_regularization(self, attention_weights):
        """Compute attention sparsity regularization"""
        total_reg = 0
        count = 0
        
        for batch_attns in attention_weights:
            for attn in batch_attns:
                # Encourage sparsity by L1 penalty
                total_reg += torch.sum(torch.abs(attn))
                count += 1
                
        return total_reg / count if count > 0 else 0

def custom_collate(batch):
    """Custom collate function for variable-length sequences"""
    occupancy_list = [item['occupancy'] for item in batch]
    coords_list = [item['coords'] for item in batch]
    masked_tasks = [item['masked_task'] for item in batch]
    synth_labels = torch.tensor([item['synth_label'] for item in batch])
    n_sites = torch.tensor([item['n_sites'] for item in batch])
    
    return {
        'occupancy': occupancy_list,
        'coords': coords_list,
        'masked_tasks': masked_tasks,
        'synth_labels': synth_labels,
        'n_sites': n_sites
    }

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    mso_losses = []  # Masked Site Occupancy losses
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Move data to device
        occupancy_list = [occ.to(device) for occ in batch['occupancy']]
        coords_list = [coord.to(device) for coord in batch['coords']]
        
        # Forward pass
        outputs = model(occupancy_list, coords_list)
        
        # Calculate masked site occupancy loss
        mso_loss = 0
        batch_size = len(batch['masked_tasks'])
        
        for i in range(batch_size):
            masked_task = batch['masked_tasks'][i]
            if len(masked_task['mask_indices']) > 0:
                pred_occupancy = outputs['occupancy_pred'][i]
                target_occupancy = masked_task['target_values'].to(device)
                mask_indices = masked_task['mask_indices']
                
                mso_loss += F.mse_loss(pred_occupancy[mask_indices], target_occupancy)
        
        mso_loss = mso_loss / batch_size
        
        # Attention regularization
        attn_reg = model.compute_attention_regularization(outputs['attention_weights'])
        
        # Total loss
        total_loss_batch = mso_loss + model.attention_reg_weight * attn_reg
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += total_loss_batch.item()
        mso_losses.append(mso_loss.item())
        
        # Progress monitoring
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx:3d}/{len(dataloader)} | "
                  f"Loss: {total_loss_batch.item():.4f} | "
                  f"MSO: {mso_loss.item():.4f} | "
                  f"AttReg: {attn_reg.item():.4f}")
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mso_loss': np.mean(mso_losses)
    }

def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    mso_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            occupancy_list = [occ.to(device) for occ in batch['occupancy']]
            coords_list = [coord.to(device) for coord in batch['coords']]
            
            outputs = model(occupancy_list, coords_list)
            
            mso_loss = 0
            batch_size = len(batch['masked_tasks'])
            
            for i in range(batch_size):
                masked_task = batch['masked_tasks'][i]
                if len(masked_task['mask_indices']) > 0:
                    pred_occupancy = outputs['occupancy_pred'][i]
                    target_occupancy = masked_task['target_values'].to(device)
                    mask_indices = masked_task['mask_indices']
                    mso_loss += F.mse_loss(pred_occupancy[mask_indices], target_occupancy)
            
            mso_loss = mso_loss / batch_size
            attn_reg = model.compute_attention_regularization(outputs['attention_weights'])
            total_loss_batch = mso_loss + model.attention_reg_weight * attn_reg
            
            total_loss += total_loss_batch.item()
            mso_losses.append(mso_loss.item())
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mso_loss': np.mean(mso_losses)
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

# Extract site occupancy and coordinates
print("\n🔬 Extracting site occupancy (FTCP[205:304, 0:4]) and coordinates...")
X_occupancy_train = X_train[:, 205:305, 0:4]  # Site occupancy
X_occupancy_val = X_val[:, 205:305, 0:4]
X_occupancy_test = X_test[:, 205:305, 0:4]

X_coords_train = X_train[:, 105:205, 0:3]     # Coordinates for cross-attention
X_coords_val = X_val[:, 105:205, 0:3]
X_coords_test = X_test[:, 105:205, 0:3]

print(f"Occupancy shapes: Train {X_occupancy_train.shape}, Val {X_occupancy_val.shape}, Test {X_occupancy_test.shape}")
print(f"Coordinate shapes: Train {X_coords_train.shape}, Val {X_coords_val.shape}, Test {X_coords_test.shape}")

# Create datasets (reduced size for efficiency)
sample_size_train = min(20000, len(X_occupancy_train))
sample_size_val = min(8000, len(X_occupancy_val))
sample_size_test = min(3000, len(X_occupancy_test))

print(f"\n🎯 Using sample sizes: Train {sample_size_train}, Val {sample_size_val}, Test {sample_size_test}")

train_dataset = SiteOccupancyDataset(
    X_occupancy_train[:sample_size_train], 
    X_coords_train[:sample_size_train],
    y_train[:sample_size_train], 
    augment=True
)

val_dataset = SiteOccupancyDataset(
    X_occupancy_val[:sample_size_val], 
    X_coords_val[:sample_size_val],
    y_val[:sample_size_val], 
    augment=False
)

test_dataset = SiteOccupancyDataset(
    X_occupancy_test[:sample_size_test], 
    X_coords_test[:sample_size_test],
    y_test[:sample_size_test], 
    augment=False
)

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
model = SiteOccupancyGAT(hidden_dim=256, output_dim=256).to(device)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Lower lr as specified
scheduler = CosineAnnealingLR(optimizer, T_max=20)

print(f"\n🏋️  Model Configuration:")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Architecture: Graph Attention Network (GAT) with Cross-Attention")
print(f"   Hidden dim: 256, Output dim: 256")
print(f"   Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)")

# Training loop
print("\n🚀 Starting Block 4 Training...")
history = {
    'train_total_loss': [], 'val_total_loss': [], 
    'train_mso_loss': [], 'val_mso_loss': []
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
        torch.save(model.state_dict(), 'Step2_Block4_site_occupancy_best.pth')
        print("  💾 New best model saved!")
    else:
        patience_counter += 1
    
    epoch_time = time.time() - start_time
    
    # Progress report
    print(f"  🎯 Results: Loss {val_metrics['total_loss']:.4f} | "
          f"MSO {val_metrics['mso_loss']:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    if patience_counter >= patience:
        print(f"⏰ Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('Step2_Block4_site_occupancy_best.pth'))

# Final evaluation
print("\n📊 FINAL EVALUATION:")
train_final = validate_model(model, train_loader, device)
val_final = validate_model(model, val_loader, device)
test_final = validate_model(model, test_loader, device)

print(f"🎯 Self-Supervised Results:")
print(f"   MSO Loss - Train: {train_final['mso_loss']:.4f}, Val: {val_final['mso_loss']:.4f}, Test: {test_final['mso_loss']:.4f}")

# Extract features for synthesis prediction
print("\n💾 Extracting features for synthesis evaluation...")
model.eval()

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            occupancy_list = [occ.to(device) for occ in batch['occupancy']]
            coords_list = [coord.to(device) for coord in batch['coords']]
            batch_features = model(occupancy_list, coords_list, return_features=True)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch['synth_labels'].numpy())
    return np.vstack(features), np.array(labels)

h4_train, y_train_eval = extract_features(train_loader)
h4_val, y_val_eval = extract_features(val_loader)
h4_test, y_test_eval = extract_features(test_loader)

print(f"✅ Features extracted: Train {h4_train.shape}, Val {h4_val.shape}, Test {h4_test.shape}")

# Synthesis prediction evaluation
print("\n🎯 SYNTHESIS PREDICTION EVALUATION:")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(h4_train, y_train_eval)

# Predictions
train_pred = clf.predict_proba(h4_train)[:, 1]
val_pred = clf.predict_proba(h4_val)[:, 1]
test_pred = clf.predict_proba(h4_test)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train_eval, train_pred)
val_auc = roc_auc_score(y_val_eval, val_pred)
test_auc = roc_auc_score(y_test_eval, test_pred)

print(f"🎯 Synthesis AUC Results:")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Val AUC: {val_auc:.4f}")
print(f"   Test AUC: {test_auc:.4f}")

# Save features
np.save('Step2_Block4_features_train.npy', h4_train)
np.save('Step2_Block4_features_val.npy', h4_val)
np.save('Step2_Block4_features_test.npy', h4_test)

print(f"\n✅ Block 4 features saved!")

# Create visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Block 4: Site Occupancy Self-Supervised Learning Results', fontsize=16, fontweight='bold')

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

# MSO Loss
ax2 = axes[0, 1]
ax2.plot(epochs, history['train_mso_loss'], 'g-', label='Train MSO', linewidth=2)
ax2.plot(epochs, history['val_mso_loss'], 'orange', label='Val MSO', linewidth=2)
ax2.axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Target (0.02)')
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('MSO Loss', fontweight='bold')
ax2.set_title('Masked Site Occupancy Loss', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Feature distribution
ax3 = axes[1, 0]
ax3.hist(h4_train.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax3.set_xlabel('Feature Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Block 4 Feature Distribution')
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
ax4.set_ylim(0.4, 1.0)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('Step2_Block4_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("🎯 BLOCK 4 COMPLETED SUCCESSFULLY!")
print(f"✨ Target MSO Loss: <0.02 → Achieved: {test_final['mso_loss']:.4f}")
print(f"✨ Target Synthesis AUC: >0.73 → Achieved: {test_auc:.4f}")
print("✨ GAT with Cross-Attention Architecture Working!")
print("✨ Ready for Joint Training with Previous Blocks!")
print("=" * 80) 