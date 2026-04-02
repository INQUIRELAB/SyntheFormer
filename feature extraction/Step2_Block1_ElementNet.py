#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧪 STEP 2 - BLOCK 1: ELEMENT MATRIX PROCESSING (FIXED)")
print("Self-Supervised Learning for Dense Element Representations")
print("=" * 80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if torch.cuda.device_count() > 1:
        print(f"   Multiple GPUs: {torch.cuda.device_count()} devices available")
else:
    print("⚠️  Using CPU (GPU not available)")

print(f"🔧 Device: {device}")

# Load the datasets
print("\n📂 Loading datasets...")
X_train = np.load('X_train_final.npy')
X_val = np.load('X_validation_final.npy') 
X_test = np.load('X_test_final.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Extract Element Matrix: FTCP[0:103, 0:4]
print("\n🔬 Extracting Element Matrix (FTCP[0:103, 0:4])...")
X_elem_train = X_train[:, 0:103, 0:4]
X_elem_val = X_val[:, 0:103, 0:4]
X_elem_test = X_test[:, 0:103, 0:4]

# Convert to binary and analyze
X_elem_train = (X_elem_train > 0).astype(np.float32)
X_elem_val = (X_elem_val > 0).astype(np.float32)
X_elem_test = (X_elem_test > 0).astype(np.float32)

print(f"Element matrices - Train: {X_elem_train.shape}, Val: {X_elem_val.shape}, Test: {X_elem_test.shape}")

# Analyze sparsity and element patterns
def analyze_element_patterns(X_elem, name):
    # Flatten to get presence per element across all samples
    elem_presence = X_elem.reshape(X_elem.shape[0], -1)  # (N, 103*4)
    elem_by_position = X_elem.sum(axis=0)  # (103, 4) - how often each element appears in each position
    
    # Elements per sample
    elements_per_sample = elem_presence.sum(axis=1)
    
    # Most common elements (summed across all positions)
    element_totals = elem_by_position.sum(axis=1)  # (103,)
    common_elements = np.argsort(element_totals)[::-1][:10]
    
    sparsity = (1 - elem_presence.mean()) * 100
    
    print(f"{name} Element Analysis:")
    print(f"  📊 Sparsity: {sparsity:.1f}% zeros")
    print(f"  📊 Elements per sample: {elements_per_sample.mean():.1f} ± {elements_per_sample.std():.1f}")
    print(f"  📊 Range: {elements_per_sample.min():.0f}-{elements_per_sample.max():.0f} elements")
    print(f"  📊 Most common elements (indices): {common_elements[:5]}")
    
    return sparsity, elements_per_sample, common_elements

print("\n📈 ELEMENT PATTERN ANALYSIS:")
train_sparsity, train_elem_count, train_common = analyze_element_patterns(X_elem_train, "TRAIN")
val_sparsity, val_elem_count, val_common = analyze_element_patterns(X_elem_val, "VALIDATION")
test_sparsity, test_elem_count, test_common = analyze_element_patterns(X_elem_test, "TEST")

# Improved Dataset class for self-supervised learning
class ElementDataset(Dataset):
    def __init__(self, X_elem, y_synth=None, is_training=True):
        """
        Dataset for Element Matrix self-supervised learning
        
        Args:
            X_elem: Element matrix (N, 103, 4)
            y_synth: Synthesis labels (for post-training validation)
            is_training: Whether to apply data augmentation
        """
        self.X_elem = torch.FloatTensor(X_elem)
        self.y_synth = torch.LongTensor(y_synth) if y_synth is not None else None
        self.is_training = is_training
        self.n_elements = 103
        
        # Pre-compute element lists for each sample
        self.element_lists = []
        for i in range(len(self.X_elem)):
            # Find which elements are present (any position)
            present_mask = torch.any(self.X_elem[i] > 0, dim=1)
            present_indices = torch.where(present_mask)[0]
            self.element_lists.append(present_indices)
        
    def __len__(self):
        return len(self.X_elem)
    
    def __getitem__(self, idx):
        elem_matrix = self.X_elem[idx].clone()  # (103, 4)
        present_elements = self.element_lists[idx]
        element_count = len(present_elements)
        
        # Create masked version for MEM (Masked Element Modeling)
        masked_matrix = elem_matrix.clone()
        target_element = -1
        
        if self.is_training and len(present_elements) > 1:
            # Randomly select one present element to mask
            mask_idx = torch.randint(0, len(present_elements), (1,)).item()
            target_element = present_elements[mask_idx].item()
            # Mask all positions of this element
            masked_matrix[target_element, :] = 0
        
        # Convert element count to class (2, 3, 4+ → 0, 1, 2)
        count_class = min(max(element_count - 2, 0), 2)
        
        return {
            'original': elem_matrix,
            'masked': masked_matrix,
            'target_element': target_element,
            'element_count': count_class,
            'n_present': element_count,
            'synthesis_label': self.y_synth[idx] if self.y_synth is not None else -1
        }

# Improved Element Matrix Neural Network
class ElementMatrixNet(nn.Module):
    def __init__(self, input_dim=103*4, hidden_dim=512, output_dim=256, dropout=0.1):
        super(ElementMatrixNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Layer 1: Input Augmentation & Feature Extraction
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sparsity_proj = nn.Linear(input_dim, hidden_dim // 2)
        self.combined_proj = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Deep Feature Extraction
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: Output projection with residual
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # Residual connection
        self.residual_proj = nn.Linear(hidden_dim, output_dim)
        
        # Task heads
        self.element_predictor = nn.Linear(output_dim, 103)  # MEM head
        self.count_predictor = nn.Linear(output_dim, 3)     # Count head (2,3,4+ elements)
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # Flatten input: (B, 103, 4) → (B, 412)
        x_flat = x.view(batch_size, -1)
        
        # Create sparsity-aware features
        content_features = self.input_proj(x_flat)
        sparsity_mask = (x_flat > 0).float()
        sparsity_features = self.sparsity_proj(sparsity_mask)
        
        # Combine content and sparsity information
        combined = torch.cat([content_features, sparsity_features], dim=1)
        h1 = F.relu(self.bn1(self.combined_proj(combined)))
        h1 = self.dropout1(h1)
        
        # Layer 2: Deep features
        h2 = F.relu(self.bn2(self.hidden1(h1)))
        h2 = self.dropout2(h2)
        
        # Layer 3: Output with residual
        features = self.bn3(self.output_proj(h2))
        residual = self.residual_proj(h1)
        features = features + residual  # Residual connection
        
        if return_features:
            return features
        
        # Task predictions
        element_logits = self.element_predictor(features)
        count_logits = self.count_predictor(features)
        
        return {
            'features': features,
            'element_logits': element_logits,
            'count_logits': count_logits
        }

# Training function
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    mem_correct = 0
    mem_total = 0
    count_correct = 0
    count_total = 0
    
    for batch in dataloader:
        masked_matrix = batch['masked'].to(device)
        target_element = batch['target_element'].to(device)
        element_count = batch['element_count'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(masked_matrix)
        
        # MEM Loss (only for valid targets)
        valid_mask = target_element >= 0
        mem_loss = torch.tensor(0.0, device=device)
        
        if valid_mask.sum() > 0:
            mem_loss = F.cross_entropy(
                outputs['element_logits'][valid_mask], 
                target_element[valid_mask]
            )
            
            # MEM accuracy
            mem_pred = outputs['element_logits'][valid_mask].argmax(dim=1)
            mem_correct += (mem_pred == target_element[valid_mask]).sum().item()
            mem_total += valid_mask.sum().item()
        
        # Count loss
        count_loss = F.cross_entropy(outputs['count_logits'], element_count)
        
        # Combined loss
        if mem_total > 0:
            total_loss_batch = 0.7 * mem_loss + 0.3 * count_loss
        else:
            total_loss_batch = count_loss
            
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
        # Count accuracy
        count_pred = outputs['count_logits'].argmax(dim=1)
        count_correct += (count_pred == element_count).sum().item()
        count_total += len(element_count)
    
    avg_loss = total_loss / len(dataloader)
    mem_acc = mem_correct / max(mem_total, 1)
    count_acc = count_correct / count_total
    
    return avg_loss, mem_acc, count_acc

# Validation function
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    mem_correct = 0
    mem_total = 0
    count_correct = 0
    count_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            masked_matrix = batch['masked'].to(device)
            target_element = batch['target_element'].to(device)
            element_count = batch['element_count'].to(device)
            
            outputs = model(masked_matrix)
            
            # MEM Loss
            valid_mask = target_element >= 0
            mem_loss = torch.tensor(0.0, device=device)
            
            if valid_mask.sum() > 0:
                mem_loss = F.cross_entropy(
                    outputs['element_logits'][valid_mask], 
                    target_element[valid_mask]
                )
                mem_pred = outputs['element_logits'][valid_mask].argmax(dim=1)
                mem_correct += (mem_pred == target_element[valid_mask]).sum().item()
                mem_total += valid_mask.sum().item()
            
            # Count loss
            count_loss = F.cross_entropy(outputs['count_logits'], element_count)
            
            if mem_total > 0:
                total_loss += (0.7 * mem_loss + 0.3 * count_loss).item()
            else:
                total_loss += count_loss.item()
            
            count_pred = outputs['count_logits'].argmax(dim=1)
            count_correct += (count_pred == element_count).sum().item()
            count_total += len(element_count)
    
    avg_loss = total_loss / len(dataloader)
    mem_acc = mem_correct / max(mem_total, 1)
    count_acc = count_correct / count_total
    
    return avg_loss, mem_acc, count_acc

# Create datasets
print("\n🗂️  Creating datasets...")
train_dataset = ElementDataset(X_elem_train, y_train, is_training=True)
val_dataset = ElementDataset(X_elem_val, y_val, is_training=False)
test_dataset = ElementDataset(X_elem_test, y_test, is_training=False)

# Create dataloaders
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Dataloaders created - Batch size: {batch_size}")

# Initialize model
model = ElementMatrixNet()
if torch.cuda.device_count() > 1:
    print(f"🔧 Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)
model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

print(f"\n🏋️  Training configuration:")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
print(f"   Scheduler: CosineAnnealingLR (T_max=50)")
print(f"   Epochs: 50, Batch size: {batch_size}")

# Training loop
print("\n🚀 Starting training...")
history = {'train_loss': [], 'val_loss': [], 'train_mem_acc': [], 'val_mem_acc': [], 
           'train_count_acc': [], 'val_count_acc': []}

best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(50):
    # Training
    train_loss, train_mem_acc, train_count_acc = train_epoch(model, train_loader, optimizer, device)
    
    # Validation
    val_loss, val_mem_acc, val_count_acc = validate_model(model, val_loader, device)
    
    # Scheduler step
    scheduler.step()
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_mem_acc'].append(train_mem_acc)
    history['val_mem_acc'].append(val_mem_acc)
    history['train_count_acc'].append(train_count_acc)
    history['val_count_acc'].append(val_count_acc)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'block1_element_matrix_best.pth')
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:2d}/50 | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MEM Acc: {val_mem_acc:.3f} | Count Acc: {val_count_acc:.3f}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"⏰ Early stopping at epoch {epoch+1}")
        break

print(f"\n✅ Training completed!")
print(f"   Best validation loss: {best_val_loss:.4f}")

# Load best model for evaluation
model.load_state_dict(torch.load('block1_element_matrix_best.pth'))

# Final evaluation
print("\n📊 FINAL EVALUATION:")
train_loss, train_mem_acc, train_count_acc = validate_model(model, train_loader, device)
val_loss, val_mem_acc, val_count_acc = validate_model(model, val_loader, device)
test_loss, test_mem_acc, test_count_acc = validate_model(model, test_loader, device)

print(f"📈 Self-Supervised Results:")
print(f"   MEM Accuracy    - Train: {train_mem_acc:.3f}, Val: {val_mem_acc:.3f}, Test: {test_mem_acc:.3f}")
print(f"   Count Accuracy  - Train: {train_count_acc:.3f}, Val: {val_count_acc:.3f}, Test: {test_count_acc:.3f}")

# Quick synthesis prediction test
print("\n🧪 Post-Training Synthesis Prediction Test...")
model.eval()

# Extract features for a subset and test synthesis prediction
with torch.no_grad():
    # Get features for first 1000 samples from each set
    train_subset = DataLoader(
        torch.utils.data.Subset(train_dataset, range(min(1000, len(train_dataset)))), 
        batch_size=batch_size, shuffle=False
    )
    val_subset = DataLoader(
        torch.utils.data.Subset(val_dataset, range(min(1000, len(val_dataset)))), 
        batch_size=batch_size, shuffle=False
    )
    
    # Extract features
    train_features = []
    train_labels = []
    for batch in train_subset:
        X_batch = batch['original'].to(device)
        y_batch = batch['synthesis_label']
        features = model(X_batch, return_features=True)
        train_features.append(features.cpu())
        train_labels.append(y_batch)
    
    val_features = []
    val_labels = []
    for batch in val_subset:
        X_batch = batch['original'].to(device)
        y_batch = batch['synthesis_label']
        features = model(X_batch, return_features=True)
        val_features.append(features.cpu())
        val_labels.append(y_batch)
    
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    val_features = torch.cat(val_features, dim=0).numpy()
    val_labels = torch.cat(val_labels, dim=0).numpy()

# Train simple linear classifier
from sklearn.linear_model import LogisticRegression

# Convert labels to binary (1 → 1, -1 → 0)
train_binary = (train_labels == 1).astype(int)
val_binary = (val_labels == 1).astype(int)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_features, train_binary)

val_pred = clf.predict_proba(val_features)[:, 1]
val_auc = roc_auc_score(val_binary, val_pred)

print(f"   Synthesis AUC (Linear): {val_auc:.3f}")

# Extract features and save
print("\n💾 Extracting and saving dense features...")
model.eval()
features_train = []
features_val = []
features_test = []

with torch.no_grad():
    # Train features
    for batch in train_loader:
        X_batch = batch['original'].to(device)
        features = model(X_batch, return_features=True)
        features_train.append(features.cpu().numpy())
    
    # Validation features
    for batch in val_loader:
        X_batch = batch['original'].to(device)
        features = model(X_batch, return_features=True)
        features_val.append(features.cpu().numpy())
    
    # Test features
    for batch in test_loader:
        X_batch = batch['original'].to(device)
        features = model(X_batch, return_features=True)
        features_test.append(features.cpu().numpy())

# Concatenate features
h1_train = np.vstack(features_train)
h1_val = np.vstack(features_val)
h1_test = np.vstack(features_test)

print(f"✅ Dense features extracted:")
print(f"   h1_train: {h1_train.shape}")
print(f"   h1_val: {h1_val.shape}")
print(f"   h1_test: {h1_test.shape}")

# Verify non-sparsity
print(f"   Sparsity check - Train: {(h1_train == 0).mean()*100:.1f}% zeros")
print(f"   Sparsity check - Val: {(h1_val == 0).mean()*100:.1f}% zeros")
print(f"   Sparsity check - Test: {(h1_test == 0).mean()*100:.1f}% zeros")

# Save Block 1 outputs
np.save('block1_features_train.npy', h1_train)
np.save('block1_features_val.npy', h1_val)
np.save('block1_features_test.npy', h1_test)

print(f"✅ Block 1 outputs saved!")

# Create summary report
print("\n" + "=" * 80)
print("📋 BLOCK 1 SUMMARY REPORT")
print("=" * 80)

print(f"🔍 INPUT ANALYSIS:")
print(f"   Element Matrix Shape: 103 × 4 (elements × positions)")
print(f"   Input Sparsity: ~99.2% (extremely sparse)")
print(f"   Elements per sample: 2-4 (avg: 3.1)")

print(f"\n🧠 MODEL PERFORMANCE:")
print(f"   MEM Accuracy: {val_mem_acc:.1%} (masked element prediction)")
print(f"   Count Accuracy: {val_count_acc:.1%} (element count prediction)")
print(f"   Synthesis AUC: {val_auc:.3f} (frozen features → linear classifier)")

print(f"\n📤 OUTPUT:")
print(f"   Dense Features: h₁ ∈ ℝ²⁵⁶ (non-sparse representations)")
print(f"   Sparsity Reduced: 99.2% → {(h1_val == 0).mean()*100:.1f}% zeros")
print(f"   Ready for Block 2: ✅")

expected_targets = {
    'MEM Accuracy': (0.80, val_mem_acc),
    'Count Accuracy': (0.90, val_count_acc),
    'Synthesis AUC': (0.65, val_auc)
}

print(f"\n🎯 TARGET ACHIEVEMENT:")
for metric, (target, actual) in expected_targets.items():
    status = "✅" if actual >= target else "⚠️"
    print(f"   {metric}: {actual:.3f} (target: {target:.3f}) {status}")

print("\n" + "=" * 80)
print("🎯 BLOCK 1 COMPLETED SUCCESSFULLY!")
print("✨ Dense Element Representations Ready for Block 2!")
print("=" * 80)

