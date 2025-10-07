#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏗️ STEP 2 - BLOCK 2 IMPROVED: LATTICE PARAMETERS → DENSE FEATURES")
print("Addressing Class Imbalance & Domain Shift Issues")
print("=" * 80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"   Using {torch.cuda.device_count()} GPUs")
print(f"🔧 Device: {device}")

# Load datasets
print("\n📂 Loading datasets...")
X_train = np.load('X_train_final.npy')
X_val = np.load('X_validation_final.npy') 
X_test = np.load('X_test_final.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

print(f"✅ Loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Extract Lattice Parameters
def extract_lattice_params(X):
    lattice_raw = X[:, 103:105, 0:4]
    lattice_6 = np.concatenate([
        lattice_raw[:, 0, :3],  # a, b, c
        lattice_raw[:, 1, :3]   # α, β, γ
    ], axis=1)
    return lattice_6.astype(np.float32)

# Enhanced Feature Engineering with normalization
def engineer_lattice_features(lattice_params):
    lattice_params = lattice_params.astype(np.float32)
    a, b, c, alpha, beta, gamma = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2], \
                                  lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]
    
    features = []
    
    # 1-6: Original parameters
    features.extend([a, b, c, alpha, beta, gamma])
    
    # 7-10: Length ratios (stabilized)
    eps = 1e-6
    features.append(np.minimum(a / (b + eps), 10.0))  # Clip extreme ratios
    features.append(np.minimum(b / (c + eps), 10.0))
    features.append(np.minimum(c / (a + eps), 10.0))
    features.append(np.minimum((a + b) / (c + eps), 20.0))
    
    # 11-14: Geometric features
    features.append(a * b)  # area-like
    features.append(a * b * c)  # volume-like
    features.append(np.sqrt(a * b))  # geometric mean of lengths
    features.append((a + b + c) / 3)  # average length
    
    # 15-18: Symmetry measures (key for crystal system classification)
    features.append(np.abs(a - b))  # length asymmetry 1
    features.append(np.abs(b - c))  # length asymmetry 2  
    features.append(np.abs(a - c))  # length asymmetry 3
    features.append(np.max([np.abs(a - b), np.abs(b - c), np.abs(a - c)], axis=0))  # max asymmetry
    
    # 19-22: Angle features
    features.append(np.abs(alpha - beta))  # angle asymmetry 1
    features.append(np.abs(beta - gamma))  # angle asymmetry 2
    features.append(np.abs(alpha - gamma))  # angle asymmetry 3  
    features.append(np.max([np.abs(alpha - beta), np.abs(beta - gamma), np.abs(alpha - gamma)], axis=0))  # max angle asymmetry
    
    # 23-24: Composite features
    length_var = np.var([a, b, c], axis=0)  # Length variance
    angle_var = np.var([alpha, beta, gamma], axis=0)  # Angle variance
    features.extend([length_var, angle_var])
    
    result = np.column_stack(features).astype(np.float32)
    
    # Normalize features to prevent outliers
    result = np.clip(result, -10, 10)
    
    return result

# More robust crystal system classification
def classify_crystal_system_robust(lattice_params):
    lattice_params = lattice_params.astype(np.float32)
    a, b, c, alpha, beta, gamma = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2], \
                                  lattice_params[:, 3], lattice_params[:, 4], lattice_params[:, 5]
    
    crystal_systems = []
    
    for i in range(len(lattice_params)):
        a_i, b_i, c_i = a[i], b[i], c[i]
        alpha_i, beta_i, gamma_i = alpha[i], beta[i], gamma[i]
        
        # Adaptive tolerances based on magnitudes
        length_tol = 0.08  # Increased tolerance
        angle_tol = 0.15   # Increased tolerance
        
        # Length relationships
        a_eq_b = abs(a_i - b_i) < length_tol
        b_eq_c = abs(b_i - c_i) < length_tol
        a_eq_c = abs(a_i - c_i) < length_tol
        
        # Angle relationships
        angles_eq = (abs(alpha_i - beta_i) < angle_tol and 
                    abs(beta_i - gamma_i) < angle_tol and 
                    abs(alpha_i - gamma_i) < angle_tol)
        
        # More nuanced classification
        length_symmetry = sum([a_eq_b, b_eq_c, a_eq_c])
        
        if length_symmetry >= 2:  # High length symmetry
            system = 0  # Cubic-like
        elif length_symmetry == 1:  # Partial length symmetry
            system = 1  # Tetragonal-like
        elif angles_eq:  # Regular angles but irregular lengths
            system = 2  # Regular angles
        else:  # No clear symmetry
            system = 3  # Irregular
            
        crystal_systems.append(system)
    
    return np.array(crystal_systems, dtype=np.int64)

# Process data with enhanced features
lattice_train = extract_lattice_params(X_train)
lattice_val = extract_lattice_params(X_val)
lattice_test = extract_lattice_params(X_test)

lattice_features_train = engineer_lattice_features(lattice_train)
lattice_features_val = engineer_lattice_features(lattice_val)
lattice_features_test = engineer_lattice_features(lattice_test)

crystal_systems_train = classify_crystal_system_robust(lattice_train)
crystal_systems_val = classify_crystal_system_robust(lattice_val)
crystal_systems_test = classify_crystal_system_robust(lattice_test)

print(f"Enhanced features shape: {lattice_features_train.shape}")

# Check distributions
print("\n📊 Crystal System Distributions:")
system_names = ['Cubic-like', 'Tetragonal-like', 'Regular angles', 'Irregular']
for split_name, systems in [('TRAIN', crystal_systems_train), ('VAL', crystal_systems_val), ('TEST', crystal_systems_test)]:
    counts = np.bincount(systems, minlength=4)
    total = len(systems)
    print(f"{split_name}: {[f'{name}:{counts[i]}({counts[i]/total*100:.1f}%)' for i, name in enumerate(system_names)]}")

# Enhanced Dataset with class balancing
class EnhancedLatticeDataset(Dataset):
    def __init__(self, lattice_params, lattice_features, crystal_systems, y_synth, augment=False):
        self.lattice_params = torch.FloatTensor(lattice_params.astype(np.float32))
        self.lattice_features = torch.FloatTensor(lattice_features.astype(np.float32))
        self.crystal_systems = torch.LongTensor(crystal_systems.astype(np.int64))
        self.y_synth = torch.LongTensor(y_synth.astype(np.int64))
        self.augment = augment
        
    def __len__(self):
        return len(self.lattice_params)
    
    def __getitem__(self, idx):
        params = self.lattice_params[idx]
        features = self.lattice_features[idx]
        
        # Data augmentation for training
        if self.augment and torch.rand(1) < 0.3:
            # Add small noise to features (helps with generalization)
            noise = torch.randn_like(features) * 0.01
            features = features + noise
            features = torch.clamp(features, -10, 10)
        
        # Create masked version for MPM
        masked_features = features.clone()
        mask_idx = torch.randint(0, 6, (1,)).item()
        target_value = params[mask_idx].clone()
        masked_features[mask_idx] = 0.0
        
        return {
            'original_features': features,
            'masked_features': masked_features,
            'target_param_idx': mask_idx,
            'target_param_value': target_value,
            'crystal_system': self.crystal_systems[idx],
            'synthesis_label': self.y_synth[idx]
        }

# Enhanced Neural Network with better regularization
class EnhancedLatticeNet(nn.Module):
    def __init__(self, input_dim=24, output_dim=256):
        super(EnhancedLatticeNet, self).__init__()
        
        # Input processing with layer normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.2)
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.hidden2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)
        
        # Residual connection
        self.residual = nn.Linear(64, 256)
        
        # Output projection
        self.output = nn.Linear(256, output_dim)
        self.bn_out = nn.BatchNorm1d(output_dim)
        
        # Task heads
        self.param_head = nn.Linear(output_dim, 1)
        self.system_head = nn.Linear(output_dim, 4)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_features=False):
        x = x.float()
        x = self.input_norm(x)
        
        # Input processing
        h1 = F.relu(self.input_proj(x))
        h1 = self.dropout1(h1)
        
        # Save for residual
        residual_input = h1
        
        # Hidden layer 1
        h2 = F.relu(self.bn1(self.hidden1(h1)))
        h2 = self.dropout2(h2)
        
        # Hidden layer 2
        h3 = F.relu(self.bn2(self.hidden2(h2)))
        h3 = self.dropout3(h3)
        
        # Add residual connection
        residual = self.residual(residual_input)
        h3 = h3 + residual
        
        # Output features
        features = self.bn_out(self.output(h3))
        
        if return_features:
            return features
        
        param_pred = self.param_head(features)
        system_pred = self.system_head(features)
        
        return {
            'features': features,
            'param_pred': param_pred,
            'system_pred': system_pred
        }

# Create weighted sampler for class balancing
def create_weighted_sampler(crystal_systems):
    class_counts = np.bincount(crystal_systems, minlength=4)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    
    sample_weights = class_weights[crystal_systems]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Create datasets
train_dataset = EnhancedLatticeDataset(lattice_train, lattice_features_train, crystal_systems_train, y_train, augment=True)
val_dataset = EnhancedLatticeDataset(lattice_val, lattice_features_val, crystal_systems_val, y_val, augment=False)
test_dataset = EnhancedLatticeDataset(lattice_test, lattice_features_test, crystal_systems_test, y_test, augment=False)

# Create weighted sampler for training
train_sampler = create_weighted_sampler(crystal_systems_train)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize enhanced model
model = EnhancedLatticeNet(input_dim=24)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=60)

print(f"✅ Enhanced model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class-weighted loss for crystal system classification
class_weights = torch.FloatTensor([1.0, 1.0, 3.0, 5.0]).to(device)  # Higher weights for minority classes

# Enhanced training function
def train_epoch_enhanced(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    param_mse = 0
    system_acc = 0
    n_samples = 0
    
    for batch in dataloader:
        masked_features = batch['masked_features'].to(device).float()
        target_param_value = batch['target_param_value'].to(device).float()
        crystal_system = batch['crystal_system'].to(device).long()
        
        optimizer.zero_grad()
        outputs = model(masked_features)
        
        # Parameter prediction loss
        param_loss = F.mse_loss(outputs['param_pred'].squeeze(), target_param_value)
        
        # Weighted crystal system classification loss
        system_loss = F.cross_entropy(outputs['system_pred'], crystal_system, weight=class_weights)
        
        # Balanced loss
        total_loss_batch = 0.6 * param_loss + 0.4 * system_loss
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += total_loss_batch.item()
        param_mse += param_loss.item()
        
        system_pred = outputs['system_pred'].argmax(dim=1)
        system_acc += (system_pred == crystal_system).float().mean().item()
        
        n_samples += 1
    
    return (total_loss / n_samples, param_mse / n_samples, system_acc / n_samples)

# Enhanced validation function
def validate_enhanced(model, dataloader, device):
    model.eval()
    total_loss = 0
    param_mse = 0
    system_acc = 0
    n_samples = 0
    
    class_correct = np.zeros(4)
    class_total = np.zeros(4)
    
    with torch.no_grad():
        for batch in dataloader:
            masked_features = batch['masked_features'].to(device).float()
            target_param_value = batch['target_param_value'].to(device).float()
            crystal_system = batch['crystal_system'].to(device).long()
            
            outputs = model(masked_features)
            
            param_loss = F.mse_loss(outputs['param_pred'].squeeze(), target_param_value)
            system_loss = F.cross_entropy(outputs['system_pred'], crystal_system, weight=class_weights)
            
            total_loss_batch = 0.6 * param_loss + 0.4 * system_loss
            
            total_loss += total_loss_batch.item()
            param_mse += param_loss.item()
            
            system_pred = outputs['system_pred'].argmax(dim=1)
            system_acc += (system_pred == crystal_system).float().mean().item()
            
            # Per-class accuracy tracking
            for i in range(4):
                mask = crystal_system == i
                if mask.sum() > 0:
                    class_correct[i] += (system_pred[mask] == i).sum().item()
                    class_total[i] += mask.sum().item()
            
            n_samples += 1
    
    # Per-class accuracies
    class_accs = class_correct / (class_total + 1e-8)
    
    return (total_loss / n_samples, param_mse / n_samples, system_acc / n_samples, class_accs)

# Enhanced training loop
print("\n🚀 Training Enhanced Model...")
best_val_loss = float('inf')
patience_counter = 0
patience = 15

for epoch in range(60):
    train_loss, train_param_mse, train_system_acc = train_epoch_enhanced(model, train_loader, optimizer, device)
    val_loss, val_param_mse, val_system_acc, val_class_accs = validate_enhanced(model, val_loader, device)
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'block2_enhanced_best.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:2d}/60 | Loss: {val_loss:.3f} | "
              f"Param MSE: {val_param_mse:.4f} | System Acc: {val_system_acc:.3f}")
        print(f"    Class Accs: {[f'{acc:.3f}' for acc in val_class_accs]}")
    
    if patience_counter >= patience:
        print(f"⏰ Early stopping at epoch {epoch+1}")
        break

# Final evaluation
model.load_state_dict(torch.load('block2_enhanced_best.pth'))

print("\n📊 FINAL ENHANCED RESULTS:")
train_metrics = validate_enhanced(model, train_loader, device)
val_metrics = validate_enhanced(model, val_loader, device)
test_metrics = validate_enhanced(model, test_loader, device)

print(f"Split      | Loss   | Param MSE | System Acc | Class Accuracies")
print(f"-----------|--------|-----------|------------|------------------")
print(f"Train      | {train_metrics[0]:.3f}  | {train_metrics[1]:.4f}    | {train_metrics[2]:.3f}      | {[f'{acc:.3f}' for acc in train_metrics[3]]}")
print(f"Validation | {val_metrics[0]:.3f}  | {val_metrics[1]:.4f}    | {val_metrics[2]:.3f}      | {[f'{acc:.3f}' for acc in val_metrics[3]]}")
print(f"Test       | {test_metrics[0]:.3f}  | {test_metrics[1]:.4f}    | {test_metrics[2]:.3f}      | {[f'{acc:.3f}' for acc in test_metrics[3]]}")

# Extract enhanced dense features
print("\n💾 Extracting enhanced dense features...")
model.eval()

def extract_features_enhanced(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch['original_features'].to(device).float()
            y_batch = batch['synthesis_label']
            feat = model(X_batch, return_features=True)
            features.append(feat.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.vstack(features), np.concatenate(labels)

h2_train_enhanced, y_train_check = extract_features_enhanced(train_loader)
h2_val_enhanced, y_val_check = extract_features_enhanced(val_loader)
h2_test_enhanced, y_test_check = extract_features_enhanced(test_loader)

print(f"✅ Enhanced dense features extracted:")
print(f"   h2_train: {h2_train_enhanced.shape}")
print(f"   h2_val: {h2_val_enhanced.shape}")
print(f"   h2_test: {h2_test_enhanced.shape}")

# Test synthesis prediction
train_binary = (y_train_check == 1).astype(int)
val_binary = (y_val_check == 1).astype(int)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(h2_train_enhanced, train_binary)
val_pred = clf.predict_proba(h2_val_enhanced)[:, 1]
synthesis_auc = roc_auc_score(val_binary, val_pred)

print(f"\n🧪 Synthesis prediction AUC: {synthesis_auc:.3f}")

# Save enhanced outputs
np.save('block2_features_train.npy', h2_train_enhanced)
np.save('block2_features_val.npy', h2_val_enhanced)
np.save('block2_features_test.npy', h2_test_enhanced)

print(f"\n✅ Enhanced Block 2 completed!")

# Performance comparison
min_class_acc_test = min(test_metrics[3])
print(f"\n🎯 IMPROVEMENT SUMMARY:")
print(f"   Minimum class accuracy (test): {min_class_acc_test:.3f}")
print(f"   Overall system accuracy (test): {test_metrics[2]:.3f}")
print(f"   Target achievement: {'✅' if min_class_acc_test > 0.6 and test_metrics[2] > 0.95 else '⚠️'}")

print("\n" + "=" * 80) 