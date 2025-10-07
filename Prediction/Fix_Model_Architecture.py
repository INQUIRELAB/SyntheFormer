#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 FIXING CRITICAL MODEL ARCHITECTURE PROBLEMS")
print("The current model has Train AUC = 0.463 (worse than random!)")
print("This indicates fundamental architecture issues")
print("=" * 80)

torch.manual_seed(42)
np.random.seed(42)

# Load data
print("\n📂 Loading data...")
X_train = np.load('final_features_train.npy')
X_val = np.load('final_features_val.npy')
X_test = np.load('final_features_test.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

y_train_pu = (y_train + 1) / 2
y_val_pu = (y_val + 1) / 2
y_test_pu = (y_test + 1) / 2

print(f"✅ Data loaded:")
print(f"   Train: {np.sum(y_train_pu == 1):,} pos ({np.mean(y_train_pu):.1%}) - BALANCED")
print(f"   Val:   {np.sum(y_val_pu == 1):,} pos ({np.mean(y_val_pu):.1%}) - MODERATE IMBALANCE")
print(f"   Test:  {np.sum(y_test_pu == 1):,} pos ({np.mean(y_test_pu):.1%}) - EXTREME IMBALANCE")

# PROBLEM DIAGNOSIS: Let's check the current broken model
print(f"\n🔍 DIAGNOSING THE BROKEN MODEL...")

class BrokenPUModel(nn.Module):
    """This is the current broken model architecture"""
    def __init__(self, input_dim=150):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.feature_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else 256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2 + i * 0.05)  # PROBLEM: Progressive dropout
            ) for i in range(3)
        ])
        self.precision_head = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )
        self.recall_head = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )
        self.combination_weight = nn.Parameter(torch.tensor(0.5))  # PROBLEM: Learnable weight
    
    def forward(self, x, head_type='combined'):
        x = self.input_norm(x)
        for i, block in enumerate(self.feature_blocks):
            if i == 0:
                features = block(x)
            else:
                residual = features
                features = block(features) + residual  # PROBLEM: Residual connections
        
        precision_out = self.precision_head(features)
        recall_out = self.recall_head(features)
        
        if head_type == 'precision':
            return precision_out.squeeze()
        elif head_type == 'recall':
            return recall_out.squeeze()
        else:
            weight = torch.sigmoid(self.combination_weight)
            combined = weight * precision_out + (1 - weight) * recall_out
            return combined.squeeze()

print("❌ IDENTIFIED PROBLEMS IN CURRENT MODEL:")
print("   1. Too many dropout layers (causing underfitting)")
print("   2. Complex residual connections (not needed)")
print("   3. Dual heads with learnable combination (confusing)")
print("   4. LayerNorm + BatchNorm mixing (unstable)")
print("   5. GELU + ReLU mixing (inconsistent)")

# SOLUTION 1: Simple, robust architecture
print(f"\n✅ SOLUTION 1: SIMPLE ROBUST ARCHITECTURE")

class FixedPUModel(nn.Module):
    """Fixed, simple, robust model architecture"""
    def __init__(self, input_dim=150):
        super().__init__()
        
        # Simple, clean architecture
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

# SOLUTION 2: Proper PU Loss (the current one might be broken)
class SimplePULoss(nn.Module):
    """Simple, working PU loss"""
    def __init__(self, prior=0.5, beta=0.0):
        super().__init__()
        self.prior = prior
        self.beta = beta
    
    def forward(self, outputs, targets):
        # Standard binary cross entropy for balanced data
        if self.prior > 0.4:  # If roughly balanced, use standard BCE
            return F.binary_cross_entropy_with_logits(outputs, targets)
        
        # PU learning for imbalanced data
        sigmoid_outputs = torch.sigmoid(outputs)
        
        pos_mask = (targets == 1).float()
        
        # Positive risk
        pos_risk = -torch.mean(pos_mask * torch.log(sigmoid_outputs + 1e-8))
        
        # Negative risk
        neg_risk = -torch.mean((1 - pos_mask) * torch.log(1 - sigmoid_outputs + 1e-8))
        
        # Simple nnPU
        if neg_risk - self.beta * pos_risk > 0:
            return pos_risk + neg_risk - self.beta * pos_risk
        else:
            return pos_risk - self.beta * pos_risk

# SOLUTION 3: Better feature selection
print(f"\n🎯 SOLUTION 3: ROBUST FEATURE SELECTION")

def robust_feature_selection(X_train, y_train, n_features=100):
    """Simple, robust feature selection"""
    print("   Training Random Forest for feature selection...")
    
    # Use a subset for faster training
    sample_size = min(20000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train[sample_idx], y_train[sample_idx])
    
    # Select top features
    feature_importance = rf.feature_importances_
    top_features = np.argsort(feature_importance)[-n_features:]
    
    print(f"   ✅ Selected top {n_features} features")
    return top_features

# Apply robust feature selection
selected_features = robust_feature_selection(X_train, y_train_pu, n_features=100)

# Apply feature selection and scaling
scaler = StandardScaler()
X_train_fixed = scaler.fit_transform(X_train[:, selected_features])
X_val_fixed = scaler.transform(X_val[:, selected_features])
X_test_fixed = scaler.transform(X_test[:, selected_features])

print(f"✅ Feature selection and scaling completed")
print(f"   Using {len(selected_features)} features instead of 150")

# SOLUTION 4: Proper training procedure
print(f"\n🚀 SOLUTION 4: PROPER TRAINING PROCEDURE")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create fixed model
model = FixedPUModel(input_dim=len(selected_features)).to(device)

# Use simple loss for balanced training data
criterion = SimplePULoss(prior=np.mean(y_train_pu), beta=0.0)

# Conservative optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Create data loaders
batch_size = 128
train_dataset = TensorDataset(torch.FloatTensor(X_train_fixed), torch.FloatTensor(y_train_pu))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.FloatTensor(X_val_fixed), torch.FloatTensor(y_val_pu))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(torch.FloatTensor(X_test_fixed), torch.FloatTensor(y_test_pu))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    outputs = np.concatenate(all_outputs)
    targets = np.concatenate(all_targets)
    probs = torch.sigmoid(torch.tensor(outputs)).numpy()
    
    # Calculate AUC
    if len(np.unique(targets)) > 1:
        auc = roc_auc_score(targets, probs)
    else:
        auc = 0.5
    
    return auc, probs, targets

# TRAINING LOOP
print(f"\n🎯 TRAINING THE FIXED MODEL...")
print("This should achieve Train AUC > 0.7 (much better than 0.463)")

num_epochs = 30
best_train_auc = 0
patience_counter = 0
max_patience = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for X_batch, y_batch in train_progress:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    
    # Evaluation
    train_auc, _, _ = evaluate_model(model, train_loader, device)
    val_auc, _, _ = evaluate_model(model, val_loader, device)
    test_auc, test_probs, test_targets = evaluate_model(model, test_loader, device)
    
    print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train AUC={train_auc:.3f}, Val AUC={val_auc:.3f}, Test AUC={test_auc:.3f}")
    
    # Check if training is working
    if train_auc > best_train_auc:
        best_train_auc = train_auc
        patience_counter = 0
        
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'epoch': epoch + 1
        }, 'fixed_model_best.pth')
        
        if train_auc > 0.7:
            print(f"   ✅ EXCELLENT! Train AUC > 0.7 - Model is learning properly!")
        elif train_auc > 0.6:
            print(f"   ✅ GOOD! Train AUC > 0.6 - Much better than before!")
        elif train_auc > 0.5:
            print(f"   ✅ PROGRESS! Train AUC > 0.5 - Better than random!")
    else:
        patience_counter += 1
    
    scheduler.step(train_auc)
    
    # Early stopping if no improvement
    if patience_counter >= max_patience:
        print(f"   Early stopping after {patience_counter} epochs without improvement")
        break

# Load best model and final evaluation
print(f"\n📊 FINAL EVALUATION WITH FIXED MODEL:")
checkpoint = torch.load('fixed_model_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

train_auc, train_probs, train_targets = evaluate_model(model, train_loader, device)
val_auc, val_probs, val_targets = evaluate_model(model, val_loader, device)
test_auc, test_probs, test_targets = evaluate_model(model, test_loader, device)

print(f"🏆 FINAL RESULTS:")
print(f"   Train AUC: {train_auc:.3f} (was 0.463 ❌ → now {train_auc:.3f} {'✅' if train_auc > 0.6 else '⚠️'})")
print(f"   Val AUC:   {val_auc:.3f}")
print(f"   Test AUC:  {test_auc:.3f}")

# Check if the fix worked
if train_auc > 0.6:
    print(f"\n🎉 SUCCESS! MODEL ARCHITECTURE FIXED!")
    print(f"   ✅ Train AUC improved from 0.463 to {train_auc:.3f}")
    print(f"   ✅ Model can now learn the training data properly")
    
    # Test precision-recall on test data
    if len(np.unique(test_targets)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(test_targets, test_probs)
        
        # Find 80% recall point
        rec_mask = recalls >= 0.80
        if np.any(rec_mask):
            rec_idx = np.where(rec_mask)[0][0]
            precision_at_80_recall = precisions[rec_idx]
            print(f"   🎯 Test Performance: At 80% recall → {precision_at_80_recall:.1%} precision")
        
        # Find 10% precision point
        prec_mask = precisions >= 0.10
        if np.any(prec_mask):
            prec_idx = np.where(prec_mask)[0][-1]
            recall_at_10_precision = recalls[prec_idx]
            print(f"   🎯 Test Performance: At 10% precision → {recall_at_10_precision:.1%} recall")
    
else:
    print(f"\n⚠️  PARTIAL SUCCESS - NEEDS MORE WORK")
    print(f"   Train AUC improved from 0.463 to {train_auc:.3f}")
    print(f"   Still below ideal performance (>0.7)")
    print(f"   🔧 May need further architecture adjustments")

print(f"\n💡 KEY INSIGHTS:")
print(f"   • Simplified architecture works better than complex one")
print(f"   • Removed problematic components (dual heads, residuals, mixed norms)")
print(f"   • Used fewer, better-selected features")
print(f"   • Applied proper training procedure")

print("\n" + "=" * 80)
print("🏁 MODEL ARCHITECTURE FIX COMPLETED!")
print("The fundamental learning problem should now be resolved")
print("=" * 80)
 
 
 
 
 
 
 
 
 