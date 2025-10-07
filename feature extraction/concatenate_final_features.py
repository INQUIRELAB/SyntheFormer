#!/usr/bin/env python3

import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("🔄 CONCATENATING FINAL FEATURES")
print("Creating 2048D feature vectors from all blocks")
print("=" * 80)

# Load features from each block
print("\n📂 Loading block features...")

# Block 1: Crystal System (256D)
h1_train = np.load('block1_features_train.npy')
h1_val = np.load('block1_features_val.npy')
h1_test = np.load('block1_features_test.npy')
print(f"✅ Block 1 features loaded: {h1_train.shape}, {h1_val.shape}, {h1_test.shape}")

# Block 2: Crystal System Enhanced (256D)
h2_train = np.load('block2_features_train.npy')
h2_val = np.load('block2_features_val.npy')
h2_test = np.load('block2_features_test.npy')
print(f"✅ Block 2 features loaded: {h2_train.shape}, {h2_val.shape}, {h2_test.shape}")

# Block 3: Atomic Sites (256D) - Using full version
h3_train = np.load('Step2_Block3_features_train_full.npy')
h3_val = np.load('Step2_Block3_features_val_full.npy')
h3_test = np.load('Step2_Block3_features_test_full.npy')
print(f"✅ Block 3 features loaded: {h3_train.shape}, {h3_val.shape}, {h3_test.shape}")

# Block 4: Site Occupancy (256D) - Using full version
h4_train = np.load('Step2_Block4_features_train_full.npy')
h4_val = np.load('Step2_Block4_features_val_full.npy')
h4_test = np.load('Step2_Block4_features_test_full.npy')
print(f"✅ Block 4 features loaded: {h4_train.shape}, {h4_val.shape}, {h4_test.shape}")

# Block 5: Reciprocal Space (256D)
h5_train = np.load('Step2_Block5_features_train.npy')
h5_val = np.load('Step2_Block5_features_val.npy')
h5_test = np.load('Step2_Block5_features_test.npy')
print(f"✅ Block 5 features loaded: {h5_train.shape}, {h5_val.shape}, {h5_test.shape}")

# Block 6: Structure Factors (768D)
h6_train = np.load('Step2_Block6_features_train.npy')
h6_val = np.load('Step2_Block6_features_val.npy')
h6_test = np.load('Step2_Block6_features_test.npy')
print(f"✅ Block 6 features loaded: {h6_train.shape}, {h6_val.shape}, {h6_test.shape}")

# Load labels
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')
print(f"✅ Labels loaded: {y_train.shape}, {y_val.shape}, {y_test.shape}")

# Concatenate features
print("\n🔄 Concatenating features...")
X_train = np.concatenate([h1_train, h2_train, h3_train, h4_train, h5_train, h6_train], axis=1)
X_val = np.concatenate([h1_val, h2_val, h3_val, h4_val, h5_val, h6_val], axis=1)
X_test = np.concatenate([h1_test, h2_test, h3_test, h4_test, h5_test, h6_test], axis=1)

print(f"✅ Final feature shapes:")
print(f"   Train: {X_train.shape}")
print(f"   Val: {X_val.shape}")
print(f"   Test: {X_test.shape}")

# Save concatenated features
print("\n💾 Saving concatenated features...")
np.save('final_features_train.npy', X_train)
np.save('final_features_val.npy', X_val)
np.save('final_features_test.npy', X_test)
print("✅ Features saved successfully!")

# Evaluate performance
print("\n📊 Evaluating performance...")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)[:, 1]
y_val_pred = clf.predict_proba(X_val)[:, 1]
y_test_pred = clf.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, y_train_pred)
auc_val = roc_auc_score(y_val, y_val_pred)
auc_test = roc_auc_score(y_test, y_test_pred)

print(f"\n🎯 Performance Results:")
print(f"   Train AUC: {auc_train:.4f}")
print(f"   Val AUC: {auc_val:.4f}")
print(f"   Test AUC: {auc_test:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
splits = ['Train', 'Val', 'Test']
aucs = [auc_train, auc_val, auc_test]
colors = ['lightblue', 'lightgreen', 'lightcoral']

bars = plt.bar(splits, aucs, color=colors, alpha=0.8, edgecolor='black')
plt.axhline(0.73, color='red', linestyle='--', linewidth=2, label='Target AUC')
plt.ylabel('AUC Score')
plt.title('🎯 Final Feature Performance\n(2048D Concatenated Features)')
plt.legend()
plt.grid(True, alpha=0.3)

for bar, auc in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('final_features_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n📊 Feature Statistics:")
print(f"   Total dimensions: {X_train.shape[1]}D")
print(f"   Components:")
print(f"   ├── Block 1 (Crystal System): {h1_train.shape[1]}D")
print(f"   ├── Block 2 (Enhanced): {h2_train.shape[1]}D")
print(f"   ├── Block 3 (Atomic Sites): {h3_train.shape[1]}D")
print(f"   ├── Block 4 (Site Occupancy): {h4_train.shape[1]}D")
print(f"   ├── Block 5 (Reciprocal): {h5_train.shape[1]}D")
print(f"   └── Block 6 (Structure): {h6_train.shape[1]}D")

print("\n✅ Final feature concatenation complete!")
print("📁 Generated files:")
print("   ├── final_features_train.npy")
print("   ├── final_features_val.npy")
print("   ├── final_features_test.npy")
print("   └── final_features_performance.png") 