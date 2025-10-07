#!/usr/bin/env python3
"""
ORIGINAL STEP 1 DATA PREPROCESSING AND SPLITTING
=================================================
This is the ACTUAL code that created the 3 CSV metadata files:
- train_metadata_final.csv
- validation_metadata_final.csv  
- test_metadata_final.csv

Date: June 4, 2025
Status: VERIFIED - This code was used to create your current data files
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== STEP 1 FINAL: 3-Way Temporal Split for PU Learning ===")
print("Train: 2011 - 08/01/2018 | Validation: 08/01/2018+ - 2019 | Test: 2019+")
print("=" * 70)

# 1. Load data from h5 file
print("\n📂 1. Loading FTCP data from h5 file...")
material_ids = []
materials_data = []

with h5py.File('data/ftcp_data.h5', 'r') as f:
    print(f"Number of batches in h5 file: {len(f.keys())}")
    
    # Iterate through all batches
    for batch_name in sorted(f.keys()):
        batch = f[batch_name]
        
        # Get material names and features from this batch
        cif_names = batch['cif_names'][:]
        ftcp_data = batch['FTCP_normalized'][:]  # Using normalized FTCP data (400, 63)
        
        print(f"  {batch_name}: {len(cif_names)} materials")
        
        # Process each material in this batch
        for i, cif_name in enumerate(cif_names):
            # Convert bytes to string and extract material ID
            cif_str = cif_name.decode('utf-8') if isinstance(cif_name, bytes) else str(cif_name)
            
            # Extract material ID from cif name
            if 'mp-' in cif_str:
                start_idx = cif_str.find('mp-')
                material_part = cif_str[start_idx:]
                if '.' in material_part:
                    material_id = material_part.split('.')[0]
                else:
                    material_id = material_part
                
                material_ids.append(material_id)
                materials_data.append(ftcp_data[i])

print(f"\n✅ Total materials loaded: {len(material_ids):,}")
print(f"✅ Feature shape per material: {materials_data[0].shape}")

# 2. Load metadata and match materials
print("\n📊 2. Loading metadata and matching materials...")
metadata_df = pd.read_excel('data/mp_structures_with_dates.xlsx')
print(f"Excel metadata shape: {metadata_df.shape}")

# Match materials between h5 and excel
h5_materials_set = set(material_ids)
matched_metadata = metadata_df[metadata_df['material_id'].isin(h5_materials_set)].copy()
print(f"Successfully matched materials: {len(matched_metadata):,}")

# Create PU labels
def create_pu_labels(synthesizable_value):
    if synthesizable_value == True or synthesizable_value == 'TRUE' or synthesizable_value == 'True':
        return 1  # Positive
    else:
        return -1  # Unknown

matched_metadata['pu_label'] = matched_metadata['synthesizable'].apply(create_pu_labels)

# Parse dates with flexible format
matched_metadata['created_at'] = pd.to_datetime(matched_metadata['created_at'], format='mixed', errors='coerce')
matched_metadata['year'] = matched_metadata['created_at'].dt.year
matched_metadata['month'] = matched_metadata['created_at'].dt.month

print(f"Date range: {matched_metadata['created_at'].min()} to {matched_metadata['created_at'].max()}")

# 3. Detailed temporal analysis BEFORE splitting
print("\n📈 3. Detailed Temporal Analysis:")
print("-" * 50)

# Year-wise distribution with positive ratios
yearly_analysis = matched_metadata.groupby('year').agg({
    'pu_label': ['count', 'sum', 'mean'],
    'material_id': 'count'
}).round(4)
yearly_analysis.columns = ['total_materials', 'positive_count', 'positive_ratio', 'verify_count']

print("Year-wise Distribution:")
print("Year | Total | Positive | Unknown | Pos_Ratio")
print("-" * 45)
for year in sorted(yearly_analysis.index):
    if not pd.isna(year):
        total = int(yearly_analysis.loc[year, 'total_materials'])
        positive = int(yearly_analysis.loc[year, 'positive_count'])
        unknown = total - positive
        ratio = yearly_analysis.loc[year, 'positive_ratio']
        print(f"{int(year)} | {total:5,} | {positive:8,} | {unknown:7,} | {ratio:7.1%}")

# 4. Implement 3-way temporal split
print("\n🔄 4. Implementing 3-Way Temporal Split:")
print("-" * 45)

# Define split boundaries
train_end = pd.Timestamp('2018-08-01')
validation_end = pd.Timestamp('2019-01-01')

# Create masks for 3-way split
train_mask = matched_metadata['created_at'] < train_end
validation_mask = (matched_metadata['created_at'] >= train_end) & (matched_metadata['created_at'] < validation_end)
test_mask = matched_metadata['created_at'] >= validation_end

# Split the metadata
train_metadata = matched_metadata[train_mask].copy()
validation_metadata = matched_metadata[validation_mask].copy()
test_metadata = matched_metadata[test_mask].copy()

print(f"TRAIN    (< 08/01/2018): {len(train_metadata):6,} materials")
print(f"VALIDATION (08/01/2018 - 2019): {len(validation_metadata):6,} materials") 
print(f"TEST     (>= 2019):     {len(test_metadata):6,} materials")

# 5. Extract features and labels for each split
print("\n⚙️  5. Extracting features and labels...")

material_to_data = dict(zip(material_ids, materials_data))

def extract_features_and_labels(metadata_subset, name):
    features = []
    labels = []
    valid_metadata = []
    
    for idx, row in metadata_subset.iterrows():
        mat_id = row['material_id']
        if mat_id in material_to_data:
            features.append(material_to_data[mat_id])
            labels.append(row['pu_label'])
            valid_metadata.append(row.to_dict())
    
    features = np.array(features)
    labels = np.array(labels)
    valid_metadata = pd.DataFrame(valid_metadata)
    
    print(f"  {name:10}: {features.shape} | Pos: {(labels==1).sum():5,} | Unk: {(labels==-1).sum():5,} | Pos%: {(labels==1).mean()*100:5.1f}%")
    return features, labels, valid_metadata

print("Dataset    | Shape              | Positive | Unknown | Pos%")
print("-" * 65)
X_train, y_train, train_meta_final = extract_features_and_labels(train_metadata, "TRAIN")
X_val, y_val, val_meta_final = extract_features_and_labels(validation_metadata, "VALIDATION")
X_test, y_test, test_meta_final = extract_features_and_labels(test_metadata, "TEST")

# 6. Save the final datasets with clear naming
print("\n💾 6. Saving final datasets...")

# Features
np.save('X_train_final.npy', X_train)
np.save('X_validation_final.npy', X_val)
np.save('X_test_final.npy', X_test)

# Labels
np.save('y_train_final.npy', y_train)
np.save('y_validation_final.npy', y_val)
np.save('y_test_final.npy', y_test)

# Metadata
train_meta_final.to_csv('train_metadata_final.csv', index=False)
val_meta_final.to_csv('validation_metadata_final.csv', index=False)
test_meta_final.to_csv('test_metadata_final.csv', index=False)

print("✅ Saved:")
print(f"  • X_train_final.npy: {X_train.shape} ({X_train.nbytes/1024**3:.1f} GB)")
print(f"  • X_validation_final.npy: {X_val.shape} ({X_val.nbytes/1024**3:.1f} GB)")
print(f"  • X_test_final.npy: {X_test.shape} ({X_test.nbytes/1024**3:.1f} GB)")
print(f"  • y_train_final.npy, y_validation_final.npy, y_test_final.npy")
print(f"  • train_metadata_final.csv, validation_metadata_final.csv, test_metadata_final.csv")

# 7. Create comprehensive visualization
print("\n📊 7. Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('Final 3-Way Temporal Split for PU Learning', fontsize=16, fontweight='bold')

# Plot 1: Timeline with split boundaries
ax1 = axes[0, 0]
year_counts = matched_metadata['year'].value_counts().sort_index()
bars = ax1.bar(year_counts.index, year_counts.values, alpha=0.7, color='lightblue', edgecolor='navy')
ax1.axvline(x=2018 + 8/12, color='red', linestyle='-', linewidth=2, label='Train|Val Split (Aug 2018)')
ax1.axvline(x=2019, color='orange', linestyle='-', linewidth=2, label='Val|Test Split (2019)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Materials')
ax1.set_title('Material Distribution by Year with Split Boundaries')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: 3-way split distribution
ax2 = axes[0, 1]
split_names = ['Train\n(< Aug 2018)', 'Validation\n(Aug 2018 - 2019)', 'Test\n(≥ 2019)']
split_counts = [len(train_meta_final), len(val_meta_final), len(test_meta_final)]
colors = ['lightcoral', 'gold', 'lightgreen']
bars = ax2.bar(split_names, split_counts, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Number of Materials')
ax2.set_title('3-Way Split Distribution')
for bar, count in zip(bars, split_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{count:,}', ha='center', va='bottom', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Positive ratio by dataset
ax3 = axes[0, 2]
datasets = ['Train', 'Validation', 'Test']
pos_ratios = [(y_train==1).mean(), (y_val==1).mean(), (y_test==1).mean()]
bars = ax3.bar(datasets, pos_ratios, color=['lightcoral', 'gold', 'lightgreen'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('Positive Ratio')
ax3.set_title('Positive Sample Ratio by Dataset')
ax3.set_ylim(0, max(pos_ratios) * 1.2)
for bar, ratio in zip(bars, pos_ratios):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Label distribution in each dataset
ax4 = axes[1, 0]
datasets = ['Train', 'Validation', 'Test']
positive_counts = [(y_train==1).sum(), (y_val==1).sum(), (y_test==1).sum()]
unknown_counts = [(y_train==-1).sum(), (y_val==-1).sum(), (y_test==-1).sum()]

x = np.arange(len(datasets))
width = 0.35
bars1 = ax4.bar(x - width/2, positive_counts, width, label='Positive (1)', color='green', alpha=0.7)
bars2 = ax4.bar(x + width/2, unknown_counts, width, label='Unknown (-1)', color='orange', alpha=0.7)

ax4.set_xlabel('Dataset')
ax4.set_ylabel('Number of Materials')
ax4.set_title('Label Distribution by Dataset')
ax4.set_xticks(x)
ax4.set_xticklabels(datasets)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Temporal positive ratio trend
ax5 = axes[1, 1]
yearly_pos_ratio = matched_metadata.groupby('year')['pu_label'].apply(lambda x: (x == 1).mean())
ax5.plot(yearly_pos_ratio.index, yearly_pos_ratio.values, marker='o', linewidth=2, markersize=6, color='blue')
ax5.axvline(x=2018 + 8/12, color='red', linestyle='-', alpha=0.7, label='Train|Val Split')
ax5.axvline(x=2019, color='orange', linestyle='-', alpha=0.7, label='Val|Test Split')
ax5.set_xlabel('Year')
ax5.set_ylabel('Positive Ratio')
ax5.set_title('Positive Ratio Trend Over Time')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Dataset size comparison
ax6 = axes[1, 2]
sizes = [X_train.nbytes/1024**3, X_val.nbytes/1024**3, X_test.nbytes/1024**3]
bars = ax6.bar(datasets, sizes, color=['lightcoral', 'gold', 'lightgreen'], alpha=0.8, edgecolor='black')
ax6.set_ylabel('Size (GB)')
ax6.set_title('Dataset Size Comparison')
for bar, size in zip(bars, sizes):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{size:.1f}GB', ha='center', va='bottom', fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_3way_split_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Final Summary Report
print("\n" + "=" * 70)
print("🎯 FINAL 3-WAY SPLIT SUMMARY REPORT")
print("=" * 70)

print(f"\n📊 OVERALL STATISTICS:")
print(f"Total materials processed: {len(matched_metadata):,}")
print(f"Total positive samples: {(matched_metadata['pu_label']==1).sum():,} ({(matched_metadata['pu_label']==1).mean()*100:.1f}%)")
print(f"Total unknown samples: {(matched_metadata['pu_label']==-1).sum():,} ({(matched_metadata['pu_label']==-1).mean()*100:.1f}%)")

print(f"\n🔄 SPLIT BOUNDARIES:")
print(f"TRAIN:      2011-05-12 → 2018-08-01 (exclusive)")
print(f"VALIDATION: 2018-08-01 → 2019-01-01 (exclusive)")  
print(f"TEST:       2019-01-01 → 2025-03-25")

print(f"\n📈 DATASET BREAKDOWN:")
print("Dataset    | Count     | Positive  | Unknown   | Pos%    | Size")
print("-" * 65)
print(f"TRAIN      | {len(X_train):8,} | {(y_train==1).sum():8,} | {(y_train==-1).sum():8,} | {(y_train==1).mean()*100:5.1f}% | {X_train.nbytes/1024**3:5.1f}GB")
print(f"VALIDATION | {len(X_val):8,} | {(y_val==1).sum():8,} | {(y_val==-1).sum():8,} | {(y_val==1).mean()*100:5.1f}% | {X_val.nbytes/1024**3:5.1f}GB") 
print(f"TEST       | {len(X_test):8,} | {(y_test==1).sum():8,} | {(y_test==-1).sum():8,} | {(y_test==1).mean()*100:5.1f}% | {X_test.nbytes/1024**3:5.1f}GB")

print(f"\n✅ WHY THIS SPLIT IS OPTIMAL FOR PU LEARNING:")
print("1. 🎯 TRAIN: Large, diverse dataset from early years with high positive ratio")
print("2. 🔍 VALIDATION: Realistic temporal distribution for hyperparameter tuning")  
print("3. 🧪 TEST: Clean future data for unbiased evaluation")
print("4. ⚖️  Addresses temporal domain shift gradually")
print("5. 📊 Maintains chronological order for time-series validation")

print(f"\n⚠️  CRITICAL OBSERVATIONS:")
if (y_val==1).mean() < (y_train==1).mean():
    print(f"• Positive ratio drops from train ({(y_train==1).mean()*100:.1f}%) to validation ({(y_val==1).mean()*100:.1f}%)")
    print("• This reflects realistic temporal domain shift")
if (y_test==1).mean() < (y_val==1).mean():
    print(f"• Further drop in test set ({(y_test==1).mean()*100:.1f}%) indicates continuing trend")
    print("• Model must handle increasing scarcity of positive examples")

print(f"\n🚀 READY FOR NEXT STEPS:")
print("• Feature analysis and preprocessing")
print("• Baseline model implementation")  
print("• PU learning algorithm development")
print("• Temporal validation strategies")

print("\n" + "=" * 70)
print("✨ 3-WAY SPLIT COMPLETED SUCCESSFULLY! ✨")
print("=" * 70)

