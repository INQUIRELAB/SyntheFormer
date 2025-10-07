#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 ADVANCED MULTI-THRESHOLD OPTIMIZATION")
print("Implementing sophisticated threshold strategies:")
print("1. Dual Threshold (High Confidence + Low Confidence zones)")
print("2. Triple Threshold (Confident Positive + Uncertain + Confident Negative)")
print("3. Adaptive Threshold (Based on prediction confidence)")
print("4. Cost-Sensitive Threshold (Minimize synthesis costs)")
print("5. Ensemble Threshold (Multiple criteria combination)")
print("=" * 80)

torch.manual_seed(42)
np.random.seed(42)

# Load data and model (same setup as before)
X_train = np.load('final_features_train.npy')
X_val = np.load('final_features_val.npy')
X_test = np.load('final_features_test.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

y_train_pu = (y_train + 1) / 2
y_val_pu = (y_val + 1) / 2
y_test_pu = (y_test + 1) / 2

# Feature selection and preprocessing
def robust_feature_selection(X_train, y_train, n_features=100, random_state=42):
    np.random.seed(random_state)
    sample_size = min(20000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X_train[sample_idx], y_train[sample_idx])
    feature_importance = rf.feature_importances_
    top_features = np.argsort(feature_importance)[-n_features:]
    return top_features

selected_features = robust_feature_selection(X_train, y_train_pu, n_features=100, random_state=42)
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train[:, selected_features])
X_val_processed = scaler.transform(X_val[:, selected_features])
X_test_processed = scaler.transform(X_test[:, selected_features])

# Load model
class FixedPUModel(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FixedPUModel(input_dim=100).to(device)
checkpoint = torch.load('fixed_model_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Get predictions
def get_predictions(model, X, y, device):
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_scores = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            scores = model(X_batch).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(scores)).numpy()
            
            all_scores.extend(scores)
            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())
    
    return np.array(all_scores), np.array(all_probs), np.array(all_targets)

print("🔮 Getting predictions for all datasets...")
train_scores, train_probs, train_targets = get_predictions(model, X_train_processed, y_train_pu, device)
val_scores, val_probs, val_targets = get_predictions(model, X_val_processed, y_val_pu, device)
test_scores, test_probs, test_targets = get_predictions(model, X_test_processed, y_test_pu, device)

print("✅ Got all predictions!")

# 1. DUAL THRESHOLD STRATEGY
print("\n" + "=" * 80)
print("🎯 STRATEGY 1: DUAL THRESHOLD OPTIMIZATION")
print("High confidence zone + Low confidence zone + Uncertain zone")
print("=" * 80)

def dual_threshold_classify(probs, threshold_high, threshold_low):
    """
    Dual threshold classification:
    - prob >= threshold_high: SYNTHESIZABLE (high confidence)
    - prob <= threshold_low: NOT SYNTHESIZABLE (high confidence)  
    - threshold_low < prob < threshold_high: UNCERTAIN (requires human review)
    """
    predictions = np.full(len(probs), -1)  # -1 = uncertain
    predictions[probs >= threshold_high] = 1   # synthesizable
    predictions[probs <= threshold_low] = 0    # not synthesizable
    return predictions

def evaluate_dual_threshold(probs, targets, threshold_high, threshold_low):
    predictions = dual_threshold_classify(probs, threshold_high, threshold_low)
    
    # Separate into confident and uncertain predictions
    confident_mask = predictions != -1
    uncertain_mask = predictions == -1
    
    if np.sum(confident_mask) == 0:
        return 0, 0, 0, 0, 0, 0  # No confident predictions
    
    confident_preds = predictions[confident_mask]
    confident_targets = targets[confident_mask]
    
    # Calculate metrics for confident predictions only
    tn, fp, fn, tp = confusion_matrix(confident_targets, confident_preds).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(confident_targets) if len(confident_targets) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    coverage = np.sum(confident_mask) / len(probs)  # Fraction of confident predictions
    uncertain_rate = np.sum(uncertain_mask) / len(probs)
    
    return precision, recall, accuracy, f1, coverage, uncertain_rate

def optimize_dual_threshold(probs, targets):
    """Find optimal dual thresholds using grid search"""
    best_score = 0
    best_thresholds = (0.5, 0.5)
    best_metrics = None
    
    # Grid search over threshold combinations
    high_thresholds = np.arange(0.3, 1.0, 0.05)
    low_thresholds = np.arange(0.0, 0.7, 0.05)
    
    results = []
    
    for th_high in high_thresholds:
        for th_low in low_thresholds:
            if th_low >= th_high:  # Invalid: low threshold must be < high threshold
                continue
                
            precision, recall, accuracy, f1, coverage, uncertain_rate = evaluate_dual_threshold(
                probs, targets, th_high, th_low
            )
            
            # Combined score: balance F1, coverage, and precision
            if coverage > 0.1:  # At least 10% coverage
                combined_score = f1 * 0.5 + precision * 0.3 + coverage * 0.2
                
                results.append({
                    'th_high': th_high,
                    'th_low': th_low,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'f1': f1,
                    'coverage': coverage,
                    'uncertain_rate': uncertain_rate,
                    'combined_score': combined_score
                })
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_thresholds = (th_high, th_low)
                    best_metrics = (precision, recall, accuracy, f1, coverage, uncertain_rate)
    
    return best_thresholds, best_metrics, results

# Optimize dual thresholds on validation data
print("🔍 Optimizing dual thresholds on validation data...")
val_dual_thresholds, val_dual_metrics, val_dual_results = optimize_dual_threshold(val_probs, val_targets)

th_high, th_low = val_dual_thresholds
precision, recall, accuracy, f1, coverage, uncertain_rate = val_dual_metrics

print(f"✅ OPTIMAL DUAL THRESHOLDS (Validation):")
print(f"   High threshold (Synthesizable): {th_high:.3f}")
print(f"   Low threshold (Not Synthesizable): {th_low:.3f}")
print(f"   Performance on confident predictions:")
print(f"   • Precision: {precision:.1%}")
print(f"   • Recall: {recall:.1%}")
print(f"   • Accuracy: {accuracy:.1%}")
print(f"   • F1 Score: {f1:.3f}")
print(f"   • Coverage: {coverage:.1%} (confident predictions)")
print(f"   • Uncertain: {uncertain_rate:.1%} (need human review)")

# Test dual thresholds on test data
print(f"\n🎯 TESTING DUAL THRESHOLDS ON TEST DATA:")
test_precision, test_recall, test_accuracy, test_f1, test_coverage, test_uncertain_rate = evaluate_dual_threshold(
    test_probs, test_targets, th_high, th_low
)

print(f"   Test Performance:")
print(f"   • Precision: {test_precision:.1%}")
print(f"   • Recall: {test_recall:.1%}")
print(f"   • Accuracy: {test_accuracy:.1%}")
print(f"   • F1 Score: {test_f1:.3f}")
print(f"   • Coverage: {test_coverage:.1%} ({int(test_coverage * len(test_probs)):,} confident predictions)")
print(f"   • Uncertain: {test_uncertain_rate:.1%} ({int(test_uncertain_rate * len(test_probs)):,} need review)")

# 2. TRIPLE THRESHOLD STRATEGY
print("\n" + "=" * 80)
print("🎯 STRATEGY 2: TRIPLE THRESHOLD OPTIMIZATION")
print("Confident Positive + Uncertain + Confident Negative + Highly Uncertain")
print("=" * 80)

def triple_threshold_classify(probs, threshold_very_high, threshold_high, threshold_low):
    """
    Triple threshold classification:
    - prob >= threshold_very_high: HIGHLY SYNTHESIZABLE (very confident)
    - threshold_high <= prob < threshold_very_high: LIKELY SYNTHESIZABLE (confident)
    - threshold_low < prob < threshold_high: UNCERTAIN (human review)
    - prob <= threshold_low: NOT SYNTHESIZABLE (confident)
    """
    predictions = np.full(len(probs), -1)  # -1 = uncertain
    predictions[probs >= threshold_very_high] = 2   # highly synthesizable
    predictions[(probs >= threshold_high) & (probs < threshold_very_high)] = 1   # likely synthesizable
    predictions[probs <= threshold_low] = 0    # not synthesizable
    return predictions

def optimize_triple_threshold(probs, targets):
    """Find optimal triple thresholds"""
    best_score = 0
    best_thresholds = (0.8, 0.5, 0.2)
    best_metrics = None
    
    very_high_thresholds = np.arange(0.7, 1.0, 0.05)
    high_thresholds = np.arange(0.4, 0.8, 0.05)
    low_thresholds = np.arange(0.0, 0.5, 0.05)
    
    for th_vh in very_high_thresholds:
        for th_h in high_thresholds:
            for th_l in low_thresholds:
                if not (th_l < th_h < th_vh):  # Must be ordered
                    continue
                
                predictions = triple_threshold_classify(probs, th_vh, th_h, th_l)
                
                # Convert to binary for evaluation (2->1, 1->1, 0->0, -1->uncertain)
                binary_preds = predictions.copy()
                binary_preds[binary_preds == 2] = 1  # highly synthesizable -> synthesizable
                
                confident_mask = binary_preds != -1
                if np.sum(confident_mask) < len(probs) * 0.1:  # At least 10% coverage
                    continue
                
                confident_preds = binary_preds[confident_mask]
                confident_targets = targets[confident_mask]
                
                if len(np.unique(confident_targets)) < 2:  # Need both classes
                    continue
                
                tn, fp, fn, tp = confusion_matrix(confident_targets, confident_preds).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                coverage = np.sum(confident_mask) / len(probs)
                
                # Bonus for having very high confidence predictions
                very_high_count = np.sum(predictions == 2)
                very_high_bonus = very_high_count / len(probs) * 0.1
                
                combined_score = f1 * 0.5 + precision * 0.3 + coverage * 0.1 + very_high_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_thresholds = (th_vh, th_h, th_l)
                    best_metrics = (precision, recall, f1, coverage)
    
    return best_thresholds, best_metrics

# Optimize triple thresholds
print("🔍 Optimizing triple thresholds on validation data...")
val_triple_thresholds, val_triple_metrics = optimize_triple_threshold(val_probs, val_targets)

th_vh, th_h, th_l = val_triple_thresholds
precision, recall, f1, coverage = val_triple_metrics

print(f"✅ OPTIMAL TRIPLE THRESHOLDS (Validation):")
print(f"   Very High threshold (Highly Synthesizable): {th_vh:.3f}")
print(f"   High threshold (Likely Synthesizable): {th_h:.3f}")
print(f"   Low threshold (Not Synthesizable): {th_l:.3f}")
print(f"   Performance: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.3f}, Coverage={coverage:.1%}")

# 3. ADAPTIVE THRESHOLD STRATEGY
print("\n" + "=" * 80)
print("🎯 STRATEGY 3: ADAPTIVE THRESHOLD BASED ON PREDICTION CONFIDENCE")
print("Different thresholds for different confidence levels")
print("=" * 80)

def adaptive_threshold_classify(probs, scores):
    """
    Adaptive threshold based on model confidence (score magnitude):
    - High confidence (|score| > 2): Lower threshold needed
    - Medium confidence (1 < |score| <= 2): Standard threshold  
    - Low confidence (|score| <= 1): Higher threshold needed
    """
    predictions = np.zeros(len(probs))
    
    high_conf_mask = np.abs(scores) > 2
    med_conf_mask = (np.abs(scores) > 1) & (np.abs(scores) <= 2)
    low_conf_mask = np.abs(scores) <= 1
    
    # Different thresholds for different confidence levels
    predictions[high_conf_mask] = (probs[high_conf_mask] >= 0.3).astype(int)  # Lower threshold for confident model
    predictions[med_conf_mask] = (probs[med_conf_mask] >= 0.5).astype(int)    # Standard threshold
    predictions[low_conf_mask] = (probs[low_conf_mask] >= 0.7).astype(int)    # Higher threshold for uncertain model
    
    return predictions

# Test adaptive threshold
print("🔍 Testing adaptive threshold strategy...")

val_adaptive_preds = adaptive_threshold_classify(val_probs, val_scores)
test_adaptive_preds = adaptive_threshold_classify(test_probs, test_scores)

# Evaluate adaptive threshold on validation
if len(np.unique(val_targets)) > 1:
    tn, fp, fn, tp = confusion_matrix(val_targets, val_adaptive_preds).ravel()
    val_adaptive_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    val_adaptive_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    val_adaptive_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    print(f"✅ ADAPTIVE THRESHOLD (Validation):")
    print(f"   Precision: {val_adaptive_precision:.1%}")
    print(f"   Recall: {val_adaptive_recall:.1%}")
    print(f"   F1 Score: {val_adaptive_f1:.3f}")

# Evaluate adaptive threshold on test
if len(np.unique(test_targets)) > 1:
    tn, fp, fn, tp = confusion_matrix(test_targets, test_adaptive_preds).ravel()
    test_adaptive_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_adaptive_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_adaptive_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    print(f"🎯 ADAPTIVE THRESHOLD (Test):")
    print(f"   Precision: {test_adaptive_precision:.1%}")
    print(f"   Recall: {test_adaptive_recall:.1%}")
    print(f"   F1 Score: {test_adaptive_f1:.3f}")

# 4. COST-SENSITIVE THRESHOLD STRATEGY
print("\n" + "=" * 80)
print("🎯 STRATEGY 4: COST-SENSITIVE THRESHOLD OPTIMIZATION")
print("Minimize synthesis costs: False positives are expensive!")
print("=" * 80)

def cost_sensitive_objective(threshold, probs, targets, fp_cost=10, fn_cost=1):
    """
    Cost-sensitive optimization:
    - False Positive (predict synthesizable but isn't): High cost (wasted synthesis)
    - False Negative (miss synthesizable): Lower cost (missed opportunity)
    """
    predictions = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    
    total_cost = fp * fp_cost + fn * fn_cost
    return total_cost

def optimize_cost_sensitive_threshold(probs, targets, fp_cost=10, fn_cost=1):
    """Find threshold that minimizes synthesis costs"""
    thresholds = np.arange(0.01, 0.99, 0.01)
    costs = []
    
    for threshold in thresholds:
        cost = cost_sensitive_objective(threshold, probs, targets, fp_cost, fn_cost)
        costs.append(cost)
    
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]
    
    return best_threshold, best_cost

# Optimize cost-sensitive threshold
print("🔍 Optimizing cost-sensitive threshold (FP cost=10x, FN cost=1x)...")
val_cost_threshold, val_cost = optimize_cost_sensitive_threshold(val_probs, val_targets, fp_cost=10, fn_cost=1)

print(f"✅ COST-SENSITIVE THRESHOLD (Validation): {val_cost_threshold:.3f}")
print(f"   Total cost: {val_cost:.0f} units")

# Test cost-sensitive threshold
test_cost_preds = (test_probs >= val_cost_threshold).astype(int)
test_cost = cost_sensitive_objective(val_cost_threshold, test_probs, test_targets, fp_cost=10, fn_cost=1)

if len(np.unique(test_targets)) > 1:
    tn, fp, fn, tp = confusion_matrix(test_targets, test_cost_preds).ravel()
    test_cost_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_cost_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_cost_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    print(f"🎯 COST-SENSITIVE RESULTS (Test):")
    print(f"   Threshold: {val_cost_threshold:.3f}")
    print(f"   Precision: {test_cost_precision:.1%} (fewer false alarms)")
    print(f"   Recall: {test_cost_recall:.1%}")
    print(f"   F1 Score: {test_cost_f1:.3f}")
    print(f"   Total cost: {test_cost:.0f} units")
    print(f"   False positives: {fp:,} (expensive!)")
    print(f"   False negatives: {fn:,} (missed opportunities)")

# 5. ENSEMBLE THRESHOLD STRATEGY
print("\n" + "=" * 80)
print("🎯 STRATEGY 5: ENSEMBLE THRESHOLD COMBINATION")
print("Combining multiple threshold strategies for robust decisions")
print("=" * 80)

def ensemble_threshold_classify(probs, scores, 
                               dual_th_high, dual_th_low,
                               triple_th_vh, triple_th_h, triple_th_l,
                               cost_threshold):
    """
    Ensemble of multiple threshold strategies:
    - If multiple strategies agree: High confidence
    - If strategies disagree: Lower confidence or uncertain
    """
    
    # Get predictions from each strategy
    dual_preds = dual_threshold_classify(probs, dual_th_high, dual_th_low)
    triple_preds = triple_threshold_classify(probs, triple_th_vh, triple_th_h, triple_th_l)
    adaptive_preds = adaptive_threshold_classify(probs, scores)
    cost_preds = (probs >= cost_threshold).astype(int)
    
    # Convert triple predictions to binary
    triple_binary = triple_preds.copy()
    triple_binary[triple_binary == 2] = 1  # highly synthesizable -> synthesizable
    triple_binary[triple_binary == -1] = 0.5  # uncertain -> neutral
    
    # Convert dual predictions
    dual_binary = dual_preds.copy().astype(float)
    dual_binary[dual_binary == -1] = 0.5  # uncertain -> neutral
    
    # Ensemble voting
    ensemble_scores = (dual_binary + triple_binary + adaptive_preds + cost_preds) / 4
    
    # Final decisions based on ensemble consensus
    final_preds = np.full(len(probs), -1)  # uncertain by default
    final_preds[ensemble_scores >= 0.75] = 1  # Strong consensus for synthesizable
    final_preds[ensemble_scores <= 0.25] = 0  # Strong consensus for not synthesizable
    
    return final_preds, ensemble_scores

# Apply ensemble strategy
print("🔍 Applying ensemble threshold strategy...")

test_ensemble_preds, test_ensemble_scores = ensemble_threshold_classify(
    test_probs, test_scores,
    th_high, th_low,  # dual thresholds
    th_vh, th_h, th_l,  # triple thresholds  
    val_cost_threshold  # cost-sensitive threshold
)

# Evaluate ensemble
confident_mask = test_ensemble_preds != -1
if np.sum(confident_mask) > 0:
    confident_preds = test_ensemble_preds[confident_mask]
    confident_targets = test_targets[confident_mask]
    
    if len(np.unique(confident_targets)) > 1:
        tn, fp, fn, tp = confusion_matrix(confident_targets, confident_preds).ravel()
        ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ensemble_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        ensemble_coverage = np.sum(confident_mask) / len(test_probs)
        ensemble_uncertain = np.sum(test_ensemble_preds == -1) / len(test_probs)
        
        print(f"✅ ENSEMBLE THRESHOLD RESULTS (Test):")
        print(f"   Precision: {ensemble_precision:.1%}")
        print(f"   Recall: {ensemble_recall:.1%}")
        print(f"   F1 Score: {ensemble_f1:.3f}")
        print(f"   Coverage: {ensemble_coverage:.1%} ({int(ensemble_coverage * len(test_probs)):,} confident)")
        print(f"   Uncertain: {ensemble_uncertain:.1%} ({int(ensemble_uncertain * len(test_probs)):,} need review)")

# FINAL COMPARISON
print("\n" + "=" * 80)
print("🏆 FINAL COMPARISON: ALL THRESHOLD STRATEGIES")
print("=" * 80)

print(f"📊 TEST DATA PERFORMANCE COMPARISON:")
print(f"   Single Threshold (0.5):     F1={0.035:.3f}, Precision={1.8:.1f}%, Recall={72.3:.1f}%")
print(f"   Best F1 Threshold (0.941):  F1={0.084:.3f}, Precision={6.6:.1f}%, Recall={11.5:.1f}%")
print(f"   Dual Threshold:             F1={test_f1:.3f}, Precision={test_precision:.1f}%, Coverage={test_coverage:.1f}%")
print(f"   Adaptive Threshold:         F1={test_adaptive_f1:.3f}, Precision={test_adaptive_precision:.1f}%")
print(f"   Cost-Sensitive:             F1={test_cost_f1:.3f}, Precision={test_cost_precision:.1f}% (Low FP)")
if 'ensemble_f1' in locals():
    print(f"   Ensemble Strategy:          F1={ensemble_f1:.3f}, Precision={ensemble_precision:.1f}%, Coverage={ensemble_coverage:.1f}%")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   🎯 For highest precision: Use Cost-Sensitive threshold ({val_cost_threshold:.3f})")
print(f"   🎯 For balanced performance: Use Dual threshold system")
print(f"   🎯 For robust decisions: Use Ensemble strategy")
print(f"   🎯 For adaptive decisions: Use confidence-based adaptive thresholds")

print(f"\n" + "=" * 80)
print(f"🏁 ADVANCED MULTI-THRESHOLD OPTIMIZATION COMPLETE!")
print(f"Multiple sophisticated strategies implemented and tested")
print(f"=" * 80)
 
 
 
 
 
 
 
 
 