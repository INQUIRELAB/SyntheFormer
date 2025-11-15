#!/usr/bin/env python3
"""
Simple inference script using ONNX model
No architecture code needed!

Usage:
    python inference_onnx.py
"""

import numpy as np
import onnxruntime as ort

print("=" * 80)
print("MLP SYNTHESIZABILITY PREDICTION (ONNX)")
print("=" * 80)

# 1. Load ONNX model
print("Loading ONNX model...")
try:
    session = ort.InferenceSession("synthesizability_mlp_model.onnx")
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("   Make sure 'synthesizability_mlp_model.onnx' is in the current directory")
    exit(1)

# 2. Helper function for sigmoid
def sigmoid(x):
    """Convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))

# 3. Prediction function
def predict_synthesizability(features):
    """
    Predict synthesizability score
    
    Args:
        features: numpy array of shape (n_samples, 100) or (100,)
                 Must be 100-dimensional feature vectors
    
    Returns:
        probabilities: numpy array of shape (n_samples,)
                      Values between 0 and 1
                      Higher = more likely synthesizable
    """
    # Handle single sample input
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Validate input shape
    if features.shape[1] != 100:
        raise ValueError(f"Expected 100 features, got {features.shape[1]}")
    
    # Convert to float32 (ONNX requirement)
    features = features.astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {'features': features})
    logits = outputs[0]
    
    # Convert logits to probabilities
    probabilities = sigmoid(logits)
    
    return probabilities

# 4. Example usage
print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)

# Example 1: Single sample
print("\nExample 1: Single sample prediction")
sample_features = np.random.randn(100)  # Replace with your actual features
score = predict_synthesizability(sample_features)
print(f"   Input shape: {sample_features.shape}")
print(f"   Synthesizability score: {score[0]:.4f}")
print(f"   Interpretation: {score[0]*100:.2f}% likely synthesizable")

# Example 2: Batch prediction
print("\nExample 2: Batch prediction (5 samples)")
batch_features = np.random.randn(5, 100)  # Replace with your actual features
scores = predict_synthesizability(batch_features)
print(f"   Input shape: {batch_features.shape}")
print(f"   Scores:")
for i, score in enumerate(scores):
    print(f"      Sample {i+1}: {score:.4f} ({score*100:.2f}% likely synthesizable)")

# Example 3: Classification with threshold
print("\nExample 3: Binary classification (threshold=0.5)")
threshold = 0.5
predictions = (scores >= threshold).astype(int)
print(f"   Threshold: {threshold}")
print(f"   Predictions (0=not synthesizable, 1=synthesizable):")
for i, (score, pred) in enumerate(zip(scores, predictions)):
    label = "Synthesizable" if pred == 1 else "Not Synthesizable"
    print(f"      Sample {i+1}: {score:.4f} -> {label}")

print("\n" + "=" * 80)
print("USAGE NOTES")
print("=" * 80)
print("Input requirements:")
print("   - Shape: (n_samples, 100) or (100,) for single sample")
print("   - Type: numpy array")
print("   - Values: 100-dimensional feature vector")
print("\nOutput:")
print("   - Shape: (n_samples,)")
print("   - Type: numpy array")
print("   - Values: Probabilities between 0 and 1")
print("   - Interpretation: Higher = more likely synthesizable")
print("\nRecommended thresholds:")
print("   - Conservative: 0.7 (fewer false positives)")
print("   - Balanced: 0.5 (standard threshold)")
print("   - Aggressive: 0.3 (catch more synthesizable, more false positives)")
print("=" * 80)

print("\n[OK] Ready to use! Modify this script to load your actual features.")

