import numpy as np
import joblib

# Load data
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

print("="*50)
print("DATA DIAGNOSTICS")
print("="*50)

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution in training data:")
class_names = {1: "Left", 2: "Right", 3: "Feet", 10: "Rest"}
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {cnt} trials")

# Load model
clf = joblib.load('models/riemann_model.pkl')
print(f"\nModel classes: {clf.classes_}")

# Test predictions on training data
preds = clf.predict(X)
pred_unique, pred_counts = np.unique(preds, return_counts=True)
print("\nPredictions on training data:")
for cls, cnt in zip(pred_unique, pred_counts):
    print(f"  Predicted class {cls} ({class_names.get(cls, 'Unknown')}): {cnt} times")

# Confusion per class
print("\nAccuracy per class:")
for cls in unique:
    mask = y == cls
    if mask.sum() > 0:
        acc = (preds[mask] == cls).sum() / mask.sum()
        print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {acc*100:.1f}%")

# Check probabilities for class 2
print("\nProbability analysis for Right (class 2):")
probas = clf.predict_proba(X)
# Find index of class 2 in clf.classes_
if 2 in clf.classes_:
    idx_2 = list(clf.classes_).index(2)
    prob_2 = probas[:, idx_2]
    print(f"  Min probability: {prob_2.min():.3f}")
    print(f"  Max probability: {prob_2.max():.3f}")
    print(f"  Mean probability: {prob_2.mean():.3f}")
    print(f"  Times Right was top prediction: {(preds == 2).sum()}")
else:
    print("  ERROR: Class 2 not in model classes!")
