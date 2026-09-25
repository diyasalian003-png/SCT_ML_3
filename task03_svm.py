import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from skimage.feature import hog
import joblib
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TRAIN_DIR = "training_set/training_set"
TEST_DIR = "test_set"
IMG_SIZE = (64, 64)          # resize target
SAMPLES_PER_CLASS_TRAIN = 1000  # subset for reasonable training time
SAMPLES_PER_CLASS_TEST = 250
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
# ----------------------------------------------------------------------
# 1. Load + preprocess images -> HOG feature vectors
# ----------------------------------------------------------------------
def load_features(folder, label, n_samples):
    files = sorted(os.listdir(folder))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    np.random.shuffle(files)
    files = files[:n_samples]

    feats, labels = [], []
    for fname in files:
        try:
            img = Image.open(os.path.join(folder, fname)).convert("L")  # grayscale
            img = img.resize(IMG_SIZE)
            arr = np.array(img) / 255.0

            hog_feat = hog(
                arr, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys"
            )
            feats.append(hog_feat)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    return feats, labels


print("Loading training images...")
cat_train_f, cat_train_l = load_features(
    os.path.join(TRAIN_DIR, "cats"), 0, SAMPLES_PER_CLASS_TRAIN
)
dog_train_f, dog_train_l = load_features(
    os.path.join(TRAIN_DIR, "dogs"), 1, SAMPLES_PER_CLASS_TRAIN
)

print("Loading test images...")
cat_test_f, cat_test_l = load_features(
    os.path.join(TEST_DIR, "cats"), 0, SAMPLES_PER_CLASS_TEST
)
dog_test_f, dog_test_l = load_features(
    os.path.join(TEST_DIR, "dogs"), 1, SAMPLES_PER_CLASS_TEST
)

X_train = np.array(cat_train_f + dog_train_f)
y_train = np.array(cat_train_l + dog_train_l)
X_test = np.array(cat_test_f + dog_test_f)
y_test = np.array(cat_test_l + dog_test_l)

print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features each")
print(f"Test set:  {X_test.shape[0]} samples")

# ----------------------------------------------------------------------
# 2. Scale features
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. Train SVM (small grid search over kernel/C for a good fit)
# ----------------------------------------------------------------------
print("\nTraining SVM (grid search)...")
param_grid = {
    "C": [1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale"],
}
grid = GridSearchCV(SVC(random_state=RANDOM_STATE), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.3f}")

best_svm = grid.best_estimator_

# ----------------------------------------------------------------------
# 4. Evaluate on test set
# ----------------------------------------------------------------------
y_pred = best_svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# ----------------------------------------------------------------------
# 5. Confusion matrix plot
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"SVM Confusion Matrix (Accuracy: {acc:.1%})")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 6. Sample predictions visualization
# ----------------------------------------------------------------------
sample_idx = np.random.choice(len(X_test), size=8, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
test_files_cats = sorted(os.listdir(os.path.join(TEST_DIR, "cats")))
test_files_dogs = sorted(os.listdir(os.path.join(TEST_DIR, "dogs")))

# Rebuild a mapping from index to actual image path for display
all_test_paths = (
    [os.path.join(TEST_DIR, "cats", f) for f in test_files_cats[:SAMPLES_PER_CLASS_TEST]]
    + [os.path.join(TEST_DIR, "dogs", f) for f in test_files_dogs[:SAMPLES_PER_CLASS_TEST]]
)

labels_map = {0: "Cat", 1: "Dog"}
for ax, idx in zip(axes.ravel(), sample_idx):
    img = Image.open(all_test_paths[idx]).convert("RGB").resize((150, 150))
    ax.imshow(img)
    true_l = labels_map[y_test[idx]]
    pred_l = labels_map[y_pred[idx]]
    color = "green" if true_l == pred_l else "red"
    ax.set_title(f"True: {true_l} | Pred: {pred_l}", color=color, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_predictions.png", bbox_inches="tight")
plt.close()
joblib.dump(best_svm, "outputs/svm_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("Model saved to outputs/svm_model.pkl")
print("\nDone. Outputs saved to outputs/")
