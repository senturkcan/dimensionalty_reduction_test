import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import seaborn as sns
import time



(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Reshape and normalize the data
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

print(f"Flattened training data shape: {X_train_flat.shape}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Apply PCA
n_components = 200

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")


# Apply k-NN classifier
print("\nTraining k-NN classifier...")
k_values = [3, 5, 7, 10, 15]
results = {}

for k in k_values:
    print(f"Testing k={k}...")
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()

    results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred
    }

    print(f"k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")

# Find best k value
best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest k value: {best_k} with accuracy: {results[best_k]['accuracy']:.4f}")

# Detailed evaluation for best k
print(f"\nDetailed evaluation for k={best_k}:")
best_predictions = results[best_k]['predictions']

print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=class_names))

# Compare with and without PCA
print("\nComparison: PCA vs No PCA")
print("Training k-NN without PCA (using full dataset)...")

start_time = time.time()
knn_no_pca = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_no_pca.fit(X_train_scaled, y_train)
y_pred_no_pca = knn_no_pca.predict(X_test_scaled)
end_time = time.time()

accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)

print(f"\nResults comparison:")
print(f"With PCA ({n_components} components):")
print(f"  - Accuracy: {results[best_k]['accuracy']:.4f}")
print(f"  - Time: {results[best_k]['time']:.2f}s")
print(f"  - Features: {n_components}")

print(f"\nWithout PCA (full dataset):")
print(f"  - Accuracy: {accuracy_no_pca:.4f}")
print(f"  - Time: {end_time - start_time:.2f}s")
print(f"  - Features: {X_train_scaled.shape[1]}")

# Visualize some test predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# Get some random test samples
sample_indices = np.random.choice(len(X_test), 10, replace=False)

for i, idx in enumerate(sample_indices):
    axes[i].imshow(X_test[idx], cmap='gray')
    true_label = class_names[y_test[idx]]
    pred_label = class_names[best_predictions[idx]]

    color = 'green' if y_test[idx] == best_predictions[idx] else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}',
                      color=color, fontsize=8)
    axes[i].axis('off')

plt.suptitle(f'Sample Predictions (k={best_k})', fontsize=14)
plt.tight_layout()
plt.show()

print(f"- PCA reduced dimensionality from {X_train_scaled.shape[1]} to {n_components}")
print(f"- Retained {pca.explained_variance_ratio_.sum():.1%} of variance")
print(f"- Best k value: {best_k}")
print(f"- Final accuracy: {results[best_k]['accuracy']:.4f}")