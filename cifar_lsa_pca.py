import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import seaborn as sns
import time

print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

y_train = y_train.flatten()
y_test = y_test.flatten()

# Reshape and normalize the data
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

print(f"Flattened training data shape: {X_train_flat.shape}")
print(f"Flattened test data shape: {X_test_flat.shape}")

# Use subset BU FİKİRDEN VAZGEÇİLDİ!
subset_size = 50000
test_subset_size = 10000

# Random sampling for training subset
train_indices = np.random.choice(X_train_flat.shape[0], subset_size, replace=False)
X_train_subset = X_train_flat[train_indices]
y_train_subset = y_train[train_indices]

# Random sampling for test subset
test_indices = np.random.choice(X_test_flat.shape[0], test_subset_size, replace=False)
X_test_subset = X_test_flat[test_indices]
y_test_subset = y_test[test_indices]

# Standardize the data for PCA
print("Standardizing data for PCA...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_subset)
X_test_scaled = scaler.transform(X_test_subset)


n_components = 200


print("Applying LSA using Truncated SVD...")
start_time = time.time()

# LSA with Truncated SVD (works directly on non-negative data)
lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_train_lsa = lsa.fit_transform(X_train_subset)
X_test_lsa = lsa.transform(X_test_subset)

lsa_time = time.time() - start_time

print(f"LSA transformation time: {lsa_time:.2f}s")
print(f"Original dimensions: {X_train_subset.shape[1]}")
print(f"LSA reduced dimensions: {X_train_lsa.shape[1]}")
print(f"LSA explained variance ratio: {lsa.explained_variance_ratio_.sum():.4f}")


print("Applying PCA...")
start_time = time.time()

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_time = time.time() - start_time

print(f"PCA transformation time: {pca_time:.2f}s")
print(f"PCA reduced dimensions: {X_train_pca.shape[1]}")
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")


print("k-NN CLASSIFICATION WITH ORIGINAL DATA")

k_values = [3, 5, 7, 10]  # Reduced k values for computational efficiency
original_results = {}

for k in k_values:
    print(f"Testing Original + k-NN with k={k}...")
    start_time = time.time()

    knn_original = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_original.fit(X_train_scaled, y_train_subset)  # Using standardized data

    y_pred_original = knn_original.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_subset, y_pred_original)

    end_time = time.time()

    original_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_original
    }

    print(f"Original k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")

print("k-NN CLASSIFICATION WITH LSA")

lsa_results = {}

for k in k_values:
    print(f"Testing LSA + k-NN with k={k}...")
    start_time = time.time()

    knn_lsa = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_lsa.fit(X_train_lsa, y_train_subset)

    y_pred_lsa = knn_lsa.predict(X_test_lsa)
    accuracy = accuracy_score(y_test_subset, y_pred_lsa)

    end_time = time.time()

    lsa_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_lsa
    }

    print(f"LSA k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")


print("k-NN CLASSIFICATION WITH PCA")

pca_results = {}

for k in k_values:
    print(f"Testing PCA + k-NN with k={k}...")
    start_time = time.time()

    knn_pca = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_pca.fit(X_train_pca, y_train_subset)

    y_pred_pca = knn_pca.predict(X_test_pca)
    accuracy = accuracy_score(y_test_subset, y_pred_pca)

    end_time = time.time()

    pca_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_pca
    }

    print(f"PCA k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")


print("RESULTS COMPARISON: ORIGINAL vs LSA vs PCA")

# Find best k for each method
best_k_original = max(original_results.keys(), key=lambda k: original_results[k]['accuracy'])
best_k_lsa = max(lsa_results.keys(), key=lambda k: lsa_results[k]['accuracy'])
best_k_pca = max(pca_results.keys(), key=lambda k: pca_results[k]['accuracy'])

print(f"Best Original: k={best_k_original}, Accuracy={original_results[best_k_original]['accuracy']:.4f}")
print(f"Best LSA: k={best_k_lsa}, Accuracy={lsa_results[best_k_lsa]['accuracy']:.4f}")
print(f"Best PCA: k={best_k_pca}, Accuracy={pca_results[best_k_pca]['accuracy']:.4f}")

# Accuracy comparison
k_vals = list(original_results.keys())
original_accuracies = [original_results[k]['accuracy'] for k in k_vals]
lsa_accuracies = [lsa_results[k]['accuracy'] for k in k_vals]
pca_accuracies = [pca_results[k]['accuracy'] for k in k_vals]

# Time comparison
original_times = [original_results[k]['time'] for k in k_vals]
lsa_times = [lsa_results[k]['time'] for k in k_vals]
pca_times = [pca_results[k]['time'] for k in k_vals]



print("FINAL ANALYSIS ")
print(f"""
Original, LSA, PCA 
Transform Time:    N/A             {lsa_time:.2f}s            {pca_time:.2f}s
Explained Var:     N/A             {lsa.explained_variance_ratio_.sum():.4f}           {pca.explained_variance_ratio_.sum():.4f}
Best k:            {best_k_original}              {best_k_lsa}               {best_k_pca}
est Accuracy:     {original_results[best_k_original]['accuracy']:.4f}           {lsa_results[best_k_lsa]['accuracy']:.4f}           {pca_results[best_k_pca]['accuracy']:.4f}
Prediction Time:   {original_results[best_k_original]['time']:.2f}s            {lsa_results[best_k_lsa]['time']:.2f}s            {pca_results[best_k_pca]['time']:.2f}s
""")