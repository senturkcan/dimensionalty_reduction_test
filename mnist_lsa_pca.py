import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import seaborn as sns
import time


print("Loading Fashion-MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Reshape and normalize the data
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

print(f"Flattened training data shape: {X_train_flat.shape}")

# Standardize the data for PCA (LSA typically doesn't require standardization)
print("Standardizing data for PCA...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

n_components = 200


print("Applying LSA using Truncated SVD...")
start_time = time.time()
lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_train_lsa = lsa.fit_transform(X_train_flat)
X_test_lsa = lsa.transform(X_test_flat)

lsa_time = time.time() - start_time

print(f"LSA transformation time: {lsa_time:.2f}s")
print(f"Original dimensions: {X_train_flat.shape[1]}")
print(f"LSA reduced dimensions: {X_train_lsa.shape[1]}")
print(f"LSA explained variance ratio: {lsa.explained_variance_ratio_.sum():.4f}")


print("Applying PCA...")
start_time = time.time()

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)  # Use standardized data
X_test_pca = pca.transform(X_test_scaled)

pca_time = time.time() - start_time

print(f"PCA transformation time: {pca_time:.2f}s")
print(f"PCA reduced dimensions: {X_train_pca.shape[1]}")
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")


k_values = [3, 5, 7, 10, 15]
lsa_results = {}

for k in k_values:
    print(f"Testing LSA + k-NN with k={k}...")
    start_time = time.time()

    knn_lsa = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_lsa.fit(X_train_lsa, y_train)

    y_pred_lsa = knn_lsa.predict(X_test_lsa)
    accuracy = accuracy_score(y_test, y_pred_lsa)

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
    knn_pca.fit(X_train_pca, y_train)

    y_pred_pca = knn_pca.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred_pca)

    end_time = time.time()

    pca_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_pca
    }

    print(f"PCA k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")

print("RESULTS COMPARISON: LSA vs PCA")


# Find best k for each method
best_k_lsa = max(lsa_results.keys(), key=lambda k: lsa_results[k]['accuracy'])
best_k_pca = max(pca_results.keys(), key=lambda k: pca_results[k]['accuracy'])

print(f"Best LSA: k={best_k_lsa}, Accuracy={lsa_results[best_k_lsa]['accuracy']:.4f}")
print(f"Best PCA: k={best_k_pca}, Accuracy={pca_results[best_k_pca]['accuracy']:.4f}")

print("FINAL ANALYSIS SUMMARY")
print(f"""


LSA, PCA            
Transform Time:    {lsa_time:.2f}s            {pca_time:.2f}s
Explained Var:     {lsa.explained_variance_ratio_.sum():.4f}           {pca.explained_variance_ratio_.sum():.4f}
Best k:            {best_k_lsa}               {best_k_pca}
Best Accuracy:     {lsa_results[best_k_lsa]['accuracy']:.4f}           {pca_results[best_k_pca]['accuracy']:.4f}
Prediction Time:   {lsa_results[best_k_lsa]['time']:.2f}s            {pca_results[best_k_pca]['time']:.2f}s
""")