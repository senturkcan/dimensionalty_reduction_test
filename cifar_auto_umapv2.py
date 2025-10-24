import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import umap
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)


print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


y_train = y_train.flatten()
y_test = y_test.flatten()

# Reshape and normalize the data
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0


# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)


embedding_dim = 200

print("AUTOENCODER DIMENSIONALITY REDUCTION")


def create_autoencoder(input_dim, encoding_dim):
    """Create an autoencoder model"""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='linear', name='encoded')(encoded)

    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Full autoencoder
    autoencoder = Model(input_layer, decoded)

    # Encoder model for dimensionality reduction
    encoder = Model(input_layer, encoded)

    return autoencoder, encoder


print("Creating and training autoencoder...")
start_time = time.time()

# Create autoencoder
autoencoder, encoder = create_autoencoder(X_train_scaled.shape[1], embedding_dim)

# Compile autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae'])

# Print model summary
print("\nAutoencoder Architecture:")
autoencoder.summary()

# Train autoencoder
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nTraining autoencoder...")
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_split=0.1,
                          callbacks=[early_stop],
                          verbose=1)

# Get encoded representations
X_train_ae = encoder.predict(X_train_scaled, verbose=0)
X_test_ae = encoder.predict(X_test_scaled, verbose=0)

ae_time = time.time() - start_time

print(f"\nAutoencoder training and encoding time: {ae_time:.2f}s")
print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Autoencoder reduced dimensions: {X_train_ae.shape[1]}")

# Calculate reconstruction error
X_train_reconstructed = autoencoder.predict(X_train_scaled, verbose=0)
reconstruction_error = np.mean(np.square(X_train_scaled - X_train_reconstructed))
print(f"Mean reconstruction error: {reconstruction_error:.6f}")



print("Applying UMAP...")
start_time = time.time()

# Configure UMAP
umap_reducer = umap.UMAP(
    n_components=embedding_dim,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=True
)

# Fit UMAP on training data
X_train_umap = umap_reducer.fit_transform(X_train_scaled)

# Transform test data
X_test_umap = umap_reducer.transform(X_test_scaled)

umap_time = time.time() - start_time

print(f"\nUMAP transformation time: {umap_time:.2f}s")
print(f"UMAP reduced dimensions: {X_train_umap.shape[1]}")


print("k-NN CLASSIFICATION WITH AUTOENCODER")


k_values = [3, 5, 7, 10, 15]
ae_results = {}

for k in k_values:
    print(f"Testing Autoencoder + k-NN with k={k}...")
    start_time = time.time()

    knn_ae = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_ae.fit(X_train_ae, y_train)

    y_pred_ae = knn_ae.predict(X_test_ae)
    accuracy = accuracy_score(y_test, y_pred_ae)

    end_time = time.time()

    ae_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_ae
    }

    print(f"Autoencoder k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")



print("k-NN CLASSIFICATION WITH UMAP")

umap_results = {}

for k in k_values:
    print(f"Testing UMAP + k-NN with k={k}...")
    start_time = time.time()

    knn_umap = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_umap.fit(X_train_umap, y_train)

    y_pred_umap = knn_umap.predict(X_test_umap)
    accuracy = accuracy_score(y_test, y_pred_umap)

    end_time = time.time()

    umap_results[k] = {
        'accuracy': accuracy,
        'time': end_time - start_time,
        'predictions': y_pred_umap
    }

    print(f"UMAP k={k}: Accuracy = {accuracy:.4f}, Time = {end_time - start_time:.2f}s")


# Find best k for each method
best_k_ae = max(ae_results.keys(), key=lambda k: ae_results[k]['accuracy'])
best_k_umap = max(umap_results.keys(), key=lambda k: umap_results[k]['accuracy'])

print(f"Best Autoencoder: k={best_k_ae}, Accuracy={ae_results[best_k_ae]['accuracy']:.4f}")
print(f"Best UMAP: k={best_k_umap}, Accuracy={umap_results[best_k_umap]['accuracy']:.4f}")

# Accuracy comparison
k_vals = list(ae_results.keys())
ae_accuracies = [ae_results[k]['accuracy'] for k in k_vals]
umap_accuracies = [umap_results[k]['accuracy'] for k in k_vals]


# Time comparison
ae_times = [ae_results[k]['time'] for k in k_vals]
umap_times = [umap_results[k]['time'] for k in k_vals]

print("FINAL ANALYSIS")
print(f"""
AUTOENCODER, UMAP
Preprocessing Time:    {ae_time:.1f}s                {umap_time:.1f}s
Best k:                {best_k_ae}                   {best_k_umap}
Best Accuracy:         {ae_results[best_k_ae]['accuracy']:.4f}              {umap_results[best_k_umap]['accuracy']:.4f}
k-NN Time (best k):    {ae_results[best_k_ae]['time']:.2f}s               {umap_results[best_k_umap]['time']:.2f}s
""")