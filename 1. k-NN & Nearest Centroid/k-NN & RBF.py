import numpy as np
from tensorflow.keras.datasets import cifar10
import time

print("Loading the CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train[:5000], y_train[:5000], x_test[:1000], y_test[:1000]

# transforing to 1D and data normalisation
print("Preparing the data...")
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class SimpleKNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        print(f"Fitted KNN with {X.shape[0]} training samples")
    
    def predict(self, X, batch_size=100):
        predictions = []
        num_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X))
            batch_X = X[start_idx:end_idx]
            batch_predictions = []
            
            print(f"Processing batch {batch_idx + 1}/{num_batches}")
            for x in batch_X:
                distances = []
                for x_train in self.X_train:
                    dist = euclidean_distance(x, x_train)
                    distances.append(dist)
                
                k_indices = np.argsort(distances)[:self.k]
                k_nearest_labels = self.y_train[k_indices]
                
                # voting
                most_common = np.bincount(k_nearest_labels.flatten()).argmax()
                batch_predictions.append(most_common)
            
            predictions.extend(batch_predictions)
            
        return np.array(predictions)

class SimpleNearestCentroid:
    def fit(self, X, y):
        self.centroids = {}
        self.classes = np.unique(y)
        
        print("Calculating centroids...")
        # centroid for each class
        for c in self.classes:
            self.centroids[c] = np.mean(X[y.flatten() == c], axis=0)
            print(f"Calculated centroid for class {class_names[c]}")
    
    def predict(self, X, batch_size=100):
        predictions = []
        num_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X))
            batch_X = X[start_idx:end_idx]
            batch_predictions = []
            
            print(f"Processing batch {batch_idx + 1}/{num_batches}")
            for x in batch_X:
                # nearest centroid calculation
                distances = []
                for c in self.classes:
                    dist = euclidean_distance(x, self.centroids[c])
                    distances.append(dist)
                batch_predictions.append(np.argmin(distances))
            
            predictions.extend(batch_predictions)
            
        return np.array(predictions)

def print_per_class_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    print("\nPer-class accuracy:")
    for c in classes:
        mask = (y_true.flatten() == c)
        class_acc = np.mean(y_pred[mask] == y_true.flatten()[mask]) * 100
        print(f"{class_names[c]}: {class_acc:.2f}%")

# # subset of training data for faster testing
# subset_size = 5000  
# test_subset_size = 1000

# print(f"\nUsing {subset_size} training samples and {test_subset_size} test samples")
# x_train_subset = x_train[:subset_size]
# y_train_subset = y_train[:subset_size]
# x_test_subset = x_test[:test_subset_size]
# y_test_subset = y_test[:test_subset_size]

# knn for k=3
print("\nTesting KNN with k=3")
knn3 = SimpleKNN(k=1)
start_time = time.time()
knn3.fit(x_train, y_train)
y_pred = knn3.predict(x_test)
accuracy = np.mean(y_pred == y_test.flatten()) * 100
print(f"Overall accuracy: {accuracy:.2f}%")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
print_per_class_accuracy(y_test, y_pred)

# Nearest Centroid
# print("\nTesting Nearest Centroid")
# nc = SimpleNearestCentroid()
# start_time = time.time()
# nc.fit(x_train, y_train)
# y_pred = nc.predict(x_test)
# accuracy = np.mean(y_pred == y_test.flatten()) * 100
# print(f"Overall accuracy: {accuracy:.2f}%")
# print(f"Time taken: {time.time() - start_time:.2f} seconds")
# print_per_class_accuracy(y_test, y_pred)