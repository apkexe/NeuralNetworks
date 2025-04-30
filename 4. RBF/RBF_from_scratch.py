import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score
import time


# PCA for dimensionality reduction and RAM saving
def apply_pca(X, variance_retained = 0.90):
    pca = PCA(variance_retained)
    X_new = pca.fit_transform(X)
    return X_new, pca

# Radial Basis Function class
class RBFNetwork:
  def __init__(self, num_centres, num_classes, learning_rate = 0.01):
    self.num_centres = num_centres
    self.num_classes = num_classes
    self.centres = None
    self.spreads = None
    self.output_weights = None
    # initialise weights randomly
    self.output_weights = np.random.randn(num_centres, num_classes)
    self.learning_rate = learning_rate


  def _kmeans(self, X, max_iters = 100):
    # ranom centres from within the dataset
    centres = X[np.random.choice(X.shape[0], self.num_centres, replace = False)]

    for _ in range(max_iters):
        # calculating the distance of a point from all centres
        distances = np.linalg.norm(X[:, np.newaxis] - centres, axis=2)
        labels = np.argmin(distances, axis=1)  # Label = label of the centre with the minimum distance

        # updating the centres as mean of the points belonging to their class
        new_centres = np.array([X[labels == i].mean(axis=0) for i in range(self.num_centres)])

        if np.allclose(centres, new_centres, atol=1e-6):
            break
        centres = new_centres

    return centres

  def _compute_spreads(self):
      # calculating the maximum distance between any two centres
      # I need it for the spread calculation below
      max_distance = 0
      for i in range(self.num_centres):
          for j in range(i + 1, self.num_centres):
              dist = np.linalg.norm(self.centres[i] - self.centres[j])
              max_distance = max(max_distance, dist)

      # spread calculation
      spread = max_distance / np.sqrt(2 * self.num_centres)
      spreads = np.full(self.num_centres, spread)  # same spread for all centres

      return spreads

  def _rbf(self, x, center, spread):
    return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * spread ** 2))

  def _rbf_layer(self, X):
    RBF_activation_array = np.zeros((X.shape[0], self.num_centres))
    for i, (center, spread) in enumerate(zip(self.centres, self.spreads)):
        RBF_activation_array[:, i] = np.apply_along_axis(self._rbf, 1, X, center, spread)
    return RBF_activation_array

  def fit(self, X, y, epochs = 100):
    self.centres = self._kmeans(X)
    self.spreads = self._compute_spreads()
    loss_values = []
    accuracy_values = []

    for epoch in range(epochs):
      RBF_activation_array = self._rbf_layer(X)
      outputs = RBF_activation_array.dot(self.output_weights)

      # weight update using delta rule
      error = y - outputs  # error (d(k)- y(k))
      weight_updates = self.learning_rate * RBF_activation_array.T.dot(error) # β(k) * a(k) * error 
      self.output_weights += weight_updates

      # loss and accuracy
      predictions = np.argmax(outputs, axis=1)
      actual = np.argmax(y, axis=1)
      loss = np.mean(-np.log(np.clip(np.sum(outputs * y, axis=1), 1e-15, 1 - 1e-15)))
      accuracy = np.mean(predictions == actual) * 100

      loss_values.append(loss)
      accuracy_values.append(accuracy)

      print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Plot the loss and accuracy over epochs
    plt.figure(figsize = (12, 6))
    plt.plot(range(1, epochs + 1), accuracy_values, label = 'Train Accuracy', color = 'blue', marker = 'o')
    plt.plot(range(1, epochs + 1), loss_values, label='Loss', color = 'red', marker = 'x')
    plt.title('Training Progress: Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

    return loss_values, accuracy_values


  def predict(self, X):
    RBF_activation_array = self._rbf_layer(X)
    outputs = RBF_activation_array.dot(self.output_weights)
    return np.argmax(outputs, axis=1)

# one-hot encoding
def one_hot_encode(y, num_classes):
  one_hot = np.zeros((y.shape[0], num_classes))
  one_hot[np.arange(y.shape[0]), y.flatten()] = 1
  return one_hot

# per-class accuracy
def print_per_class_accuracy(y_true, y_pred):
  print('\nPer-class accuracy:')
  for i, class_name in enumerate(class_names):
    mask = (y_true.flatten() == i)
    class_acc = np.mean(y_pred[mask] == y_true.flatten()[mask]) * 100
    print(f'{class_name}: {class_acc:.2f}%')

# visualise correct and false predictions
def visualize_predictions(x_data, y_true, y_pred, class_names, num_samples=4):
    false_indices = np.where(y_true != y_pred)[0]
    correct_indices = np.where(y_true == y_pred)[0]

    num_false = min(num_samples, len(false_indices))
    num_correct = min(num_samples, len(correct_indices))

    fig, axes = plt.subplots(2, max(num_false, num_correct), figsize=(15, 6))
    fig.suptitle("Predictions", fontsize=16)

    for i in range(max(num_false, num_correct)):
        if i < num_false:
            false_idx = false_indices[i]
            axes[0, i].imshow(x_data[false_idx].reshape(32, 32, 3))
            axes[0, i].set_title(f"False\nTrue: {class_names[y_true[false_idx]]}\nPred: {class_names[y_pred[false_idx]]}")
        else:
            axes[0, i].axis("off")

        if i < num_correct:
            correct_idx = correct_indices[i]
            axes[1, i].imshow(x_data[correct_idx].reshape(32, 32, 3))
            axes[1, i].set_title(f"Correct\nTrue: {class_names[y_true[correct_idx]]}\nPred: {class_names[y_pred[correct_idx]]}")
        else:
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == '__main__':
    start = time.time()
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    X_train_new, pca = apply_pca(X_train)
    X_test_new = pca.transform(X_test)

    # uncomment the following lines to test with a smaller dataset
    # X_train_new = X_train_new[:1000]
    # y_train = y_train[:1000]
    # X_test_new = X_test_new[:1000]
    # y_test = y_test[:1000]

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = 10
    input_size = X_train_new.shape[1]  # new dimensionality 
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)
    num_centres = 100
    
    print(f'\nTesting with {num_centres} hidden neurons')

    # train and evaluate RBF network
    rbf_net = RBFNetwork(num_centres=num_centres, num_classes=num_classes)
    rbf_net.fit(X_train_new, y_train_onehot)

    train_predictions = rbf_net.predict(X_train_new)
    test_predictions = rbf_net.predict(X_test_new)

    train_accuracy = accuracy_score(y_train.flatten(), train_predictions) * 100
    test_accuracy = accuracy_score(y_test.flatten(), test_predictions) * 100

    print(f'Train Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    print_per_class_accuracy(y_test, test_predictions)
    end = time.time()
    seconds = end - start
    minutes = seconds/60
    print(f'Time elapsed: {seconds:.2f} seconds or {minutes:.2f} minutes')

""" Uncomment the following if you want to test with different number of hidden neurons """
# hidden_neurons_list = [50, 100, 150, 200]
#     hidden_neurons_list = [50, 100, 150, 200]
#     for num_centres in hidden_neurons_list:
#         print(f'\nTesting with {num_centres} hidden neurons')

#         # train and evaluate RBF network
#         rbf_net = RBFNetwork(num_centres=num_centres, num_classes=num_classes)
#         rbf_net.fit(X_train_new, y_train_onehot)

#         train_predictions = rbf_net.predict(X_train_new)
#         test_predictions = rbf_net.predict(X_test_new)

#         train_accuracy = accuracy_score(y_train.flatten(), train_predictions) * 100
#         test_accuracy = accuracy_score(y_test.flatten(), test_predictions) * 100

#         print(f'Train Accuracy: {train_accuracy:.2f}%')
#         print(f'Test Accuracy: {test_accuracy:.2f}%')

#         print_per_class_accuracy(y_test, test_predictions)
