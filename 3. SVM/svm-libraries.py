import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import time
import matplotlib.pyplot as plt

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize the dataset between 0 and 1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # uncomment the following for testing purposes
    # X_train = X_train[:1000]
    # X_test = X_test [:1000]
    # y_train = y_train[:1000]
    # y_test = y_test[:1000]


    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return X_train, y_train, X_test, y_test

def pca_feature_selection(X_train, y_train, X_test):
    # use PCA for dimensionality reduction
    pca = IncrementalPCA(n_components=100, batch_size=500)  # Batch processing because there was RAM overflow

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # fit PCA on the training data
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    print('Original feature size:', X_train_flat.shape[1])
    print('Reduced feature size (PCA):', X_train_pca.shape[1])

    # feature selection using mutual information
    # selector = SelectKBest(mutual_info_classif, k=100)
    # X_train_selected = selector.fit_transform(X_train_flat, y_train)
    # X_test_selected = selector.transform(X_test_flat)

    # print('Selected feature size (SelectKBest):', X_train_selected.shape[1])

    return X_train_pca, X_test_pca

# def perform_grid_search(X_train, y_train):
#   # perform grid search for hyperparameter tuning
#   param_grid = {
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'C': [0.1, 1, 10],
#     'gamma': ['scale', 'auto'],
#     'decision_function_shape': ['ovo', 'ovr'],
#     'verbose' : [True]
#     }
#   grid_search = GridSearchCV(SVC(random_state = 42), param_grid, cv = 3, scoring='accuracy', n_jobs = -1)
#   grid_search.fit(X_train, y_train)

#   print('Best parameters from Grid Search:', grid_search.best_params_)
#   return grid_search.best_params_

def visualize_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = labels)
    disp.plot(cmap ='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_predictions(X_data, y_true, y_pred, num_samples = 4):
    incorrect_indices = np.where(y_true != y_pred)[0]
    correct_indices = np.where(y_true == y_pred)[0]

    # choose the number of samples to visualise
    num_incorrect = min(num_samples, len(incorrect_indices))
    num_correct = min(num_samples, len(correct_indices))

    fig, axes = plt.subplots(2, max(num_incorrect, num_correct), figsize=(15, 6))
    fig.suptitle('Predictions', fontsize=16)

    # incorrect predictions in the first row
    for i in range(max(num_incorrect, num_correct)):
        if i < num_incorrect:
            incorrect_idx = incorrect_indices[i]
            axes[0, i].imshow(X_data[incorrect_idx].reshape(32, 32, 3))
            axes[0, i].set_title(f'Incorrect\nTrue: {labels[y_true[incorrect_idx]]}\nPred: {labels[y_pred[incorrect_idx]]}')
        else:
            axes[0, i].axis('off')

        axes[0, i].axis('off')

    # correct predictions in the second row
    for i in range(max(num_incorrect, num_correct)):
        if i < num_correct:
            correct_idx = correct_indices[i]
            axes[1, i].imshow(X_data[correct_idx].reshape(32, 32, 3))
            axes[1, i].set_title(f'Correct\nTrue: {labels[y_true[correct_idx]]}\nPred: {labels[y_pred[correct_idx]]}')
        else:
            axes[1, i].axis('off')

        axes[1, i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# def explain_with_shap(model, original_data, transformed_data, sample_indices):
#     def model_wrapper(data):
#         return model.decision_function(data)

#     explainer = shap.Explainer(model_wrapper, transformed_data)
#     shap_values = explainer(transformed_data)

#     # visualize shap values for selected samples using the original data
#     for index in sample_indices:
#         print(f'Explaining sample {index}...')
#         shap.image_plot([shap_values[index].values], original_data[index:index+1])


def evaluate_kernels(X_train, y_train, X_test, y_test):
    kernels = ['linear', 'poly', 'sigmoid']
    for kernel in kernels:
        print(f'\nEvaluating SVM with kernel={kernel}')
        if kernel == 'poly':
            model = SVC(kernel=kernel, C=1, degree=2, gamma='scale', random_state=42)
        else:
            model = SVC(kernel=kernel, C=1, gamma='scale', random_state=42)
        model.fit(X_train, y_train)

        # predictions and metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')

        # confusion matrix
        visualize_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
  start_time = time.time()

  X_train, y_train, X_test, y_test = load_and_preprocess_data()
  X_train_pca, X_test_pca = pca_feature_selection(X_train, y_train, X_test)

  X_train_pca, X_val_pca, y_train, y_val = train_test_split(
      X_train_pca, y_train, test_size=0.2, random_state=42
  )

  # best params found from GridSearch
  best_params = {
      'kernel': 'rbf',
      'C': 10,
      'gamma': 'auto',
      'decision_function_shape': 'ovo',
      'verbose': True
      }


  # train the final model with the best parameters
  final_model = SVC(**best_params, random_state=42, probability=True)
  final_model.fit(X_train_pca, y_train)

  # evaluate the final model on the test set
  y_test_pred = final_model.predict(X_test_pca)

  test_accuracy = accuracy_score(y_test, y_test_pred)
  print('\nFinal Test Accuracy:', test_accuracy)

  # visualize confusion matrix
  visualize_confusion_matrix(y_test, y_test_pred)

  # visualize predictions
  visualize_predictions(X_test, y_test, y_test_pred)

  # # explain predictions with SHAP
  # sample_indices = [0, 10, 20, 30]  # select few test samples to explain
  # explain_with_shap(final_model, X_test, X_test_pca, sample_indices)

  # evaluate with different kernels
  print('\nEvaluating different kernels:')
  evaluate_kernels(X_train_pca, y_train, X_test_pca, y_test)

  end_time = time.time()
  total_time = end_time - start_time
  print(f'Total Time (training + prediction): {total_time:.2f} seconds')