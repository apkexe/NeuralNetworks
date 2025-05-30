import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preprocessing
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

# one-hot encoding
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = Sequential([
    Dense(256, input_dim=x_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

history = model.fit(
    x_train, y_train_onehot,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(x_test), axis=1)

# confusion matrix
cm = confusion_matrix(y_test.flatten(), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

# classification report
class_report = classification_report(y_test.flatten(), y_pred, target_names=labels)
per_class_accuracy = {}
for i, class_name in enumerate(labels):
    class_mask = y_test.flatten() == i
    class_true = y_test.flatten()[class_mask]
    class_pred = y_pred[class_mask]
    per_class_accuracy[class_name] = accuracy_score(class_true, class_pred)

print("\nPer-Class Accuracy:")
for class_name, accuracy in per_class_accuracy.items():
    print(f"{class_name}: {accuracy * 100:.2f}%")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

correct_indices = np.where(y_pred == y_test.flatten())[0]
incorrect_indices = np.where(y_pred != y_test.flatten())[0]


def plot_single_example(idx, title):
    plt.figure(figsize=(3, 3))
    plt.imshow(x_test[idx].reshape(32, 32, 3))  # img reshape
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.show()

if len(correct_indices) > 0:
    correct_idx = correct_indices[0]  # choose the 1st correct
    plot_single_example(
        correct_idx,
        title=f"Correct\nTrue: {labels[y_test.flatten()[correct_idx]]}\nPred: {labels[y_pred[correct_idx]]}"
    )
else:
    print("There are no correct predictions to show.")

if len(incorrect_indices) > 0:
    incorrect_idx = incorrect_indices[0]  # choose the 1st incorrect
    plot_single_example(
        incorrect_idx,
        title=f"Incorrect\nTrue: {labels[y_test.flatten()[incorrect_idx]]}\nPred: {labels[y_pred[incorrect_idx]]}"
    )
else:
    print("There are no incorrect predictions to show.")
