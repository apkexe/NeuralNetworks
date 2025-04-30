import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import time
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# one-hot encoding for Hinge Loss
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)), 
    layers.Dense(128, activation='relu'),    
    layers.Dense(num_classes, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='hinge',
    metrics=['accuracy']
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

start_time = time.time()

history = model.fit(
    x_train, y_train,
    epochs=50,           
    batch_size=256,        
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f"Training Time: {elapsed_time_minutes:.2f} minutes")

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
