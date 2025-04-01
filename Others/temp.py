import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

# Define MLP model
def create_mlp():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Output layer for 10 digits
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_mlp()
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


from tensorflow.keras.layers import LeakyReLU

def experiment_mlp(activation_fn, loss_fn):
    print(f"\nTraining with Activation: {activation_fn}, Loss: {loss_fn}")

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    # First hidden layer
    model.add(Dense(128))
    if activation_fn == 'leaky_relu':
        model.add(LeakyReLU(alpha=0.01))
    else:
        model.add(Dense(128, activation=activation_fn))

    # Second hidden layer
    model.add(Dense(64))
    if activation_fn == 'leaky_relu':
        model.add(LeakyReLU(alpha=0.01))
    else:
        model.add(Dense(64, activation=activation_fn))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Compile model with variable loss function
    model.compile(optimizer=Adam(), loss=loss_fn, metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), verbose=0)

    # Evaluate performance
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc

# Activation and loss function combinations
activation_functions = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
loss_functions = ['categorical_crossentropy', 'mean_squared_error']

# Store results
results = {}

for activation in activation_functions:
    for loss in loss_functions:
        key = f"Act: {activation}, Loss: {loss}"
        results[key] = experiment_mlp(activation, loss)

# Display results
print("\nComparison of Activation & Loss Functions:")
for key, acc in results.items():
    print(f"{key} → Accuracy: {acc:.4f}")
