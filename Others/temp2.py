import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load & preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encoding

# MLP Model Function
def create_mlp(activation='relu', loss='categorical_crossentropy'):
    model = Sequential([Flatten(input_shape=(28, 28))])
    
    for units in [128, 64]:  # Two hidden layers
        model.add(Dense(units))
        model.add(LeakyReLU(alpha=0.01) if activation == 'leaky_relu' else Dense(units, activation=activation))
    
    model.add(Dense(10, activation='softmax'))  # Output layer
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
    return model

# Train & Evaluate Default Model
model = create_mlp()
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), verbose=1)
print(f"Test Accuracy: {model.evaluate(x_test, y_test, verbose=0)[1]:.4f}")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Experiment with Different Activations & Loss Functions
activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
losses = ['categorical_crossentropy', 'mean_squared_error']
results = {f"{act}, {loss}": create_mlp(act, loss).fit(x_train, y_train, epochs=5, batch_size=32, 
             validation_data=(x_test, y_test), verbose=0).history['val_accuracy'][-1] 
             for act in activations for loss in losses}

# Display Results
print("\nActivation & Loss Function Impact on Accuracy:")
for key, acc in results.items():
    print(f"{key} → Accuracy: {acc:.4f}")
