```python
# Module 3: Deep Learning and Neural Networks

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class DeepLearningNeuralNetworks:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

    def build_model(self, input_shape):
        # Build a sequential model
        model = Sequential()

        # Add convolutional layer
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

        # Add max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add dropout layer
        model.add(Dropout(0.25))

        # Add flatten layer
        model.add(Flatten())

        # Add dense layer
        model.add(Dense(128, activation='relu'))

        # Add output layer
        model.add(Dense(10, activation='softmax'))

        return model

    def compile_model(self, model):
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

    def train_model(self, model):
        # Train the model
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=32)

        return history

    def evaluate_model(self, model):
        # Evaluate the model
        loss, accuracy = model.evaluate(self.X_test, self.y_test)

        return loss, accuracy

    def predict_model(self, model, data):
        # Predict the data
        predictions = model.predict(data)

        return predictions

if __name__ == "__main__":
    from module1.foundations_of_data_science import DataScienceFoundations
    from module2.machine_learning_essentials import MachineLearningEssentials

    ds = DataScienceFoundations('data.csv')
    cleaned_data = ds.data_cleaning()
    normalized_data = ds.data_preprocessing()

    ml = MachineLearningEssentials(normalized_data)
    X_train, X_test, y_train, y_test = ml.data_split()
    X_train_scaled, X_test_scaled = ml.data_scaling(X_train, X_test)

    dl = DeepLearningNeuralNetworks(X_train_scaled, y_train)
    model = dl.build_model((28, 28, 1))
    compiled_model = dl.compile_model(model)
    history = dl.train_model(compiled_model)
    loss, accuracy = dl.evaluate_model(compiled_model)

    print("Model Loss:\n", loss)
    print("Model Accuracy:\n", accuracy)
```
