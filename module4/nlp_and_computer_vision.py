```python
# Module 4: NLP and Computer Vision

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional

class NLPComputerVision:
    def __init__(self, text_data, image_data, labels):
        self.text_data = text_data
        self.image_data = image_data
        self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.text_data, self.labels, test_size=0.2, random_state=42)

    def build_text_model(self, vocab_size, embedding_dim, max_length):
        # Build a sequential model for text data
        model = Sequential()

        # Add embedding layer
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

        # Add bidirectional LSTM layer
        model.add(Bidirectional(LSTM(64)))

        # Add dense layer
        model.add(Dense(24, activation='relu'))

        # Add output layer
        model.add(Dense(1, activation='sigmoid'))

        return model

    def build_image_model(self, input_shape):
        # Build a sequential model for image data
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
    from module3.deep_learning_and_neural_networks import DeepLearningNeuralNetworks

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

    nlp_cv = NLPComputerVision(X_train_scaled, y_train)
    text_model = nlp_cv.build_text_model(10000, 16, 120)
    compiled_text_model = nlp_cv.compile_model(text_model)
    history = nlp_cv.train_model(compiled_text_model)
    loss, accuracy = nlp_cv.evaluate_model(compiled_text_model)

    print("Text Model Loss:\n", loss)
    print("Text Model Accuracy:\n", accuracy)
```
