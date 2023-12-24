```python
# Module 2: Machine Learning Essentials

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class MachineLearningEssentials:
    def __init__(self, data):
        self.data = data
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

    def data_split(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def data_scaling(self, X_train, X_test):
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def model_training(self, model, X_train, y_train):
        # Train the model
        model.fit(X_train, y_train)

        return model

    def model_evaluation(self, model, X_test, y_test):
        # Predict the test set results
        y_pred = model.predict(X_test)

        # Calculate the accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return accuracy, cm

    def hyperparameter_tuning(self, model, param_grid):
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid_search.fit(self.X, self.y)

        return grid_search.best_params_

if __name__ == "__main__":
    from module1.foundations_of_data_science import DataScienceFoundations

    ds = DataScienceFoundations('data.csv')
    cleaned_data = ds.data_cleaning()
    normalized_data = ds.data_preprocessing()

    ml = MachineLearningEssentials(normalized_data)
    X_train, X_test, y_train, y_test = ml.data_split()
    X_train_scaled, X_test_scaled = ml.data_scaling(X_train, X_test)

    # Logistic Regression
    lr = LogisticRegression()
    trained_lr = ml.model_training(lr, X_train_scaled, y_train)
    lr_accuracy, lr_cm = ml.model_evaluation(trained_lr, X_test_scaled, y_test)
    print("Logistic Regression Accuracy:\n", lr_accuracy)
    print("Logistic Regression Confusion Matrix:\n", lr_cm)

    # Random Forest
    rf = RandomForestClassifier()
    trained_rf = ml.model_training(rf, X_train_scaled, y_train)
    rf_accuracy, rf_cm = ml.model_evaluation(trained_rf, X_test_scaled, y_test)
    print("Random Forest Accuracy:\n", rf_accuracy)
    print("Random Forest Confusion Matrix:\n", rf_cm)

    # Support Vector Machine
    svm = SVC()
    trained_svm = ml.model_training(svm, X_train_scaled, y_train)
    svm_accuracy, svm_cm = ml.model_evaluation(trained_svm, X_test_scaled, y_test)
    print("SVM Accuracy:\n", svm_accuracy)
    print("SVM Confusion Matrix:\n", svm_cm)

    # Hyperparameter Tuning for Random Forest
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
    best_params = ml.hyperparameter_tuning(rf, param_grid)
    print("Best Parameters for Random Forest:\n", best_params)
```
