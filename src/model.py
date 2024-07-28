import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk

class LoanDefaultPredictor:
    def __init__(self, train_dir, test_dir, model_dir):
        """
        Initializes a new instance of the LoanDefaultPredictor class.

        Args:
            train_dir (str): The directory containing the training datasets.
            test_dir (str): The directory containing the testing datasets.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_dir = model_dir
        self.model = RandomForestClassifier(n_estimators=100)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def load_data(self):
        """
        Load the training and testing data from CSV files.

        Returns:
            None
        """
        self.X_train = pd.read_csv(os.path.join(self.train_dir, 'X_train.csv'))
        self.y_train = pd.read_csv(os.path.join(self.train_dir, 'y_train.csv'))
        self.X_test = pd.read_csv(os.path.join(self.test_dir, 'X_test.csv'))
        self.y_test = pd.read_csv(os.path.join(self.test_dir, 'y_test.csv'))

    def train_model(self):
        """
        Train the RandomForest model on the training data.

        Returns:
            None
        """
        self.model.fit(self.X_train, self.y_train.values.ravel())
        # For simulation, we'll use the model's score as accuracy and 1 - score as loss
        train_acc = self.model.score(self.X_train, self.y_train)
        val_acc = self.model.score(self.X_test, self.y_test)
        self.history['accuracy'].append(train_acc)
        self.history['val_accuracy'].append(val_acc)
        self.history['loss'].append(1 - train_acc)
        self.history['val_loss'].append(1 - val_acc)

    def make_predictions(self):
        """
        Make predictions on the test data.

        Returns:
            numpy.ndarray: The predicted labels for the test data.
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_model(self):
        """
        Evaluate the model by computing accuracy, classification report, and confusion matrix.

        Returns:
            tuple: A tuple containing the accuracy, classification report, and confusion matrix.
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, report, matrix

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix using seaborn.

        Returns:
            None
        """
        matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def plot_training_history(self):
        '''
        Plot the training history of the model.

        Returns:
            None
        '''
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.savefig('training_history.png')
        plt.close()

    def save_model(self):
        """
        Save the trained model with a unique name.

        Returns:
            None
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_')]
        model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
        next_model_number = max(model_numbers, default=0) + 1
        model_filename = os.path.join(self.model_dir, f'model_{next_model_number}.pkl')
        pk.dump(self.model, open(model_filename, 'wb'))
        # print(f'Model saved as {model_filename}')


if __name__ == "__main__":
    # Setting the default parameters
    train_dir = '../data/train'
    test_dir = '../data/test'
    model_dir = '../models'

    predictor = LoanDefaultPredictor(train_dir, test_dir, model_dir)

    # Load data
    predictor.load_data()

    # Train the model
    predictor.train_model()

    # Make predictions
    predictor.make_predictions()

    # Evaluate the model
    accuracy, report, matrix = predictor.evaluate_model()
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', matrix)

    # Plot the confusion matrix
    predictor.plot_confusion_matrix()

    # Save the model
    predictor.save_model()
