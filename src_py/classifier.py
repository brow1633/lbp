import os
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np

from lbp_histogram import calc_lbp_rust, multi_lbp

class Classifier():
    def __init__(self, descriptor_func):
        """
        Initializes the classifier.

        Parameters:
        descriptor_func: A function that takes an image and returns its descriptor (e.g., SIFT, SURF, ORB).
        training_dir: Directory containing training images organized in subfolders for each class.
        """
        self.descriptor_func = descriptor_func
        self.classes = []
        self.training_features = None
        self.testing_features = None
        self.model = None
    
    def load_training_data(self, training_dir, save=False, load=False, filename_base='training'):
        self.training_features = self._calculate_data(training_dir, save=save, load=load, filename_base=filename_base)
    
    def load_testing_data(self, testing_dir, save=False, load=False, filename_base='testing'):
        self.testing_features = self._calculate_data(testing_dir, save=save, load=load, filename_base=filename_base)
    
    def _calculate_data(self, dir='', save=False, load=False, filename_base=''):
        # Populate the class names based on the folder names in the training directory
        self.classes = [foldername for foldername in os.listdir(dir) 
                        if os.path.isdir(os.path.join(dir, foldername))]

        if load:
            return self._load_features(dir, filename_base)

        features = {}
        # Iterate over each class
        for cls in self.classes:
            class_path = os.path.join(dir, cls)
            features[cls] = []

            # Iterate over each image file in the class folder
            for filename in os.listdir(class_path):
                if not filename.endswith('.jpg'):
                    continue
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)

                if image is not None:
                    descriptor = self.descriptor_func(image)
                    if descriptor is not None:
                        features[cls].append(descriptor)
                else:
                    print(f"Could not read image: {image_path}")

            # Save the features if requested
            if save:
                self._save_features(features, dir, filename_base)
                print(f"Features saved to {dir + filename_base + '_' + cls + '.npz'}")

        for cls, feats in features.items():
            print(f"Class: {cls}, Number of feature vectors: {len(feats)}")

        return features
        

    def _save_features(self, features, dir, filename):
        """
        Saves the features to a file.

        Parameters:
        filename (str): The name of the file to save the features.
        """
        # Convert dictionary to a format suitable for saving with numpy
        for (cls, feats) in features.items():
            np.savez_compressed(os.path.join(dir, filename + '_' + cls + '.npz'), features=np.array(feats))

    def _load_features(self, dir, filename_base='trained'):
        """
        Loads the features from separate files for each class.

        Parameters:
        filename_base (str): The base name of the files from which to load the features (without extension).
        """
        self.classes = [foldername for foldername in os.listdir(dir) 
                if os.path.isdir(os.path.join(dir, foldername))]

        features = {}
        for cls in self.classes:
            # Construct the filename for each class
            filename = os.path.join(dir, f"{filename_base}_{cls}.npz")
            if os.path.exists(filename):
                data = np.load(filename, allow_pickle=True)
                # Extract the 'features' array and convert it back to a list
                features[cls] = data['features'].tolist()
                print(f"Loaded features for class '{cls}' from {filename}")
            else:
                print(f"File not found for class '{cls}': {filename}")
        return features

    def train(self):
        """
        Trains an SVM using OpenCV's cv2.ml.SVM.
        """
        # Prepare the training data
        X = []
        y = []

        label_map = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls, descriptors in self.training_features.items():
            label = label_map[cls]
            for descriptor in descriptors:
                if descriptor is not None:
                    for d in descriptor:
                        X.append(d)
                        y.append(label)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32).reshape(-1, 1)

        # Initialize OpenCV SVM
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_RBF)  # Or cv2.ml.SVM_RBF for RBF kernel
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(1.0)

        # Train the SVM
        svm.train(X, cv2.ml.ROW_SAMPLE, y)
        self.model = svm

        print("SVM training complete.")
    
    def test(self):
        """
        Tests the loaded test data using the trained model and outputs a confusion matrix.
        """
        if not self.model:
            print("Model not trained. Please train the model before testing.")
            return

        if not self.testing_features:
            print("Testing features not loaded. Please load testing data before testing.")
            return

        X_test = []
        y_true = []
        label_map = {cls: idx for idx, cls in enumerate(self.classes)}

        # Prepare the test data
        for cls, descriptors in self.testing_features.items():
            label = label_map[cls]
            for descriptor in descriptors:
                if descriptor is not None:
                    for d in descriptor:
                        X_test.append(d)
                        y_true.append(label)

        # Convert to numpy arrays
        X_test = np.array(X_test, dtype=np.float32)

        # Predict labels for the test data
        y_pred = self.model.predict(X_test)[1].ravel()

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"Accuracy: {accuracy:.4f}")

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        # print("\nConfusion Matrix:")
        # print(conf_matrix)

        # Detailed classification report
        report = classification_report(y_true, y_pred, target_names=self.classes)
        # print("\nClassification Report:")
        # print(report)
        return accuracy

class ClassifierWithParameterSearch(Classifier):
    def __init__(self, descritpor_func, param_ranges, training_dir='training/', testing_dir='testing/'):
        self.param_ranges = np.array(param_ranges)
        init_params = self.param_ranges[:,0].tolist()
        super().__init__(lambda im: descriptor_func(im, *init_params))

        self.base_descriptor_func = descritpor_func
        self.best_params = init_params
        self.best_accuracy = 0

        self.training_dir = training_dir
        self.testing_dir = testing_dir
    
    def optimize(self, iterations=50):
        for i in range(iterations):
            params = [np.random.uniform(low, high) for low, high in zip(self.param_ranges[:, 0], self.param_ranges[:, 1])]

            self.descriptor_func = lambda im: self.base_descriptor_func(im, *params)
            self.load_training_data(self.training_dir)
            self.train()
            self.load_testing_data(self.testing_dir)
            accuracy = self.test()
            print(f"Testing {params}, Accuracy achieved: {accuracy}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params = params
        print(f"Best Accuracy: {self.best_accuracy}, with params: {self.best_params}")


def test(image):
    return image[0, 0, :]

if __name__ == '__main__':
    # classifier = Classifier(lambda im: calc_lbp_rust(im, 5, 25.0))
    # classifier.load_training_data('training/', save=False, load=False, filename_base='test')
    # classifier.train()
    # classifier.load_testing_data('testing/', save=False, load=False, filename_base='test')
    # classifier.test()
    param_ranges = [[21, 27], [1, 5.0], [21, 29], [1, 5.0]]
    optimizer = ClassifierWithParameterSearch(multi_lbp, param_ranges)
    optimizer.optimize()
