import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

from features import extraction as fe


class AudioClassifier:
    def __init__(self, learner):
        self.feature_names = None
        self.class_names = None
        self.train_features = None
        self.train_labels = None
        self.train_predictions = None
        self.inner_learner = learner

    def extract_features(self, directories, feature_mask):
        """
        Extracts features from audio files in the specified directories.
        :param directories: list of directories. Each directory contains a single audio class whose samples
                            are stored in separate WAV files
        :param feature_mask: boolean mask for feature selection
        :return: None
        """
        [features, class_names, _, self.train_labels] = fe.extract_from_dirs(directories, feature_mask)
        self.train_features = fe.normalize(features)
        self.class_names = np.array([d.split(os.sep)[-1] for d in class_names])
        self.feature_names = fe.FEATURE_NAMES[feature_mask]

    def train_model(self, directories, new_model_name, from_file=None, feature_mask=None, to_file=None):
        """
        Trains the learner, using the files specified in the directories. Alternatively, a file with existing features
        can be provided for training.
        :param directories: directories of audio files to train on
        :param new_model_name: model name to use for pickling
        :param from_file: path to existing features to train with
        :param feature_mask: boolean mask for feature selection
        :param to_file: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if from_file is None:
            print('Extracting features')
            self.extract_features(directories, feature_mask)
            if to_file is not None:
                fe.save_data(to_file, self.train_features, self.feature_names, self.train_labels)
        else:
            # Only include features according to mask
            print('Loading features')
            [self.train_features, self.train_labels] = fe.load_data(from_file, feature_mask)
        # Fit the training data
        print('Fitting model')
        self.inner_learner.fit(self.train_features, self.train_labels)
        # Get and report the training accuracy of the classifier
        self.train_predictions = self.inner_learner.predict(self.train_features)
        print('Training accuracy: {}'.format(100 * np.mean(self.train_labels == self.train_predictions)))
        # Save the training data to a pickle
        self.save_model(new_model_name, self)

    @staticmethod
    def test_model(classifier, test_files_directory=None, to_file=None, from_file=None, feature_mask=None):
        """
        Test the generalization accuracy of the learner, using the test files in the specified directory.
        Alternatively, a path to existing extracted features can be provided for use in the testing.
        :param classifier: classifier to use for generalization testing
        :param test_files_directory: directory where test files are located
        :param to_file: path to save extract features to
        :param from_file: path to existing extracted features
        :param feature_mask: boolean mask for feature selection
        :return: 2-tuple of (predicted class labels, actual class labels)
        """
        if test_files_directory is not None:
            [test_features, _, _, y_test] = fe.extract_from_dirs(test_files_directory, feature_mask)
            test_features = np.array(test_features)
            if to_file is not None:
                fe.save_data(to_file, test_features, classifier.feature_names, y_test)
        elif from_file is not None:
            [test_features, y_test] = fe.load_data(from_file, feature_mask)
        else:
            raise Exception('Did not pass a test file directory or feature matrix to test_model()')

        y_predict = classifier.predict(test_features)
        return np.array(y_predict), np.array(y_test)

    def predict(self, x):
        """
        Invoke the sklearn-style predict() on the internal learner
        :param x: features to use for prediction
        :return: predicted class label
        """
        return self.inner_learner.predict(x)

    @staticmethod
    def classify_file(classifier, filename, feature_mask=None):
        """
        Obtain a class label output for the specified audio file
        :param classifier: classifier to use for prediction
        :param filename: audio file to predict class label for
        :param feature_mask: boolean mask for feature selection
        :return: class label prediction for specified audio file
        """
        [features, _] = fe.extract_from_file(filename, feature_mask)
        # TODO normalize
        return classifier.predict(features)

    @staticmethod
    def load_model(model_name):
        """
        Loads a learner model.
        :param model_name: name of model to load, without the ".model" extension
        :return: loaded model
        """
        classifier = None
        try:
            # Load data from the stored pickle file
            with open(model_name + ".model", "rb") as fh:
                classifier = pickle.load(fh)
        except IOError:
            raise Exception('Could not find model file to load: ' + model_name)
        return classifier

    @staticmethod
    def save_model(model_name, classifier):
        """
        Saves a learner model.
        :param model_name: model name to save, without the ".model" extension
        :param classifier: learner model to save
        :return: None
        """
        print("Saving classifier model to pickle")
        with open(model_name + ".model", "wb") as fh:
            pickle.dump(classifier, fh)


class GMMAudioClassifier(AudioClassifier):
    def __init__(self, learner):
        self.inner_learner = learner
        AudioClassifier.__init__(self, self.inner_learner)

    @staticmethod
    def classify_file(classifier, filename, feature_mask=None):
        """
        Obtain a class label output for the specified audio file
        :param classifier: classifier to use for prediction
        :param filename: audio file to predict class label for
        :param feature_mask: boolean mask for feature selection
        :return: class label prediction for specified audio file
        """
        return super().classify_file(classifier, filename, feature_mask)[0]

    def train_model(self, directories, new_model_name, from_file=None, feature_mask=None, to_file=None):
        """
        Trains the learner, using the files specified in the directories. Alternatively, a file with existing features
        can be provided for training.
        :param directories: directories of audio files to train on
        :param new_model_name: model name to use for pickling
        :param from_file: path to existing features to train with
        :param feature_mask: boolean mask for feature selection
        :param to_file: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if from_file is None:
            print('Extracting features')
            self.extract_features(directories, feature_mask)
            if to_file is not None:
                fe.save_data(to_file, self.train_features, self.feature_names, self.train_labels)
        else:
            # Only include features according to mask
            print('Loading features')
            [self.train_features, self.train_labels] = fe.load_data(from_file, feature_mask)
        print('Fitting model')
        # Set initial means for the mixture model
        self.inner_learner.means_ = np.array(
            [self.train_features[self.train_labels == self.class_names[i]].mean(axis=0)
             for i in range(self.inner_learner.n_components)])
        # Fit the other parameters of the mixture model using EM
        self.inner_learner.fit(self.train_features)
        # Reconcile the resulting Gaussian estimates to their corresponding labels
        indices = self.inner_learner.predict(self.train_features)
        modes = [stats.mode(indices[self.train_labels == self.class_names[i]]).mode[0]
                 for i in range(self.inner_learner.n_components)]
        self.class_names = self.class_names[modes]
        self.train_predictions = self.class_names[indices]
        print('Training accuracy: {}'.format(100 * np.mean(self.train_labels == self.train_predictions)))
        # Save the training data to a pickle
        print("Saving to " + new_model_name)
        self.save_model(new_model_name, self)
