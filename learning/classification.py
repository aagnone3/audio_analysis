from __future__ import print_function

import os
import pickle
from features import extraction as fe
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


class AudioClassifier:
    def __init__(self, learner):
        self.feature_names = None
        self.class_names = None
        self.train_features = None
        self.frame_labels = None
        self.train_predictions = None
        self.sample_indices = None
        self.inner_learner = learner

    def extract_features(self, directories, feature_mask):
        """
        Extracts features from audio files in the specified directories.
        :param directories: list of directories. Each directory contains a single audio class whose samples
                            are stored in separate WAV files
        :param feature_mask: boolean mask for feature selection
        :return: None
        """
        [self.train_features, self.class_names, _, self.frame_labels,
         self.sample_indices] = fe.extract_from_dir(directories, feature_mask)
        self.feature_names = fe.FEATURE_NAMES[feature_mask]

    def predict(self, x):
        """
        Invoke the sklearn-style predict() on the internal learner
        :param x: features to use for prediction
        :return: predicted class label
        """
        return self.inner_learner.predict(x)

    def train_model(self, directories, new_model_name, load_features_from=None,
                    feature_mask=None, save_features_to=None):
        """
        Trains the learner, using the files specified in the directories. Alternatively, a file with existing features
        can be provided for training.
        :param directories: directories of audio files to train on
        :param new_model_name: model name to use for pickling
        :param load_features_from: path to existing features to train with
        :param feature_mask: boolean mask for feature selection
        :param save_features_to: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if load_features_from is None:
            print('Extracting features')
            self.extract_features(directories, feature_mask)
            if save_features_to is not None:
                fe.save_data(save_features_to, self.train_features, self.feature_names, self.frame_labels)
        else:
            # Only include features according to mask
            print('Loading features')
            [self.train_features, self.frame_labels] = fe.load_data(load_features_from, feature_mask)
        # Fit the training data
        print('Fitting model')
        self.inner_learner.fit(self.train_features, self.frame_labels)
        # Get and report the training accuracy of the classifier
        self.train_predictions = []
        labels = []
        for i in range(len(self.sample_indices)-1):
            beg = self.sample_indices[i]
            end = self.sample_indices[i+1]
            labels.append(self.frame_labels[beg])
            frame_predictions = self.inner_learner.predict(self.train_features[beg:end, :])
            self.train_predictions.append(stats.mode(frame_predictions).mode[0])
        # Save the training data to a pickle
        self.save_model(new_model_name, self)
        return np.array(self.train_predictions), np.array(labels)

    @staticmethod
    def test_model(classifier, test_files_directory=None, save_features_to=None,
                   load_features_from=None, feature_mask=None):
        """
        Test the generalization accuracy of the learner, using the test files in the specified directory.
        Alternatively, a path to existing extracted features can be provided for use in the testing.
        :param classifier: classifier to use for generalization testing
        :param test_files_directory: directory where test files are located
        :param save_features_to: path to save extract features to
        :param load_features_from: path to existing extracted features
        :param feature_mask: boolean mask for feature selection
        :return: 2-tuple of (predicted class labels, actual class labels)
        """
        if test_files_directory is not None:
            [test_features, _, _, frame_labels, sample_indices] = fe.extract_from_dir(test_files_directory, feature_mask)
            test_features = np.array(test_features)
            if save_features_to is not None:
                fe.save_data(save_features_to, test_features, classifier.feature_names, frame_labels)
        elif load_features_from is not None:
            [test_features, frame_labels] = fe.load_data(load_features_from, feature_mask)
        else:
            raise Exception('Did not pass a test file directory or feature matrix to test_model()')

        predictions = []
        labels = []
        for i in range(len(sample_indices) - 1):
            beg = sample_indices[i]
            end = sample_indices[i+1]
            frame_predictions = classifier.predict(test_features[beg:end, :])
            predictions.append(stats.mode(frame_predictions).mode[0])
            labels.append(frame_labels[beg])
        return np.array(predictions), np.array(labels)

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
        return classifier.predict(features)

    @staticmethod
    def load_model(model_name):
        """
        Loads a learner model.
        :param model_name: name of model to load, without the ".model" extension
        :return: loaded model
        """
        try:
            # Load data from the stored pickle file
            with open(model_name + ".model", "rb") as fh:
                classifier = pickle.load(fh)
            return classifier
        except IOError:
            raise Exception('Could not find model file to load: ' + model_name)

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
        return AudioClassifier.classify_file(classifier, filename, feature_mask)[0]

    def train_model(self, directories, new_model_name, load_features_from=None, feature_mask=None, save_features_to=None):
        """
        Trains the learner, using the files specified in the directories. Alternatively, a file with existing features
        can be provided for training.
        :param directories: directories of audio files to train on
        :param new_model_name: model name to use for pickling
        :param load_features_from: path to existing features to train with
        :param feature_mask: boolean mask for feature selection
        :param save_features_to: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if load_features_from is None:
            print('Extracting features')
            self.extract_features(directories, feature_mask)
            if save_features_to is not None:
                fe.save_data(save_features_to, self.train_features, self.feature_names, self.frame_labels)
        else:
            # Only include features according to mask
            print('Loading features')
            [self.train_features, self.frame_labels] = fe.load_data(load_features_from, feature_mask)
        print('Fitting model')
        # Set initial means for the mixture model
        self.inner_learner.means_ = np.array(
            [self.train_features[self.frame_labels == self.class_names[i]].mean(axis=0)
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
