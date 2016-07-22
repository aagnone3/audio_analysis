from __future__ import print_function

import os
import pickle
from features import extraction as fe
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from scipy import stats


class AudioClassifier:
    def __init__(self, learner):
        self.feature_names = None
        self.class_names = None
        self.x_train = None
        self.per_frame_labels = None
        self.y_train_predict = None
        self.y_train_actual = None
        self.frame_indices = None
        self.inner_learner = learner

    def predict(self, x):
        """
        Invoke the sklearn-style predict() on the internal learner
        :param x: features to use for prediction
        :return: predicted class label
        """
        return self.inner_learner.predict(x)

    def train_model(self, data_directory, save_model_to, load_features_from=None, save_features_to=None):
        """
        Trains the learner, using the files specified in the data_directory. Alternatively, a file with existing features
        can be provided for training.
        :param data_directory: data_directory of audio files to train on
        :param save_model_to: model name to use for pickling
        :param load_features_from: path to existing features to train with
        :param save_features_to: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if load_features_from is None:
            print('Extracting features')
            [self.x_train, self.class_names, self.per_frame_labels, self.frame_indices] =\
                AudioClassifier.extract_features(data_directory)
            if save_features_to is not None:
                fe.save_data(save_features_to, self.x_train, self.per_frame_labels, self.frame_indices)
        else:
            print('Loading features')
            [self.x_train, self.per_frame_labels, self.frame_indices] = fe.load_data(load_features_from)

        # fit the training data
        print('Fitting model')
        self.inner_learner.fit(self.x_train, self.per_frame_labels)

        # get and report the training accuracy of the classifier
        n_samples = len(self.frame_indices) - 1
        self.y_train_predict = [None] * n_samples
        self.y_train_actual = [None] * n_samples
        for i in range(n_samples):
            beg = self.frame_indices[i]
            end = self.frame_indices[i + 1]
            frame_predictions = self.inner_learner.predict(self.x_train[beg:end, :])
            self.y_train_predict[i] = stats.mode(frame_predictions).mode[0]
            self.y_train_actual[i] = self.per_frame_labels[beg]

        # save the training data to a pickle
        self.save_model(save_model_to, self)

        return np.array(self.y_train_predict), np.array(self.y_train_actual)

    @staticmethod
    def test_model(classifier, data_directory=None, save_features_to=None,
                   load_features_from=None):
        """
        Test the generalization accuracy of the learner, using the test files in the specified directory.
        Alternatively, a path to existing extracted features can be provided for use in the testing.
        :param classifier: classifier to use for generalization testing
        :param data_directory: directory where test files are located
        :param save_features_to: path to save extract features to
        :param load_features_from: path to existing extracted features
        :return: 2-tuple of (predicted class labels, actual class labels)
        """
        if data_directory is not None:
            [x_test, _, frame_labels, frame_indices] = AudioClassifier.extract_features(data_directory)
            x_test = np.array(x_test)
            if save_features_to is not None:
                fe.save_data(save_features_to, x_test, frame_labels, frame_indices)
        elif load_features_from is not None:
            [x_test, frame_labels] = fe.load_data(load_features_from)
        else:
            raise Exception('Must pass either a test file directory or path to existing classifier model.')

        n_samples = len(frame_indices) - 1
        y_test_predict = [None] * n_samples
        y_test_actual = [None] * n_samples
        for i in range(n_samples):
            beg = frame_indices[i]
            end = frame_indices[i+1]
            frame_predictions = classifier.predict(x_test[beg:end, :])
            y_test_predict[i] = stats.mode(frame_predictions).mode[0]
            y_test_actual[i] = frame_labels[beg]
        return np.array(y_test_predict), np.array(y_test_actual)

    @staticmethod
    def extract_features(directories):
        """
        Extracts features from audio files in the specified directories.
        :param directories: list of directories. Each directory contains a single audio class whose samples
                            are stored in separate WAV files
        :return: None
        """
        [x_train, class_names, per_frame_labels, frame_indices] = fe.extract_from_dir(directories)
        x_train = RandomizedPCA().fit_transform(x_train)
        return x_train, class_names, per_frame_labels, frame_indices

    @staticmethod
    def classify_file(classifier, filename):
        """
        Obtain a class label output for the specified audio file
        :param classifier: classifier to use for prediction
        :param filename: audio file to predict class label for
        :return: class label prediction for specified audio file
        """
        [features, _] = fe.extract_from_file(filename)
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
    def classify_file(classifier, filename):
        """
        Obtain a class label output for the specified audio file
        :param classifier: classifier to use for prediction
        :param filename: audio file to predict class label for
        :return: class label prediction for specified audio file
        """
        return AudioClassifier.classify_file(classifier, filename)[0]

    def train_model(self, data_directory, save_model_to, load_features_from=None, save_features_to=None):
        """
        Trains the learner, using the files specified in the data_directory. Alternatively, a file with existing features
        can be provided for training.
        :param data_directory: data_directory of audio files to train on
        :param save_model_to: model name to use for pickling
        :param load_features_from: path to existing features to train with
        :param save_features_to: path to save extracted features to
        :return: None
        """
        # Obtain features through extraction or the given file
        if load_features_from is None:
            print('Extracting features')
            [self.x_train, self.class_names, self.per_frame_labels, self.frame_indices] =\
                AudioClassifier.extract_features(data_directory)
            if save_features_to is not None:
                fe.save_data(save_features_to, self.x_train, self.per_frame_labels, self.frame_indices)
        else:
            print('Loading features')
            [self.x_train, self.per_frame_labels, self.frame_indices] = fe.load_data(load_features_from)
        print('Fitting model')

        # set initial means for the mixture model
        self.inner_learner.means_ = np.array(
            [self.x_train[self.per_frame_labels == self.class_names[i]].mean(axis=0)
             for i in range(self.inner_learner.n_components)])

        # fit the other parameters of the mixture model using EM
        self.inner_learner.fit(self.x_train)

        # reconcile the resulting Gaussian estimates to their corresponding labels
        indices = self.inner_learner.predict(self.x_train)
        modes = [stats.mode(indices[self.train_labels == self.class_names[i]]).mode[0]
                 for i in range(self.inner_learner.n_components)]
        self.class_names = self.class_names[modes]
        self.y_train_predict = self.class_names[indices]
        print('Training accuracy: {}'.format(100 * np.mean(self.train_labels == self.y_train_predict)))

        # save the training data to a pickle
        print("Saving to " + save_model_to)
        self.save_model(save_model_to, self)
