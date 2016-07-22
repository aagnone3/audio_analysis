# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:55:34 2016

@author: aagnone3
"""
from __future__ import print_function
import csv
import glob
import logging
import os
from learning.classification import AudioClassifier, GMMAudioClassifier
from common import environment
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB

# data directories and paths
DATA_DIR = os.sep.join(['res', 'data', 'speaker_recognition'])
TRAIN_DATA_DIR = os.sep.join([DATA_DIR, 'train'])
TEST_DATA_DIR = os.sep.join([DATA_DIR, 'test'])

# training file information
TRAIN_FILES = glob.glob(os.sep.join((TRAIN_DATA_DIR, "*.wav")))
TRAIN_FILE_PREFIXES = [f[:f.find("_")] for f in TRAIN_FILES]
TRAIN_FILE_COUNTS = [TRAIN_FILE_PREFIXES.count(p) for p in np.unique(TRAIN_FILE_PREFIXES)]

# testing file information
TEST_FILES = glob.glob(os.sep.join((TEST_DATA_DIR, "*.wav")))
TEST_FILE_PREFIXES = [f[:f.find("_")] for f in TEST_FILES]
TEST_FILE_COUNTS = [TEST_FILE_PREFIXES.count(p) for p in np.unique(TEST_FILE_PREFIXES)]

# keywords for naming
KEYWORD = 'Test'
ALGORITHM = "audio_classification"


def evaluate_classifier(paths, learner, extract_features=True):
    """
    Performs training and testing with the specified learner by extracting features from the files in the training
    and testing directories specified globally
    :param paths: paths to save learning results
    :param learner: learner to use
    :param extract_features: whether to extract features from the training and testing directories. If set to False,
        the classifier assumes that training and testing feature matrices exist at paths['train_data'] and
        paths['test_data'], respectively
    :return: None
    """
    # instantiate the classifier with the passed learner
    classifier = AudioClassifier(learner)

    # train the classifier
    if extract_features:
        [y_train_predict, y_train_actual] = classifier.train_model(data_directory=TRAIN_DATA_DIR,
                                                                   save_model_to=paths['model_name'],
                                                                   save_features_to=paths['train_data'])
    else:
        [y_train_predict, y_train_actual] = classifier.train_model(data_directory=TRAIN_DATA_DIR,
                                                                   save_model_to=paths['model_name'],
                                                                   load_features_from=paths['train_data'])

    # test the classifier
    if extract_features:
        [y_test_predict, y_test_actual] = AudioClassifier.test_model(
            classifier=AudioClassifier.load_model(paths['model_name']),
            data_directory=TEST_DATA_DIR,
            save_features_to=paths['test_data'])
    else:
        [y_test_predict, y_test_actual] = AudioClassifier.test_model(
            classifier=AudioClassifier.load_model(paths['model_name']),
            load_features_from=paths['test_data'])

    process_results(y_train_predict, y_train_actual, paths['train_results'])
    process_results(y_test_predict, y_test_actual, paths['test_results'])


def process_results(y_predict, y_actual, results_file_name):
    """
    Prints a simple analysis of test results to the console
    :param y_predict: predicted class labels
    :param y_actual: actual class labels
    :param results_file_name: path to use for saving off results
    :return: None
    """
    environment.print_lines(1)
    # compute simple statistics on the results
    columns = ['Overall Accuracy (%)']
    metrics = {
        'percents': [100.0 * np.mean(y_predict == y_actual)],
        'partials': [np.sum(y_predict == y_actual)],
        'wholes': [len(y_actual)]
    }
    for label in np.unique(y_actual):
        columns.append("%s Accuracy (%%)" % label)
        metrics['percents'].append(100.0 * np.mean(y_predict[y_actual == label] == y_actual[y_actual == label]))
        metrics['partials'].append(np.sum(y_predict[y_actual == label] == y_actual[y_actual == label]))
        metrics['wholes'].append(len(y_actual[y_actual == label]))

    # write the results to a file
    with open(results_file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(y_predict.shape[0]):
            writer.writerow([y_predict[i], y_actual[i]])
        csv_file.close()

    # print the results to the command line
    for i, percent in enumerate(metrics['percents']):
        print("{:22}:\t\t{:6.2f} %\t\t{:2d}/{:2d}".format(columns[i],
                                                          percent,
                                                          metrics['partials'][i],
                                                          metrics['wholes'][i]))

    return metrics, columns


if __name__ == '__main__':
    logging.basicConfig(filename='application.log',
                        level=logging.INFO,
                        format='%(asctime)s\t\t%(message)s')
    logging.info('Hello logging world')
    print("Training file counts {}".format(TRAIN_FILE_COUNTS))

    paths = {
        'model_name': os.sep.join(['res', 'models', ALGORITHM]),
        'train_results': os.sep.join(['res', 'results', ALGORITHM + '_train.csv']),
        'test_results': os.sep.join(['res', 'results', ALGORITHM + '_test.csv']),
        'train_data': os.sep.join(['res', 'features', '']) + 'TrainData_' + KEYWORD + '.pkl',
        'test_data': os.sep.join(['res', 'features', '']) + 'TestData_' + KEYWORD + '.pkl'
    }
    environment.print_lines(5)
    print("Testing file counts {}".format(TEST_FILE_COUNTS))

    # learner = mixture.GMM(n_components=len(TRAINING_DIRS))
    # learner = GaussianNB()
    learner = SVC()
    evaluate_classifier(paths, learner)
