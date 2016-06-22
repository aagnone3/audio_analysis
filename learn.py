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
import pickle
from learning.classification import AudioClassifier, GMMAudioClassifier
from common import environment
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB

# load feature masks and their names
FEATURE_MASK = np.array([False] * 22)
FEATURE_MASK[-13:] = True
FEATURE_MASK_NAME = ["Only 7 MFCC"]

# data directories and paths
DATA_DIR = os.sep.join(['res', 'data', 'audio_classification'])
TRAIN_DATA_DIR = os.sep.join([DATA_DIR, 'train'])
TEST_DATA_DIR = os.sep.join([DATA_DIR, 'test'])

# constants
training_files = glob.glob(os.sep.join((TRAIN_DATA_DIR, "*.wav")))
train_prefixes = [f[:f.find("_")] for f in training_files]
train_file_counts = [train_prefixes.count(p) for p in np.unique(train_prefixes)]
testing_files = glob.glob(os.sep.join((TEST_DATA_DIR, "*.wav")))
test_prefixes = [f[:f.find("_")] for f in testing_files]
test_file_counts = [test_prefixes.count(p) for p in np.unique(test_prefixes)]


def main():
    logging.basicConfig(filename='application.log',
                        level=logging.INFO,
                        format='%(asctime)s\t\t%(message)s')
    logging.info('Hello logging world')
    print("Training file counts {}".format(train_file_counts))

    keyword = 'Test'
    algorithm = "audio_classification"
    paths = {
        'model': os.sep.join(['res', 'models', algorithm]),
        'train_results': os.sep.join(['res', 'results', algorithm + '_train.csv']),
        'test_results': os.sep.join(['res', 'results', algorithm + '_test.csv']),
        'train_data': os.sep.join(['res', 'features', '']) + 'TrainData_' + keyword + '.pkl',
        'test_data': os.sep.join(['res', 'features', '']) + 'TestData_' + keyword + '.pkl'
    }
    environment.print_lines(5)
    print("Testing file counts {}".format(test_file_counts))

    # learner = mixture.GMM(n_components=len(TRAINING_DIRS))
    # learner = GaussianNB()
    learner = SVC()
    fresh_train_test(paths, learner)
    # modified_train_test(paths, learner)


def modified_train_test(paths, learner):
    """
    Performs training and testing using paths to previous extracted features
    :param paths: paths to previous extracted features
    :param learner: learner to use
    :return: None
    """
    data = pd.DataFrame([])
    AudioClassifier(learner).train_model(TRAIN_DATA_DIR, paths['model'],
                                         load_features_from=paths['train_data'],
                                         feature_mask=FEATURE_MASK)
    [y_predict, y_actual] = AudioClassifier.test_model(classifier=AudioClassifier.load_model(paths['model']),
                                                       load_features_from=paths['test_data'],
                                                       feature_mask=FEATURE_MASK)
    results, columns = analyze_test_results(y_predict, y_actual, paths['results'])
    percents = np.reshape(results['percents'], (1, -1))

    data = data.append(pd.DataFrame(percents))
    data.columns = columns
    data.index = pd.Index(FEATURE_MASK_NAME)
    data.to_csv(os.sep.join(['res', 'results', 'results.csv']))


def fresh_train_test(paths, learner):
    """
    Performs training and testing with the specified learner by extracting features from the files in the training
    and testing directories specified globally
    :param paths: paths to save learning results
    :param learner: learner to use
    :return: None
    """
    classifier = AudioClassifier(learner)
    [y_predict, y_actual] = classifier.train_model(TRAIN_DATA_DIR,
                                                   new_model_name=paths['model'],
                                                   feature_mask=FEATURE_MASK,
                                                   save_features_to=paths['train_data'])
    analyze_test_results(y_predict, y_actual, paths['train_results'])
    [y_predict, y_actual] = AudioClassifier.test_model(classifier=AudioClassifier.load_model(paths['model']),
                                                       test_files_directory=TEST_DATA_DIR,
                                                       save_features_to=paths['test_data'],
                                                       feature_mask=FEATURE_MASK)
    analyze_test_results(y_predict, y_actual, paths['test_results'])


def log_test_results(file_name, results, fieldnames):
    """
    Logs results
    :param file_name: file name to use
    :param results: results to save
    :param fieldnames: column header names
    :return:
    """
    with open(file_name, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result for result in results)


def analyze_test_results(y_predict, y_actual, results_file_name):
    """
    Prints a simple analysis of test results to the console
    :param y_predict: predicted class labels
    :param y_actual: actual class labels
    :param results_file_name: path to use for saving off results
    :return: None
    """
    environment.print_lines(1)
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
    with open(results_file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(y_predict.shape[0]):
            writer.writerow([y_predict[i], y_actual[i]])
        csv_file.close()
    for i, percent in enumerate(metrics['percents']):
        print("{:22}:\t\t{:6.2f} %\t\t{:2d}/{:2d}".format(columns[i], percent, metrics['partials'][i], metrics['wholes'][i]))
    return metrics, columns


if __name__ == '__main__':
    main()
