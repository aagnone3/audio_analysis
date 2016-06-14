# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:55:34 2016

@author: aagnone3
"""

import csv
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB

from learning.classification import AudioClassifier, GMMAudioClassifier
from common import environment

# load feature masks and their names
with open("features/feature_masks.pkl", "rb") as fh:
    FEATURE_MASK_NAMES = pickle.load(fh)
    FEATURE_MASKS = pickle.load(fh)

# set up directories and paths
DATA_DIR = os.sep.join(['res', 'data'])
# training data directories
TRAIN_DATA_DIR = os.sep.join([DATA_DIR, 'Train'])
TRAIN_MUSIC_DIR = os.sep.join([TRAIN_DATA_DIR, 'Music'])
TRAIN_SPEECH_DIR = os.sep.join([TRAIN_DATA_DIR, 'Speech'])
TRAIN_NOISE_DIR = os.sep.join([TRAIN_DATA_DIR, 'Noise'])
TRAIN_SILENCE_DIR = os.sep.join([TRAIN_DATA_DIR, 'Silence'])
TRAINING_DIRS = [
    TRAIN_MUSIC_DIR,
    TRAIN_SPEECH_DIR,
    TRAIN_NOISE_DIR,
    TRAIN_SILENCE_DIR
]
# testing data directories
TEST_DATA_DIR = os.sep.join([DATA_DIR, 'Test'])
TEST_MUSIC_DIR = os.sep.join([TEST_DATA_DIR, 'Music'])
TEST_SPEECH_DIR = os.sep.join([TEST_DATA_DIR, 'Speech'])
TEST_NOISE_DIR = os.sep.join([TEST_DATA_DIR, 'Noise'])
TEST_SILENCE_DIR = os.sep.join([TEST_DATA_DIR, 'Silence'])
TESTING_DIRS = [
    TEST_MUSIC_DIR,
    TEST_SPEECH_DIR,
    TEST_NOISE_DIR,
    TEST_SILENCE_DIR
]

# constants
NUM_MUSIC_TRAIN = len(environment.files_with_extension(TRAIN_MUSIC_DIR, '.wav'))
NUM_SPEECH_TRAIN = len(environment.files_with_extension(TRAIN_SPEECH_DIR, '.wav'))
NUM_NOISE_TRAIN = len(environment.files_with_extension(TRAIN_NOISE_DIR, '.wav'))
NUM_SILENCE_TRAIN = len(environment.files_with_extension(TRAIN_SILENCE_DIR, '.wav'))

NUM_MUSIC_TEST = len(environment.files_with_extension(TEST_MUSIC_DIR, '.wav'))
NUM_SPEECH_TEST = len(environment.files_with_extension(TEST_SPEECH_DIR, '.wav'))
NUM_NOISE_TEST = len(environment.files_with_extension(TEST_NOISE_DIR, '.wav'))
NUM_SILENCE_TEST = len(environment.files_with_extension(TEST_SILENCE_DIR, '.wav'))


def main():
    logging.basicConfig(filename='application.log',
                        level=logging.INFO,
                        format='%(asctime)s\t\t%(message)s')
    logging.info('Hello logging world')
    print(('Training with {} Music, {} Noise, {} Speech, and {} Silence file(s).'
           .format(NUM_MUSIC_TRAIN, NUM_NOISE_TRAIN, NUM_SPEECH_TRAIN, NUM_SILENCE_TRAIN)))

    keyword = 'Test'
    paths = {
        'model': os.sep.join(['res', 'models', 'svm_temp']),
        'temp_model': os.sep.join(['res', 'models', 'hmm_temp']),
        'train_data': os.sep.join(['res', 'features', '']) + 'TrainData_' + keyword + '.pkl',
        'test_data': os.sep.join(['res', 'features', '']) + 'TestData_' + keyword + '.pkl'
    }
    print(('\n'.join([n for n in 5 * ''])))
    print(('Testing with {0} Music, {1} Noise, {2} Speech, and {3} Silence file(s).'
           .format(NUM_MUSIC_TEST, NUM_NOISE_TEST, NUM_SPEECH_TEST, NUM_SILENCE_TEST)))

    # learner = mixture.GMM(n_components=len(TRAINING_DIRS))
    learner = GaussianNB()
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
    for i, mask in enumerate(FEATURE_MASKS):
        AudioClassifier(learner).train_model(TRAINING_DIRS, paths['temp_model'],
                                             from_file=paths['train_data'],
                                             feature_mask=mask)
        [y_predict, y_actual] = AudioClassifier.test_model(classifier=AudioClassifier.load_model(paths['temp_model']),
                                                           from_file=paths['test_data'],
                                                           feature_mask=mask)
        results, columns = analyze_test_results(y_predict, y_actual, paths['temp_model'])
        percents = np.reshape(results['percents'], (1, -1))
        data = data.append(pd.DataFrame(percents))
    data.columns = columns
    data.index = pd.Index(FEATURE_MASK_NAMES)
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
    classifier.train_model(TRAINING_DIRS, paths['model'],
                           feature_mask=FEATURE_MASKS[0],
                           to_file=paths['train_data'])

    [y_predict, y_actual] = AudioClassifier.test_model(classifier=AudioClassifier.load_model(paths['model']),
                                                       test_files_directory=TESTING_DIRS,
                                                       to_file=paths['test_data'],
                                                       feature_mask=FEATURE_MASKS[0])
    analyze_test_results(y_predict, y_actual, paths['model'])


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


def analyze_test_results(y_predict, y_actual, path_to_model):
    """
    Prints a simple analysis of test results to the console
    :param y_predict: predicted class labels
    :param y_actual: actual class labels
    :param path_to_model: path to use for saving off results
    :return: None
    """
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
    results_file_name = path_to_model + '_TestResults.csv'
    with open(results_file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(y_predict.shape[0]):
            writer.writerow([y_predict[i], y_actual[i]])
        csv_file.close()
    for i, percent in enumerate(metrics['percents']):
        print('%s:\t\t%f\t\t%i/%i' % (columns[i], percent, metrics['partials'][i], metrics['wholes'][i]))
    return metrics, columns


if __name__ == '__main__':
    main()
