from __future__ import division, print_function, absolute_import
from DSP import Signal
import numpy as np
import tensorflow as tf
from config import ROOT_DIR, AUDIO_REQ

from .helpers import loadAudio
from random import shuffle
from math import ceil, fabs

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import highway_conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

sample_rate = 44100
buf_size = 1024
num_buffs = int(ceil(sample_rate / buf_size))

def main():
    library = dict(
        kick=loadAudio('kick'),
        snare=loadAudio('snare'),
    )


    # All tracks in a matrix
    audio_matrix = np.zeros((0, buf_size, num_buffs, 2))

    # Each track classified
    classifications = np.zeros((0, len(library)))

    # kind : ie kick, snare, etc
    # samples_list : tracks
    for i, (kind, samples_list) in enumerate(library.items()):
        # Randomize list of audio
        shuffle(samples_list)

        # Set binary classifications for this kind
        class_row = np.zeros((len(samples_list), len(library)))
        ones = np.ones((len(samples_list), 1))
        class_row[:, i:i + 1] = ones
        classifications = np.vstack((classifications, class_row))

        # append each track to the audio matrix
        for j, track in enumerate(samples_list):
            dft = track.signalDFT()
            dft = dft.reshape((1, dft.shape[0], dft.shape[1], 2))
            print(audio_matrix.shape)
            audio_matrix = np.vstack((audio_matrix, dft))


    # Randomize Audio
    combined = list(zip(audio_matrix, classifications))
    shuffle(combined)
    audio_matrix[:], classifications[:] = zip(*combined)


    # Ratios
    ratio_train = .8
    ratio_test = .1
    ratio_validation = .1

    # Length of each type according to respective ratios
    train_len = int(ratio_train * classifications.shape[0])
    test_len = int(ratio_test * classifications.shape[0])
    validation_len = int(ratio_validation * classifications.shape[0])

    # Binary Classifications
    training_labels = classifications[:train_len, :]
    test_labels = classifications[train_len + 1:train_len + test_len, :]
    validation_labels = classifications[train_len + test_len + 1:, :]

    # Respective datasets
    training_matrix = audio_matrix[:train_len, :]
    testing_matrix = audio_matrix[train_len + 1:train_len + test_len, :]
    validation_matrix = audio_matrix[train_len + test_len + 1:, :]

    # track_count, sample_length = audio_matrix.shape


    # Data loading and preprocessing
    X, Y, testX, testY = training_matrix, training_labels, testing_matrix, test_labels
    print(X.shape)
    print(testX.shape)

    X = X.reshape([-1, X.shape[1], X.shape[2], 2])
    testX = testX.reshape([-1, testX.shape[1], testX.shape[2], 2])

    # Building convolutional network
    network = input_data(shape=[None, 1024, 44, 2], name='input')
    # highway convolutions with pooling and dropout
    for i in range(3):
        for j in [3, 2, 1]:
            network = highway_conv_2d(network, 16, j, activation='elu')
        network = max_pool_2d(network, 2)
        network = batch_normalization(network)

    network = fully_connected(network, 128, activation='elu')
    network = fully_connected(network, 256, activation='elu')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
              show_metric=True, run_id='convnet_highway_mnist')
