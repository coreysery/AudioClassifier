from __future__ import division, print_function, absolute_import
import numpy as np
from random import shuffle
from math import ceil
import tensorflow as tf

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import highway_conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

from .conf import *
from .data import loadData, library


num_buffs = int(ceil(sample_rate / buff_size))

def main():
    # ================================
    # Data loading and preprocessing
    # ================================
    audio_matrix, classifications = loadData()

    # Ratios
    ratio_train = .8
    ratio_test = .2

    # Length of each type according to respective ratios
    train_len = int(ratio_train * classifications.shape[0])

    # Respective datasets
    X = audio_matrix[:train_len, :]
    testX = audio_matrix[train_len + 1:, :]

    # Binary Classifications
    Y = classifications[:train_len, :]
    testY = classifications[train_len + 1:, :]

    # dimensions : [tracks, freqs, buffs, (real,imag)]
    X = X.reshape([-1, X.shape[1], X.shape[2], 2])
    testX = testX.reshape([-1, testX.shape[1], testX.shape[2], 2])


    # ================================
    # Building convolutional network
    # ================================
    network = input_data(shape=[None, buff_size, num_buffs, 2], name='input')



    # highway convolutions with pooling and dropout
    for i in range(3):
        for j in [8, 4, 2, 1]:
            # https://github.com/tflearn/tflearn/blob/2faad812dc35e08457dc6bd86b15392446cffd87/tflearn/layers/conv.py#L1346
            network = highway_conv_2d(network, 16, j, activation='leaky_relu')

        # https://github.com/tflearn/tflearn/blob/2faad812dc35e08457dc6bd86b15392446cffd87/tflearn/layers/conv.py#L266
        network = max_pool_2d(network, 8)
        # https://github.com/tflearn/tflearn/blob/2faad812dc35e08457dc6bd86b15392446cffd87/tflearn/layers/normalization.py#L20
        network = batch_normalization(network)

    # https://github.com/tflearn/tflearn/blob/51399601c1a4f305db894b871baf743baa15ea00/tflearn/layers/core.py#L96
    network = fully_connected(network, 512, activation='leaky_relu')
    network = fully_connected(network, 256, activation='elu')
    network = fully_connected(network, len(library), activation='softmax')

    # https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/layers/estimator.py#L14
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    print("Training")
    # Training
    # https://github.com/tflearn/tflearn/blob/66c0c9c67b0472cbdc85bae0beb7992fa008480e/tflearn/models/dnn.py#L10
    model = tflearn.DNN(network, tensorboard_verbose=3)
    # https://github.com/tflearn/tflearn/blob/66c0c9c67b0472cbdc85bae0beb7992fa008480e/tflearn/models/dnn.py#L89
    model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
              show_metric=True, run_id='convnet_highway_dsp')
