from __future__ import division, print_function, absolute_import

from math import ceil
import tensorflow as tf

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import highway_conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

from ..helpers import compare
from ..classify import *

from deepdsp.conf import *
from deepdsp.data import library



num_buffs = int(ceil(sample_rate / buff_size))

# tensorboard
# tensorboard --logdir='/tmp/tflearn_logs'

def classifyConv2d():

    # ================================
    # Building convolutional network
    # ================================
    network = input_data(shape=[None, buff_size, num_buffs, 2], name='input')



    network = batch_normalization(network)

    # https://github.com/tflearn/tflearn/blob/51399601c1a4f305db894b871baf743baa15ea00/tflearn/layers/core.py#L96
    # network = fully_connected(network, 512, activation='leaky_relu')
    network = fully_connected(network, 256, activation='elu', name="elu")
    network = fully_connected(network, len(library), activation='softmax', name='softmax')

    # https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/layers/estimator.py#L14
    reg = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    print("Training")
    # Training
    # https://github.com/tflearn/tflearn/blob/66c0c9c67b0472cbdc85bae0beb7992fa008480e/tflearn/models/dnn.py#L10
    model = tflearn.DNN(reg, tensorboard_verbose=3)

    model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
              show_metric=True, run_id='convnet_dsp')


    # Validation
    pred = model.predict(valX)
    val_acc = compare(pred, valY)



    print("Validation accuracy : ", val_acc)