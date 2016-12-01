from __future__ import division, print_function, absolute_import
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

from deepdsp.classify import X, Y, testX, testY, valX, valY
from deepdsp.helpers import compare
from deepdsp.models.conv_highway import model


# tensorboard
# tensorboard --logdir='/tmp/tflearn_logs'

print("fitting conv highway")

# https://github.com/tflearn/tflearn/blob/66c0c9c67b0472cbdc85bae0beb7992fa008480e/tflearn/models/dnn.py#L89
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
          show_metric=True, run_id='convnet_highway_dsp')

# Validation
pred = model.predict(valX)
val_acc = compare(pred, valY)
print("Validation accuracy : ", val_acc)