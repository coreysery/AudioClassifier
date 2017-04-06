import numpy as np
from deepdsp.data import loadData

# ================================
# Data loading and preprocessing
# ================================

audio_matrix, classifications = loadData()

print("Classification counts: ", np.sum(classifications, axis=0))

# Ratios
ratio_train = .8
ratio_test = .1
ratio_val = .1

# Length of each type according to respective ratios
train_len = int(ratio_train * classifications.shape[0])
test_len = int(ratio_test * classifications.shape[0])

# Respective datasets
X = audio_matrix[:train_len, :]
testX = audio_matrix[train_len + 1:(test_len + train_len), :]
valX = audio_matrix[(test_len + train_len) + 1:, :]

# Binary Classifications
Y = classifications[:train_len, :]
testY = classifications[train_len + 1:(test_len + train_len), :]
valY = classifications[(test_len + train_len) + 1:, :]

print("Train Classification count: ", np.sum(Y, axis=0))
print("Test Classification count: ", np.sum(testY, axis=0))
print("Validation Classification count: ", np.sum(valY, axis=0))

# dimensions : [tracks * buffs * freqs * channel(real,imag)]
X = X.reshape([-1, X.shape[1], X.shape[2], 4])
testX = testX.reshape([-1, testX.shape[1], testX.shape[2], 4])
valX = valX.reshape([-1, valX.shape[1], valX.shape[2], 4])
