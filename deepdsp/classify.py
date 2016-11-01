from DSP import Signal
import numpy as np
import tensorflow as tf

from .helpers import loadAudio
from random import shuffle


def main():
    library = dict(
        kick=loadAudio('kick'),
        snare=loadAudio('snare'),
    )

    # All tracks in a matrix
    audio_matrix = np.zeros((0, 88200))

    # Each track classified
    classifications = np.zeros((0, len(library)))

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
            audio_matrix = np.vstack((audio_matrix, track.signal.reshape((1, 88200))))


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
