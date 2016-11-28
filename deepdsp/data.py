from __future__ import division, print_function, absolute_import

import numpy as np
from random import shuffle
from math import ceil

from .helpers import loadAudio
from .conf import *

num_buffs = int(ceil(sample_rate / buff_size))

library = dict(
    kick=loadAudio('kick'),
    snare=loadAudio('snare'),
    # clap=loadAudio('clap'),
    # tom=loadAudio('tom'),
    # hihat=loadAudio('hihat'),
)

# Shuffle to arrays in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def loadData():
    # All tracks in a matrix
    audio_matrix = np.zeros((0, buff_size, num_buffs, 2))

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
            print("loading: ", kind, " track: ", j + 1, "/", len(samples_list))

            dft = track.signalDFT()
            dft = dft.reshape((1, dft.shape[0], dft.shape[1], 2))
            audio_matrix = np.vstack((audio_matrix, dft))

    print("done loading")

    # Randomize Audio
    return unison_shuffled_copies(audio_matrix, classifications)
