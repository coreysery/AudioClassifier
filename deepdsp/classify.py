from DSP import Signal
import numpy as np
import tensorflow as tf

from .helpers import loadAudio

def main():

    samples = dict(
        kick = loadAudio('kick'),
        snare = loadAudio('snare'),
    )

    # All tracks in a matrix
    audio_matrix = np.zeros((0,4096))
    # Each track classified
    classifications = np.zeros((0, len(samples)))

    for i, (kind, list) in enumerate(samples.items()):
        # Set binary classifications for this kind
        class_row = np.zeros((len(list), len(samples)))
        ones = np.ones((len(list), 1))
        class_row[:,i:i+1] = ones
        classifications = np.vstack((classifications,class_row ))

        # append each track to the audio matrix
        for track in list:
            audio_matrix = np.vstack((audio_matrix, track.signal))


    print(audio_matrix.shape)
    print(classifications.shape)


