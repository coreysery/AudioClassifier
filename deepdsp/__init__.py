from DSP import Signal
import numpy as np
import tensorflow as tf
import os
from config import ROOT_DIR

dir = os.path.dirname(__file__)

def loadAudio(dir):
    tracks = np.zeros((0,16))

    for fn in os.listdir(ROOT_DIR + '/resources/audio/' + dir):
        # Double check we are loading a wav file
        if not fn.lower().endswith(('.wav')):
            continue

        filepath = os.path.join(ROOT_DIR, 'resources/audio/', dir, fn)
        track = Signal(filepath)
        tracks = np.vstack((tracks, track.signal))

    return tracks


def main():
    signals = np.zeros((0,16))

    kicks = loadAudio('kick')
    snares = loadAudio('snare')

    signals = np.vstack((signals, kicks, snares))

    print(signals.shape)

    # # 1 x (signal length) matrix
    # snare_matrix = tf.constant(snare.signal)
    #
    # # Launch the default graph.
    # sess = tf.Session()
    #
    # sess.close()