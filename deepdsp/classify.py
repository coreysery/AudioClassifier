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
    audio_matrix = np.zeros((0,88200))
    testing_matrix = np.zeros((0, 88200))
    validation_matrix = np.zeros((0, 88200))
    training_matrix = np.zeros((0, 88200))
    # Each track classified
    classifications = np.zeros((0, len(samples)))

    for i, (kind, list) in enumerate(samples.items()):
        # Set binary classifications for this kind
        class_row = np.zeros((len(list), len(samples)))
        ones = np.ones((len(list), 1))
        class_row[:,i:i+1] = ones
        classifications = np.vstack((classifications,class_row ))

        # append each track to the audio matrix
        for j, track in enumerate(list):
            # 20% go to testing and validation
            # 80 to training
            if(j%10==0):
                testing_matrix = np.vstack((testing_matrix, track.signal.reshape((1,88200))))
            elif(j%10==1):
                validation_matrix = np.vstack((validation_matrix, track.signal.reshape((1,88200))))
            else:
                training_matrix = np.vstack((training_matrix, track.signal.reshape((1,88200))))
            audio_matrix = np.vstack((audio_matrix, track.signal.reshape((1,88200))))


    track_count, sample_length = audio_matrix.shape






