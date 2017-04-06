import os
import numpy as np

from config import ROOT_DIR
from .sig import Signal
from .conf import conf



"""Load all the .wav files in a given directory, return list of type Signal"""
def loadAudio(dir):
    tracks = []
    i = 1

    for fn in os.listdir(ROOT_DIR + '/resources/audio/' + dir):
        # Double check we are loading a wav file
        if not fn.lower().endswith(('.wav')):
            continue

        filepath = os.path.join(ROOT_DIR, 'resources/audio/', dir, fn)

        print(filepath)

        # try:
        tracks.append(Signal(filepath))
        # except:
            # Sample wasn't valid, so delete it
            # os.remove(filepath)
            # print("error")
            # pass

        if i > conf["max_tracks"]:
            break
        else:
            i += 1

    return tracks

# Compare predictions to the correct labels, return precentage correct
def compare(predictions, labels):
    num_correct = 0

    for i in range(len(predictions)):
        # (index, val)
        highest = (0,0)
        class_index = -1
        for j in range(len(predictions[i])):
            if predictions[i][j] > highest[1]:
                highest = (j, predictions[i][j])

            if labels[i][j] == 1:
                class_index = j

        if highest[0] == class_index:
            num_correct += 1

    flatten = np.vectorize(lambda x: round(x))
    f = flatten(predictions)
    counts = np.sum(f, axis=0)
    print("Predictions for each class: ", counts)
    print("Actual for each class: ", np.sum(labels, axis=0))

    return  num_correct/len(predictions)
