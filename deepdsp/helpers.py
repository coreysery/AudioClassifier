from .sig import Signal
import os
import subprocess
import numpy as np
from config import ROOT_DIR
from .conf import sample_rate

subprocess.call("./bin/downsample.sh", shell=True)

"""Load all the .wav files in a given directory, return list of type Signal"""
def loadAudio(dir):
    tracks = []
    max_tracks = 200
    i = 1

    for fn in os.listdir(ROOT_DIR + '/resources/tmp/' + dir):
        # Double check we are loading a wav file
        if not fn.lower().endswith(('.wav')):
            continue

        filepath = os.path.join(ROOT_DIR, 'resources/tmp/', dir, fn)

        try:
            tracks.append(Signal(filepath, sample_rate))
        except:
            # Sample wasn't valid, so delete it
            os.remove(filepath)
            pass

        if i > max_tracks:
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

    return  num_correct/len(predictions)
