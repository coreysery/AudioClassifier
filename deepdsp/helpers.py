from .sig import Signal
import os
from config import ROOT_DIR
from .conf import sample_rate

"""Load all the .wav files in a given directory, return list of type Signal"""
def loadAudio(dir):
    tracks = []
    max_tracks = 55
    i = 1

    for fn in os.listdir(ROOT_DIR + '/resources/audio/' + dir):
        # Double check we are loading a wav file
        if not fn.lower().endswith(('.wav')):
            continue

        filepath = os.path.join(ROOT_DIR, 'resources/audio/', dir, fn)

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

