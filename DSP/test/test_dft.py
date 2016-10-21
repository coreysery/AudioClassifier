import os
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

from DSP import Signal
from config import AUDIO_REQ
from config import ROOT_DIR


kickpath = os.path.join(ROOT_DIR, 'resources/audio/kick/k8.wav')
snarepath = os.path.join(ROOT_DIR, 'resources/audio/snare/s2.wav')

kick = Signal(kickpath, 44100)
# snare = Signal(snarepath, 44100)

# freqDomain = kick.dft(64)
# print(freqDomain)


freq = 100
Fs = AUDIO_REQ['sample_rate']

time = np.arange(Fs) / Fs

s = lambda x: sin(x)
vs = np.vectorize(s)

sine = vs(2 * pi * time * freq)
sine = sine.reshape(1, len(sine))

sine_wave = Signal(sine, len(sine))
sine_wave.plotFreqs(512)
