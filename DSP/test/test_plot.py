import os
import numpy as np
from math import sin, pi
from DSP import Signal

from config import ROOT_DIR, AUDIO_REQ


kickpath = os.path.join(ROOT_DIR, 'resources/audio/kick/k8.wav')
snarepath = os.path.join(ROOT_DIR, 'resources/audio/snare/s2.wav')

kick = Signal(kickpath, 22100)
snare = Signal(snarepath, 44100)


freq = 100
Fs = AUDIO_REQ['sample_rate']
amp = 0.8
time = np.arange(Fs) / Fs

s = lambda x: amp * sin(x)
vs = np.vectorize(s)

sine = vs(2 * pi * time * freq)
length = len(sine)
sine = sine.reshape((1, length))
sine_wave = Signal(sine, length)


kick.plot(256)
# sine_wave.plot(256)