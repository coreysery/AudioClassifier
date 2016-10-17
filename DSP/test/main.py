import os
from DSP import Signal

from config import ROOT_DIR

kickpath = os.path.join(ROOT_DIR, 'resources/audio/kick/k1.wav')
snarepath = os.path.join(ROOT_DIR, 'resources/audio/snare/s2.wav')

kick = Signal(kickpath)
snare = Signal(snarepath)

kick.plot()
snare.plot()
