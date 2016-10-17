import wave
import matplotlib.pyplot as plt
import numpy as np
import os

from config import AUDIO_REQ

class Signal():
    def __init__(self, filepath):
        # Open wave file
        spf = wave.open(filepath, mode='rb')
        Signal.validate(spf, filepath)

        # Extract Raw Audio from Wav File
        signal = np.fromstring(spf.readframes(-1), 'Int16')
        self.length = spf.getnframes()
        self.sampleRate = spf.getframerate()

        # audio is 16 bit so we have to normalize amplitude to be
        maxAmp = (2 ** 16) / 2
        normalize = np.vectorize(lambda amp: amp / maxAmp)

        # Sets the digital amp to be between -1 and 1. type float64
        self.signal = normalize(signal)
        # self.signal = self.signal.reshape(1, self.length)

        spf.close()

    def plot(self):
        plt.figure(1)
        plt.title('Signal Wave')
        print(self.signal)

        plt.plot(self.signal)
        plt.show()

    @staticmethod
    def validate(spf, filepath):
        if not spf.getnchannels() == AUDIO_REQ["channels"]:
            raise FileExistsError("Audio file (%s) must have %d channel(s), it has %d" % (filepath, AUDIO_REQ["channels"], spf.getnchannels()) )

        if not spf.getframerate() == AUDIO_REQ["sample_rate"]:
            raise FileExistsError("Audio file (%s) must  have a sample rate of %d, this has %d" % (filepath, AUDIO_REQ["sample_rate"], spf.getframerate()) )
