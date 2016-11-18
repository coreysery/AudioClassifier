import wave
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from math import ceil, fabs
from config import AUDIO_REQ

MAX_FREQ = AUDIO_REQ["sample_rate"]/2
MIN_FREQ = 20
BUF_SIZE = 1024

class Signal():
    def __init__(self, input, length):
        #  Input is a wav file to open
        if type(input) == str:
            # Open wave file
            spf = wave.open(input, mode='rb')
            Signal.validate(spf, input)

            # Extract Raw Audio from Wav File
            signal = np.fromstring(spf.readframes(-1), 'Int16')

            # Initialize signal array
            self.signal = np.zeros((length,1))
            self.length = length
            self.sampleRate = spf.getframerate()

            # audio is 16 bit
            # Sets the digital amp to be between -1 and 1. type float64
            maxAmp = (2 ** 16) / 2
            normalize = np.vectorize(lambda amp: amp / maxAmp)
            self.signal[:spf.getnframes(),0] = normalize(signal)
            self.signal = self.signal.reshape(length, 1)

            spf.close()

        # Input is already a signal
        if type(input) == np.ndarray:
            self.signal = input
            self.length = length
            self.sampleRate = AUDIO_REQ['sample_rate']


    def signalDFT(self):
        _, N = self.signal.shape

        num_buffs = int(ceil(self.length / BUF_SIZE))

        # Perform DFT for each buffer
        # buf_size x num_buffs x complex
        out = np.zeros((BUF_SIZE, 0, 2))

        for i in range(num_buffs):
            start = i * BUF_SIZE
            end = (i + 1) * BUF_SIZE

            if end > self.length:
                pad_length = end - self.length
                pad = np.zeros((pad_length, 1))
                signal_chunk = np.vstack((self.signal[start:], pad))
            else:
                signal_chunk = self.signal[ start:end ]

            # chunk = util.fft(signal_chunk)
            chunk = np.fft.fft(signal_chunk)
            newChunk = np.zeros((0, 2))
            for _, val in enumerate(chunk):
                k = np.array([val.real, val.imag]).reshape((1,2))
                newChunk = np.vstack((newChunk, k))

            newChunk = newChunk.reshape((newChunk.shape[0], 1, 2))

            out = np.hstack((out, newChunk))

        return out



    def plot(self, res):
        plt.grid(True)

        # Plot time domain
        plt.figure(1)
        plt.subplot(211)
        plt.title('Signal Wave')
        plt.xscale('linear')
        timeDomain = self.signal.reshape(self.length, 1)
        plt.plot(timeDomain)

        # Plot freq domain
        plt.subplot(212)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        freqs, amps = self.signalDFT(res)

        _, length = amps.shape
        vecreal = np.vectorize(lambda a: fabs(a.real))
        amps = vecreal(amps)

        plt.ion()

        for i in range(length):
            plt.title('Frequencies')
            plt.xscale('log')
            ax.set_autoscale_on(False)
            # plt.axis([0.0, 25000.0, -0.1, 1.0])

            plt.plot(freqs, amps[:,i])
            plt.pause(0.5)
            plt.cla()

        plt.pause(4)

        plt.show()


    """Validate that audio files fill requirement or throw an error otherwise"""
    @staticmethod
    def validate(spf, filepath):
        if not spf.getnchannels() == AUDIO_REQ["channels"]:
            raise FileExistsError("Audio file (%s) must have %d channel(s), it has %d" % (filepath, AUDIO_REQ["channels"], spf.getnchannels()) )

        if not spf.getframerate() == AUDIO_REQ["sample_rate"]:
            raise FileExistsError("Audio file (%s) must  have a sample rate of %d, this has %d" % (filepath, AUDIO_REQ["sample_rate"], spf.getframerate()) )
