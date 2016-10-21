import wave
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi, cos, sin, ceil, fabs, log
import DSP.util as util
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
            self.signal = np.zeros((length))
            self.length = length
            self.sampleRate = spf.getframerate()

            # audio is 16 bit
            # Sets the digital amp to be between -1 and 1. type float64
            maxAmp = (2 ** 16) / 2
            normalize = np.vectorize(lambda amp: amp / maxAmp)
            self.signal[:spf.getnframes()] = normalize(signal)
            self.signal = self.signal.reshape(1, length)

            spf.close()

        # Input is already a signal
        if type(input) == np.ndarray:
            self.signal = input
            self.length = length
            self.sampleRate = AUDIO_REQ['sample_rate']


    def signalDFT(self, res=128):
        _, N = self.signal.shape

        num_buffs = int(ceil(self.length / BUF_SIZE))

        # Compute freqs to use
        base = MAX_FREQ ** (1/res)
        # Start freqs at 20
        t = int(ceil( log(MIN_FREQ, base) ))
        freqs = np.zeros((res-t), dtype=int)
        for i in range(t, res):
            freqs[i-t] = int(base ** i)

        # Perform DFT for each buffer
        out = np.zeros((len(freqs), 0))
        print(num_buffs)
        for i in range(0, num_buffs):
            print(i)
            start = i * BUF_SIZE
            end = (i + 1) * BUF_SIZE
            if end > self.length:
                end = self.length % BUF_SIZE

            signal_chunk = self.signal[ 0,start:end ]
            chunk = util.dft(signal_chunk, freqs)
            out = np.hstack((out, chunk))

        return (freqs, out)


    def plotFreqs(self, res):
        freqs, amps = self.signalDFT(res)

        print(amps)

        plt.plot(freqs, amps)
        plt.xscale('log')
        plt.title('Frequencies')
        plt.grid(True)
        plt.show()

    def plotAmp(self):
        plt.figure(1)
        plt.title('Signal Wave')

        signal = self.signal.reshape(self.length, 1)
        plt.plot(signal)
        plt.show()

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
        # plt.figure(2)
        plt.subplot(212)
        plt.title('Frequencies')
        plt.xscale('log')
        plt.axis([0.0, 25000.0, -1.0, 1.0])
        ax = plt.gca()
        ax.set_autoscale_on(False)

        freqs, amps = self.signalDFT(res)

        print(amps.shape)

        vecreal = np.vectorize(lambda a: fabs(a.real))
        amps = vecreal(amps)

        _, length = amps.shape

        plt.ion()

        for i in range(length):
            plt.cla()
            plt.title('Frequencies')
            plt.xscale('log')
            plt.axis([0.0, 25000.0, -1.0, 1.0])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            plt.plot(freqs, amps[:,i])
            plt.pause(0.1)

        plt.pause(60)

        plt.show()


    """Validate that audio files fill requirement or throw an error otherwise"""
    @staticmethod
    def validate(spf, filepath):
        if not spf.getnchannels() == AUDIO_REQ["channels"]:
            raise FileExistsError("Audio file (%s) must have %d channel(s), it has %d" % (filepath, AUDIO_REQ["channels"], spf.getnchannels()) )

        if not spf.getframerate() == AUDIO_REQ["sample_rate"]:
            raise FileExistsError("Audio file (%s) must  have a sample rate of %d, this has %d" % (filepath, AUDIO_REQ["sample_rate"], spf.getframerate()) )
