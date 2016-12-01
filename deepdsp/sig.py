import wave
import numpy as np

from .conf import *


class Signal():
    def __init__(self, input, length):
        #  Input is a wav file to open
        if type(input) == str:
            # Open wave file
            spf = wave.open(input, mode='rb')
            # Signal.validate(spf, input)

            # Extract Raw Audio from Wav File
            signal = np.fromstring(spf.readframes(-1), 'Int16')

            # Initialize signal array
            self.signal = np.zeros((length,1))
            self.length = length
            self.sampleRate = spf.getframerate()
            self.channels = spf.getnchannels()

            # audio is 16 bit
            # Sets the digital amp to be between -1 and 1. type float64
            maxAmp = (2 ** 16) / 2
            normalize = np.vectorize(lambda amp: amp / maxAmp)

            if spf.getnframes() > length:
                self.signal[:length,0] = normalize(signal[:length])
            else:
                self.signal[:spf.getnframes(),0] = normalize(signal)

            self.signal = self.signal.reshape(length, 1)

            spf.close()

        # Input is already a signal
        if type(input) == np.ndarray:
            self.signal = input
            self.length = length
            self.sampleRate = sample_rate


    def signalDFT(self):
        _, N = self.signal.shape

        num_buffs = int(ceil(self.length / buff_size))

        # Perform DFT for each buffer
        # buf_size x num_buffs x complex
        out = np.zeros((buff_size, 0, 2))

        for i in range(num_buffs):
            start = i * buff_size
            end = (i + 1) * buff_size

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

    @staticmethod
    def IDFT(track):
        """Take a input from the frequency domain and turns it into a wav file"""
        out = np.zeros((0))
        print(track.size)
        print(out.size)
        for i in range(len(track)):
            buff = track[i]
            print(buff.size)
            buff_sig = np.fft.ifft(buff)
            print(buff_sig.size)
            out = np.hstack((out, buff_sig))
            print(out.size)


    """Validate that audio files fill requirement or throw an error otherwise"""
    @staticmethod
    def validate(spf, filepath):
        if not spf.getnchannels() == channels:
            raise FileExistsError("Audio file (%s) must have %d channel(s), it has %d" % (filepath, AUDIO_REQ["channels"], spf.getnchannels()) )

        if not spf.getframerate() == sample_rate:
            raise FileExistsError("Audio file (%s) must  have a sample rate of %d, this has %d" % (filepath, AUDIO_REQ["sample_rate"], spf.getframerate()) )
