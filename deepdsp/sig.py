import wave
import numpy as np
from pydub import AudioSegment

from .conf import conf

length = conf["max_length"] * conf["sample_rate"]

class Signal():
    def __init__(self, input):
        #  Input is a wav file to open
        if type(input) == str:
            # Open wave file
            # spf = wave.open(input, mode='rb')
            # Signal.validate(spf, input)

            sound = AudioSegment.from_wav(input)
            sound = sound.set_channels(conf["channels"])
            sound = sound.set_frame_rate(conf["sample_rate"])
            sound = sound.set_sample_width(conf["sample_width"])

            # Extract Raw Audio from Wav File
            # signal = np.fromstring(spf.readframes(-1), 'Int16')

            # Initialize signal array
            self.signal = np.zeros((length, conf["channels"]))
            self.length = length
            # self.sampleRate = spf.getframerate()
            # self.channels = spf.getnchannels()

            # duration of sound in samples
            dur = sound.duration_seconds * conf["sample_rate"]

            if dur > length:
                # If file is longer than the max length
                sound = sound[length:]
            else:
                # If the file is shorter than the max length
                pad = (length - dur)
                silence = AudioSegment.silent(duration=pad, frame_rate=conf["sample_rate"])
                sound = sound.append(silence, crossfade=0)

            left, right = sound.split_to_mono()

            # print(left.get_array_of_samples())
            signal = np.array([left.get_array_of_samples(), right.get_array_of_samples()])

            # audio is 16 bit
            # Sets the digital amp to be between -1 and 1. type float64
            maxAmp = (2 ** conf["bitDepth"]) / 2
            normalize = np.vectorize(lambda amp: amp / maxAmp)
            self.signal = normalize(signal[:, :length])

            self.signal = self.signal.reshape(length, conf["channels"])

            # spf.close()

        # Input is already a signal
        if type(input) == np.ndarray:
            self.signal = input
            self.length = length
            self.sampleRate = conf.sample_rate


    def signalDFT(self):

        # Perform DFT for each buffer
        # buf_size x num_buffs x complex channels
        out = np.zeros((conf["buff_size"], 0, 4))

        for i in range(conf["num_buffs"]):
            start = i * conf["buff_size"]
            end = (i + 1) * conf["buff_size"]

            if end > self.length:
                # if signal is to short
                pad_length = end - self.length
                pad = np.zeros((pad_length, conf["channels"]))
                signal_chunk = np.vstack((self.signal[start:], pad))
            else:
                signal_chunk = self.signal[ start:end, : ]

            chunk = np.fft.fft2(signal_chunk)

            newChunk = np.zeros((0, 4))

            for _, val in enumerate(chunk):
                l, r = val
                k = np.array([[l.real, l.imag, r.real, r.imag]])
                newChunk = np.vstack((newChunk, k))

            newChunk = newChunk.reshape((newChunk.shape[0], 1, 4))

            out = np.hstack((out, newChunk))

        return out

    @staticmethod
    def IDFT(track):
        """Take a input from the frequency domain and turns it into a wav file"""
        out = np.zeros((0))
        print(track.shape)
        print(out.shape)
        for i in range(track.shape[1]):
            buff = track[:, i, :]
            print(buff.shape)
            buff_sig = np.fft.ifft(buff)
            print(buff_sig.shape)
            out = np.hstack((out, buff_sig))
            print(out.shape)


    """Validate that audio files fill requirement or throw an error otherwise"""
    @staticmethod
    def validate(spf, filepath):
        if not spf.getnchannels() == conf["channels"]:
            raise FileExistsError("Audio file (%s) must have %d channel(s), it has %d" % (filepath, conf["channels"], spf.getnchannels()) )

        if not spf.getframerate() == conf["sample_rate"]:
            raise FileExistsError("Audio file (%s) must  have a sample rate of %d, this has %d" % (filepath, conf["sample_rate"], spf.getframerate()) )
