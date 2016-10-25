from math import pi, e, fabs, log
import numpy as np
from config import AUDIO_REQ

"""Discrete Fourier transform
    Return frequency domain of this Signal.
    :arg input => Numpy vector, input signal to transform
    :arg freqs => List of frequencies"""
def dft(input, freqs):
    N = input.shape[0]
    out = np.zeros((len(freqs),1), dtype=np.complex)

    for i, freq in enumerate(freqs):

        val = np.complex(0, 0)
        for n in range(N):
            exponent = -(2 * pi * n * freq) / N
            complex = e ** np.complex(0, exponent)
            val += (fabs(input[n]) * complex)

        out[i] = 0.5 * val

    return out

"""Fast Fourier transform, Cooley-Tukey implementation - O(NlogN)
    :return numpy complex vector representing phase and magnitude"""
# Still needs work
def fft(input):
    # N = len(input)

    twiddle = np.complex(e, 0) ** np.complex(0, -2 * pi)


    def radix2(x, N, s=1):
        if N == 1:
            return x
        else:
            mid = int(N / 2)

            time1 = np.arange((N/(2)), dtype=int) * 2
            time2 = np.arange((N/(2)), dtype=int) * 2 + 1

            half1 = np.array([x[i,0] for i in time1], dtype=np.complex).reshape(len(time1), 1)
            half2 = np.array([x[i,0] for i in time2], dtype=np.complex).reshape(len(time2), 1)

            first = radix2(half1, mid, (2*s))
            sec = radix2(half2,  mid, (2*s))

            out = np.vstack((first, sec))
            for k in range(mid):
                t = out[k]
                b = (twiddle ** (k/N)) * out[k + mid]
                out[k] = t + b
                out[k + mid] = t - b

            return out

    return (1/len(input)) * radix2(input, len(input))
