from math import pi, e, fabs
import numpy as np

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
            val +=  (fabs(input[n]) * complex)

        out[i] = val

    return out


