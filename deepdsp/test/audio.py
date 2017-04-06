
from deepdsp.sig import Signal
from deepdsp.conf import sample_rate
from config import ROOT_DIR

snare = Signal(ROOT_DIR + "/resources/tmp/snare/s2.wav", length=sample_rate)
print(snare.signal.shape)
freq_domain = snare.signalDFT()
print(freq_domain.shape)
back_to_time = Signal.IDFT(freq_domain)
# print(back_to_time.shape)

