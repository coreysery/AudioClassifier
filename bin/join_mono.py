import os
from pydub import AudioSegment

from config import ROOT_DIR
from deepdsp.conf import conf

source_path = os.path.join(ROOT_DIR, 'resources', 'source')
target_path = os.path.join(ROOT_DIR, 'resources', 'tmp')

bounce_length = 2 # in seconds

loaded = []

for fn in os.listdir(source_path):
    if not fn.lower().endswith(('.wav')):
        continue
    print("load: ", fn)

    """ Load and join mono files to stereo """
    split = fn.split('.')
    name = split[0]
    channel = split[1]

    if name in loaded:
        continue
    loaded.append(name)

    otherChannel = "L" if (channel == "R") else "R"
    otherFile = "{}.{}.wav".format(name, otherChannel)

    ch1 = AudioSegment.from_wav(os.path.join(source_path, fn))
    ch2 = AudioSegment.from_wav(os.path.join(source_path, otherFile))

    if channel == "L":
        a = AudioSegment.from_mono_audiosegments(ch1, ch2)
    else:
        a = AudioSegment.from_mono_audiosegments(ch2, ch1)

    # file config
    a = a.set_channels(conf["channels"])
    a = a.set_frame_rate(conf["sample_rate"])
    a = a.set_sample_width(conf["sample_width"])

    target_name = "{}.wav".format(name)
    a.export( os.path.join(target_path, target_name), format="wav")


