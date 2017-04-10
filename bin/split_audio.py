import os
import math
from pydub import AudioSegment

from config import ROOT_DIR
from deepdsp.conf import conf

source_path = os.path.join(ROOT_DIR, 'resources', 'tmp')
target_path = os.path.join(ROOT_DIR, 'resources', 'audio')

bounce_length = 2 # in seconds
min_rms = 500

loaded = []

for fn in os.listdir(source_path):
    if not fn.lower().endswith(('.wav')):
        continue
    print("load: ", fn)

    """ Load and join mono files to stereo """
    split = fn.split('.')
    deg = split[1]

    fp = os.path.join(source_path, fn)
    a = AudioSegment.from_wav(fp)

    tar_dir = os.path.join(target_path, "head_{}".format(deg))
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    num_segments = math.floor(a.duration_seconds / bounce_length)
    start, end = 0, bounce_length * 1000
    for i in range(num_segments):
        seg = a[start:end]

        start += bounce_length * 1000
        end += bounce_length * 1000

        if seg.rms < min_rms:
            continue

        name = "seg{}.{}.wav".format(i, deg)

        print("Exporting: ", name)

        seg.export(os.path.join(tar_dir, name), format="wav")



