import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_REQ = dict(
    sample_rate=44100,
    bitDepth=16,
    # mono
    channels=1,
)