from math import ceil

sample_rate = 48000
max_length = 2 # in seconds
buff_size = 1024

conf = {
    # Audio settings
    "sample_rate": sample_rate,
    "max_length": max_length,
    "buff_size": buff_size,

    "num_buffs": int(ceil( max_length * (sample_rate / buff_size) )),

    "bitDepth": 16,
    "sample_width": 2, # 16 bit
    "channels": 2,  # stereo

    # Whether or not to downsample the audio
    "downSample": False,

    # Limit tracks per type to keep types balanced
    "max_tracks": 200,

    "randomize": True,

    "classes": [
        "head_0",
        "head_30",
        "head_60",
        "head_90",
        "head_120",
        "head_150",
        "head_180",
        "head_210",
        "head_240",
        "head_270",
        "head_300",
        "head_330",
    ],
}