from math import ceil

sample_rate = 48000
max_length = 1 # in seconds
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

    "classes": [
        "head.0",
        "head.30",
        "head.60",
        "head.90",
        "head.120",
        "head.150",
        "head.180",
        "head.210",
        "head.240",
        "head.270",
        "head.300",
        "head.330",
    ],
}