from math import ceil

# Audio settings
sample_rate = 11025
buff_size = 1024
num_buffs = int(ceil(sample_rate/buff_size))

bitDepth = 16
channels = 1 # mono

# Limit tracks per type to keep types balanced
max_tracks = 200

classes = ["snare", "kick", "tom", "hihat"]