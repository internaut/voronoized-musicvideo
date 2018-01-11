"""
Taken and adapted from https://github.com/aubio/aubio/blob/master/python/demos/demo_onset_plot.py
"""

import numpy as np
import aubio


def get_onsets(input_wav, onset_thresh, win_s=512, max_read_sec=None):
    hop_s = win_s // 2  # hop size

    # set audio source
    s = aubio.source(input_wav, 0, hop_s)
    samplerate = s.samplerate

    if max_read_sec:
        max_read_samples = samplerate * max_read_sec
    else:
        max_read_samples = None

    # create onset detector
    o = aubio.onset("default", win_s, hop_s, samplerate)
    o.set_threshold(onset_thresh)

    # list of onsets: denotes the samples at which an onset occured
    onsets = []

    # storage for plotted data
    desc = []   # the "green line" in the plot. a descriptor for each frame of size `hop_s`
    # tdesc = []
    # allsamples_max = np.zeros(0,)
    # downsample = 2  # to plot n samples / hop_s

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        if o(samples):
            print("onset at sec. %f" % (o.get_last_s()))
            onsets.append(o.get_last())
        # keep some data to plot it later
        # new_maxes = (abs(samples.reshape(hop_s//downsample, downsample))).max(axis=0)
        # allsamples_max = np.hstack([allsamples_max, new_maxes])
        desc.append(o.get_descriptor())
        # tdesc.append(o.get_thresholded_descriptor())
        total_frames += read
        if read < hop_s or (max_read_samples is not None and total_frames >= max_read_samples): break

    # return onsets, desc, tdesc, allsamples_max
    return onsets, desc
