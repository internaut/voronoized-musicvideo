"""
Taken and adapted from https://github.com/aubio/aubio/blob/master/python/demos/demo_onset_plot.py
"""

import numpy as np
import aubio

DOWNSAMPLE = 2   # to plot n samples / hop_s
WIN_S = 512
HOP_S = WIN_S // 2  # hop size


def open_audio_source(input_wav):
    s = aubio.source(input_wav, 0, HOP_S)
    samplerate = s.samplerate
    return s, samplerate


def get_onsets(source, samplerate, onset_thresh, max_read_sec=None):
    if max_read_sec:
        max_read_samples = samplerate * max_read_sec
    else:
        max_read_samples = None

    # create onset detector
    o = aubio.onset("default", WIN_S, HOP_S, samplerate)
    o.set_threshold(onset_thresh)

    # list of onsets: denotes the samples at which an onset occured
    onsets = []

    # storage for plotted data
    desc = []   # the "green line" in the plot. a descriptor for each frame of size `hop_s`
    # tdesc = []
    # allsamples_max = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = source()
        if o(samples):
            print("onset at sec. %f" % (o.get_last_s()))
            onsets.append(o.get_last())
        new_maxes = (abs(samples.reshape(HOP_S // DOWNSAMPLE, DOWNSAMPLE))).max(axis=0)
        # allsamples_max.extend(new_maxes)
        desc.append(o.get_descriptor())
        # tdesc.append(o.get_thresholded_descriptor())
        total_frames += read
        if read < HOP_S or (max_read_samples is not None and total_frames >= max_read_samples): break

    # return onsets, desc, tdesc, allsamples_max
    return np.array(onsets), np.array(desc) #, np.array(allsamples_max)


def plot_onsets(onsets, desc, samplerate):
    import matplotlib.pyplot as plt

    # allsamples_max = (allsamples_max > 0) * allsamples_max
    # allsamples_max_times = [float(t) * HOP_S / DOWNSAMPLE / samplerate for t in range(len(allsamples_max))]

    desc_times = [float(t) * HOP_S / samplerate for t in range(len(desc))]
    desc_max = max(desc) if max(desc) != 0 else 1.
    desc_plot = [d / desc_max for d in desc]
    for stamp in onsets:
        stamp /= float(samplerate)
        plt.plot([stamp, stamp], [0, max(desc_plot)], '-r')

    plt.plot(desc_times, desc_plot, '-g')
    plt.axis(ymin=0, ymax=max(desc_plot))
    plt.xlabel('time (s)')
    plt.show()


source, samplerate = open_audio_source('audio/kiriloff-fortschritt-unmastered.wav')
onsets, desc = get_onsets(source, samplerate, 0.3, max_read_sec=10)
plot_onsets(onsets, desc, samplerate)