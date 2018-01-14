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


def get_onsets(source, samplerate, onset_thresh, onset_ampl_window=10, max_read_sec=None):
    """
    Get onset markers from `source` with `samplerate` using onset detection threshold `onset_thresh`,
    which is a value between [0, 1]. Optionally only read `max_read_sec` sec. from `source`.
    Returns two NumPy arrays as tuple:
    - an array of N detected onset markers as sample numbers
    - an array of N maximum amplitudes per onset section (normalized to [0, 1])
    - an array of M = samples/HOP_S sample frame descriptors
    """
    if max_read_sec:
        max_read_samples = samplerate * max_read_sec
    else:
        max_read_samples = None

    # create onset detector
    o = aubio.onset("default", WIN_S, HOP_S, samplerate)
    o.set_threshold(onset_thresh)

    # list of onsets: denotes the samples at which an onset occurred
    onsets = []
    onset_max_ampl = []   # maximum amplitude within this onset section

    # storage for plotted data
    desc = []   # the "green line" in the plot. a descriptor for each frame of size `hop_s`
    # tdesc = []
    # allsamples_max = []

    # total number of frames read
    total_frames = 0
    cur_onset_max_ampl = 0
    #last_onset_hop = None
    while True:
        samples, read = source()
        # new_maxes = (abs(samples.reshape(HOP_S // DOWNSAMPLE, DOWNSAMPLE))).max(axis=0)
        # allsamples_max.extend(new_maxes)
        ampl = o.get_descriptor()
        desc.append(ampl)
        cur_onset_max_ampl = max(cur_onset_max_ampl, ampl)

        if o(samples):  # onset detected
            print("onset at sec. %f with max. amplitude %f" % (o.get_last_s(), cur_onset_max_ampl))
            onsets.append(o.get_last())
            onset_max_ampl.append(cur_onset_max_ampl)
            cur_onset_max_ampl = 0

        # tdesc.append(o.get_thresholded_descriptor())
        total_frames += read
        if read < HOP_S or (max_read_samples is not None and total_frames >= max_read_samples): break

    onset_max_ampl = np.array(onset_max_ampl)
    onset_max_norm = np.max(onset_max_ampl) or 1
    onset_max_ampl /= onset_max_norm    # normalize

    # return onsets, desc, tdesc, allsamples_max
    return np.array(onsets), onset_max_ampl, np.array(desc) #, np.array(allsamples_max)


def plot_onsets(onsets, onset_max_ampl, desc, samplerate):
    import matplotlib.pyplot as plt

    # allsamples_max = (allsamples_max > 0) * allsamples_max
    # allsamples_max_times = [float(t) * HOP_S / DOWNSAMPLE / samplerate for t in range(len(allsamples_max))]

    desc_times = [float(t) * HOP_S / samplerate for t in range(len(desc))]
    desc_max = max(desc) if max(desc) != 0 else 1.
    desc_plot = [d / desc_max for d in desc]
    for stamp, ampl in zip(onsets, onset_max_ampl):
        stamp /= float(samplerate)
        plt.plot([stamp, stamp], [0, ampl], '-r', linewidth=3.0)

    plt.plot(desc_times, desc_plot, '-g')
    plt.axis(ymin=0, ymax=max(desc_plot))
    plt.xlabel('time (s)')
    plt.show()



source, samplerate = open_audio_source('audio/kiriloff-fortschritt-unmastered.wav')
onsets, onset_max_ampl, desc = get_onsets(source, samplerate, 0.3, max_read_sec=10)
plot_onsets(onsets, onset_max_ampl, desc, samplerate)

onsets_sec = onsets / samplerate
desc_times = [float(t) * HOP_S / samplerate for t in range(len(desc))]
