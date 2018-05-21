"""
Taken and adapted from https://github.com/aubio/aubio/blob/master/python/demos/demo_onset_plot.py
"""

import sys
import pickle

import numpy as np
import aubio

DOWNSAMPLE = 2   # to plot n samples / hop_s
WIN_S = 512
HOP_S = WIN_S // 2  # hop size


def open_audio_source(input_wav):
    """
    Open the audio source file `input_wav` and return a tuple with
    - the aubio source object
    - its samplerate
    """
    s = aubio.source(input_wav, 0, HOP_S)
    samplerate = s.samplerate
    return s, samplerate


def get_onsets(source, samplerate, onset_thresh, onset_ampl_window=20, max_read_sec=None):
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
    last_onset_hop = None
    cur_hop = 0
    while True:
        samples, read = source()
        # new_maxes = (abs(samples.reshape(HOP_S // DOWNSAMPLE, DOWNSAMPLE))).max(axis=0)
        # allsamples_max.extend(new_maxes)
        ampl = o.get_descriptor()
        desc.append(ampl)

        if o(samples):  # onset detected
            print("> onset at sec. %f" % o.get_last_s())
            onsets.append(o.get_last())
            last_onset_hop = cur_hop

        cur_hop += 1

        if cur_hop - onset_ampl_window//2 == last_onset_hop:
            cur_onset_max_ampl = max(desc[cur_hop-onset_ampl_window:cur_hop])
            onset_max_ampl.append(cur_onset_max_ampl)

        # tdesc.append(o.get_thresholded_descriptor())
        total_frames += read

        if read < HOP_S or (max_read_samples is not None and total_frames >= max_read_samples): break

    n_max_ampl_missing = len(onsets) - len(onset_max_ampl)

    if n_max_ampl_missing > 0:   # if the last onset amplitude is missing
        onset_max_ampl.extend([max(desc[last_onset_hop:])] * n_max_ampl_missing)

    onset_max_ampl = np.array(onset_max_ampl)
    onset_max_norm = np.max(onset_max_ampl) or 1
    onset_max_ampl /= onset_max_norm    # normalize

    # return onsets, onset amplitudes, frame descriptors
    return np.array(onsets), onset_max_ampl, np.array(desc) #, np.array(allsamples_max)


def plot_onsets(ax, onsets, onset_max_ampl, desc, samplerate):
    # allsamples_max = (allsamples_max > 0) * allsamples_max
    # allsamples_max_times = [float(t) * HOP_S / DOWNSAMPLE / samplerate for t in range(len(allsamples_max))]

    desc_times = [float(t) * HOP_S / samplerate for t in range(len(desc))]
    desc_max = max(desc) if max(desc) != 0 else 1.
    desc_plot = [d / desc_max for d in desc]
    for stamp, ampl in zip(onsets, onset_max_ampl):
        stamp /= float(samplerate)
        ax.plot([stamp, stamp], [0, ampl], '-r', linewidth=3.0)

    ax.plot(desc_times, desc_plot, '-g')
    ax.axis(ymin=0, ymax=max(desc_plot))
    ax.set_xlabel('time (s)')


if __name__ == '__main__':
    n_args = len(sys.argv)

    if n_args < 3:
        print('run script as: %s <audio input file> <onsets output pickle file> '
              '[plot output file] [number of seconds to read]'
              % sys.argv[0], file=sys.stderr)
        exit(1)

    audio_file, pickle_file = sys.argv[1:3]

    if n_args >= 4:
        plot_file = sys.argv[3]
    else:
        plot_file = None

    if n_args >= 5:
        max_read_sec = int(sys.argv[4])
    else:
        max_read_sec = None

    print('reading audio file "%s" (%s)'
          % (audio_file, 'complete' if max_read_sec is None else str(max_read_sec) + 'sec.'))
    source, samplerate = open_audio_source(audio_file)

    print('sample rate is %d' % samplerate)

    print('detecting onsets...')
    onsets, onset_max_ampl, frame_desc = get_onsets(source, samplerate, 0.3, max_read_sec=max_read_sec)
    assert len(onsets) == len(onset_max_ampl)

    print('writing output file to "%s"' % pickle_file)

    with open(pickle_file, 'wb') as f:
        pickle.dump((samplerate, onsets, onset_max_ampl, frame_desc), f)

    if plot_file:
        import matplotlib.pyplot as plt

        print('plotting onsets...')
        fig, ax = plt.subplots()
        plot_onsets(ax, onsets, onset_max_ampl, frame_desc, samplerate)

        plt.tight_layout()
        print('saving plot output to "%s"' % plot_file)
        fig.savefig(plot_file)
        fig.show()

