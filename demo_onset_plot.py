"""
Demo for finding onsets in audio. Shows a plot of the found onsets.
Copied from https://github.com/aubio/aubio/blob/master/python/demos/demo_onset_plot.py and modified.

2018, Markus Konrad <post@mkonrad.net>
"""

import sys
from aubio import onset, source
from numpy import hstack, zeros

win_s = 512                 # fft size
hop_s = win_s // 2          # hop size

if len(sys.argv) < 2:
    print("Usage: %s <filename> [samplerate] [max. read sec.]" % sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]

samplerate = 0
max_read_sec = None
if len(sys.argv) > 2: samplerate = int(sys.argv[2])
if len(sys.argv) > 3: max_read_sec = int(sys.argv[3])


s = source(filename, samplerate, hop_s)
samplerate = s.samplerate
o = onset("default", win_s, hop_s, samplerate)
print('thresh', o.get_threshold())
o.set_threshold(0.3)

if max_read_sec:
    max_read_samples = samplerate * max_read_sec
    print('max read: %d sec, %d samples' % (max_read_sec, max_read_samples))
else:
    max_read_samples = None

# list of onsets, in samples
onsets = []

# storage for plotted data
desc = []
tdesc = []
allsamples_max = zeros(0,)
downsample = 2  # to plot n samples / hop_s

# total number of frames read
total_frames = 0
while True:
    samples, read = s()
    if o(samples):
        print("%f" % (o.get_last_s()))
        onsets.append(o.get_last())
    # keep some data to plot it later
    new_maxes = (abs(samples.reshape(hop_s//downsample, downsample))).max(axis=0)
    allsamples_max = hstack([allsamples_max, new_maxes])
    desc.append(o.get_descriptor())
    tdesc.append(o.get_thresholded_descriptor())
    total_frames += read
    if read < hop_s or (max_read_samples is not None and total_frames >= max_read_samples): break

# do plotting
import matplotlib.pyplot as plt
allsamples_max = (allsamples_max > 0) * allsamples_max
allsamples_max_times = [ float(t) * hop_s / downsample / samplerate for t in range(len(allsamples_max)) ]
plt1 = plt.axes([0.1, 0.75, 0.8, 0.19])
plt2 = plt.axes([0.1, 0.1, 0.8, 0.65], sharex = plt1)
plt.rc('lines',linewidth='.8')
for stamp in onsets:
    stamp /= float(samplerate)
    plt1.plot([stamp, stamp], [-1., 1.], '-r')
plt1.axis(xmin = 0., xmax = max(allsamples_max_times) )
plt1.xaxis.set_visible(False)
plt1.yaxis.set_visible(False)
desc_times = [ float(t) * hop_s / samplerate for t in range(len(desc)) ]
desc_max = max(desc) if max(desc) != 0 else 1.
desc_plot = [d / desc_max for d in desc]
tdesc_plot = [d / desc_max for d in tdesc]
for stamp in onsets:
    stamp /= float(samplerate)
    plt2.plot([stamp, stamp], [min(tdesc_plot), max(desc_plot)], '-r')

plt1.plot(allsamples_max_times,  allsamples_max, '-b')
plt1.plot(allsamples_max_times, -allsamples_max, '-b')
plt2.plot(desc_times, desc_plot, '-g')
plt2.plot(desc_times, tdesc_plot, '-y')
plt2.axis(ymin = min(tdesc_plot), ymax = max(desc_plot))
plt.xlabel('time (s)')
plt.savefig('tmp/demo_onsets.png', dpi=120)
plt.show()
