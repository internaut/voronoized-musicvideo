import cv2
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, AudioClip

from onsets import open_audio_source, get_onsets


CLIP_SEC = 10
CLIP_FPS = 24
CLIP_W = 640
CLIP_H = 480

r_min = 1
r_max = CLIP_H/2
impulse = 30
decay = 1        # linear decay of r per frame

source, samplerate = open_audio_source('audio/kiriloff-fortschritt-unmastered.wav')
onsets, onset_max_ampl, _ = get_onsets(source, samplerate, 0.3, max_read_sec=CLIP_SEC)
assert len(onsets) == len(onset_max_ampl)

# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

r = 0
center = (int(round(CLIP_W/2)), int(round(CLIP_H/2)))


def make_video_frame(t):
    global r

    fnum = round(t * CLIP_FPS)   # frame number

    frame = np.full((CLIP_H, CLIP_W, 3), (0, 0, 0), np.uint8)

    onset_ampl = onset_frame_ampl.get(fnum, 0)
    radd = onset_ampl * impulse
    r = r + radd - decay
    r = max(min(r, r_max), r_min)

    cv2.circle(frame, center, int(round(r)), (255, 0, 0), -1)

    return frame


audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')

clip = VideoClip(make_video_frame, duration=CLIP_SEC)
audioclip = audioclip.set_duration(clip.duration)
clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
