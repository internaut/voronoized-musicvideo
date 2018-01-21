import cv2
import numpy as np
from moviepy.editor import VideoClip, AudioFileClip, AudioClip

from onsets import open_audio_source, get_onsets


CLIP_SEC = 10
CLIP_FPS = 24
CLIP_W = 640
CLIP_H = 480

#source, samplerate = open_audio_source('audio/kiriloff-fortschritt-unmastered.wav')
#onsets, onset_max_ampl, _ = get_onsets(source, samplerate, 0.3, max_read_sec=CLIP_SEC)
#assert len(onsets) == len(onset_max_ampl)


def make_video_frame(t):
    frame = np.full((CLIP_H, CLIP_W, 3), (0, 0, 0), np.uint8)
    return frame

def make_audio_frame(t):
    return 2 * [np.sin(440 * 2 * np.pi * t)]


#audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.mp3')
audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')
#audioclip = AudioClip(make_audio_frame, duration=CLIP_SEC)
#audioclip.preview()

clip = VideoClip(make_video_frame, duration=CLIP_SEC)
audioclip = audioclip.set_duration(clip.duration)
clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
#clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS, audio_codec='pcm_s16le', audio_fps=samplerate)
# clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS, audio='audio/kiriloff-fortschritt-unmastered.wav', audio_fps=samplerate)



sec = 10
fps = 24
n_frames = fps*sec
w = 1000
h = 1000
r_min = 1
r_max = h/2
impulse = 100
decay = 1        # linear decay of r per frame
#writer = cv2.VideoWriter('opencv_video.mkv', cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
writer = cv2.VideoWriter('out/opencv_video_test.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center = (int(round(w/2)), int(round(h/2)))
onset_sec  = np.array([1.0, 2.0, 5.0, 5.5])
onset_frames = onset_sec * fps
onset_frames = onset_frames.round().astype(np.int)
onset_ampl = np.array([0.3, 0.2, 1.0, 0.9])
onset_frame_ampl = dict(zip(onset_frames, onset_ampl))

r = 0
for i in range(n_frames):
    print('frame %d' % (i+1))

    frame = np.full((w, h, 3), (0, 0, 0), np.uint8)

    radd = onset_frame_ampl.get(i, 0) * impulse
    r = r + radd - decay
    r = max(min(r, r_max), r_min)

    cv2.circle(frame, center, int(round(r)), (255, 0, 0), -1)

    writer.write(frame)

