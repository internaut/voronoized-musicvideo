import pickle

import cv2
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, AudioClip
import moviepy.video.fx.all as vfx

from voronoize import features_from_img, voronoi_from_feature_samples, lines_for_voronoi, draw_lines
from helpers import create_frame

CLIP_SEC = 10
CLIP_FPS = 24
CLIP_W = 640
#CLIP_H = 480


class VideoFrameGenerator:
    def __init__(self, input_clip, onset_frame_ampl, fps=CLIP_FPS):
        self.input_clip = input_clip
        self.onset_frame_ampl = onset_frame_ampl
        self.fps = fps

        self.w, self.h = input_clip.size
        self.r = 0

        self.r_min = 1
        self.r_max = self.h / 2
        self.impulse = 30
        self.decay = 1  # linear decay of r per frame

        self.vor_lines_alpha_decay_basefactor = 1.0
        self.vor_lines = []    # holds tuples (voronoi lines frame, current alpha, alpha decay factor)

    def make_video_frame(self, t):
        fnum = round(t * self.fps)   # frame number

        in_frame = self.input_clip.get_frame(t)

        bin_frame, features = features_from_img(in_frame, blur_radius=5)
        out_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)

        if fnum == 24:
            vor = voronoi_from_feature_samples(features, 1000)
            lines = lines_for_voronoi(vor, self.w, self.h)
            alpha_decay = self.vor_lines_alpha_decay_basefactor * 0.01
            vor_lines_frame = create_frame(self.w, self.h)
            draw_lines(vor_lines_frame, lines, (255, 0, 0))
            self.vor_lines.append((vor_lines_frame, 1.0, alpha_decay))

        self._update_voronoi_lines(out_frame)

        return out_frame

    def _update_voronoi_lines(self, frame):
        tmp_vor_lines = []
        for lines_frame, alpha, alpha_decay in self.vor_lines:
            cv2.scaleAdd(lines_frame, alpha, frame, dst=frame)
            #cv2.addWeighted(frame, 1.0, lines_frame, alpha, 0.0, dst=frame)

            alpha -= alpha_decay
            if alpha > 0:
                tmp_vor_lines.append((lines_frame, alpha, alpha_decay))

        self.vor_lines = tmp_vor_lines

        # #frame = np.full((CLIP_H, CLIP_W, 3), (0, 0, 0), np.uint8)
        #
        # onset_ampl = self.onset_frame_ampl.get(fnum, 0)
        # radd = onset_ampl * self.impulse
        # self.r = self.r + radd - self.decay
        # self.r = max(min(self.r, self.r_max), self.r_min)
        # self.r = int(round(self.r))
        #
        # cv2.blur(frame, (self.r, self.r), frame)
        #
        # #cv2.circle(frame, center, int(round(r)), (255, 0, 0), -1)
        #
        # return frame


with open('tmp/onsets.pickle', 'rb') as f:
    samplerate, onsets, onset_max_ampl, _ = pickle.load(f)
    assert len(onsets) == len(onset_max_ampl)

input_clip = VideoFileClip('video/stockvideotest.mp4', audio=False).subclip(0, CLIP_SEC)
input_clip = input_clip.fx(vfx.resize, width=CLIP_W)


# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

frame_gen = VideoFrameGenerator(input_clip, onset_frame_ampl)

audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')

clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=CLIP_SEC)
audioclip = audioclip.set_duration(clip.duration)
clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
