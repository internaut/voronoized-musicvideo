import pickle
import os

import cairocffi as cairo
import gizeh as gz
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, AudioClip
import moviepy.video.fx.all as vfx

from voronoize import features_from_img, voronoi_from_feature_samples, lines_for_voronoi
from graphics import draw_lines, nparray_from_surface
from helpers import restrict_line

CLIP_FPS = 24
CLIP_W = 640
CLIP_H = 480


class VideoFrameGenerator:
    def __init__(self, onset_frame_ampl):
        self.onset_frame_ampl = onset_frame_ampl

        self.w = CLIP_W
        self.h = CLIP_H

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        self.vor_lines = []

        self.clip_t = 0
        clip = VideoFileClip(os.path.join('video', 'live.3gp'), audio=False)
        clip = clip.subclip(10)
        clip = clip.fx(vfx.resize, width=CLIP_W)
        self.cur_clip = clip

    def make_video_frame(self, t):
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        fnum = int(round(t * CLIP_FPS))   # frame number

        onset_ampl = self.onset_frame_ampl.get(fnum, 0)

        self.clip_t += 1 / CLIP_FPS

        in_frame = self.cur_clip.get_frame(self.clip_t)

        if in_frame.dtype != np.uint8:
            in_frame = in_frame.astype(np.uint8)

        _, bin_frame, features = features_from_img(in_frame,
                                                   blur_radius=5,
                                                   features_where=255)

        self.ctx.set_source_rgba(0, 0, 0, 1)
        self.ctx.paint()

        if onset_ampl > 0:
            n_vor_features = int(round(20000 * onset_ampl))
            vor = voronoi_from_feature_samples(features, n_vor_features)
            lines = lines_for_voronoi(vor, self.w, self.h)

            alpha_decay = 0.01 * (1.5-onset_ampl)
            initial_alpha = max(min(2.0 * onset_ampl, 1.0), 0.2)

            self.vor_lines.append((lines, initial_alpha, alpha_decay))

#        in_frame_rgba = np.concatenate((in_frame, np.full((in_frame.shape[0], in_frame.shape[1], 1), 255,
#                                                          dtype=in_frame.dtype)), axis=2)
        self._update_voronoi_lines(in_frame)

        return nparray_from_surface(self.surface)

    def _update_voronoi_lines(self, baseframe):
        #self.ctx.set_operator(cairo.OPERATOR_MULTIPLY)    # TODO

        tmp_vor_lines = []
        for lines, lines_alpha, lines_alpha_decay in self.vor_lines:
            #alphaframe = baseframe.astype(np.float)
            #alphaframe = baseframe.copy()
            #alphaframe[:,:,3] = int(round(lines_alpha * 255))   # TODO: das funktioniert noch nicht
            #alphaframe[:,:,3] = lines_alpha
            #stroke = gz.ImagePattern(alphaframe)
            #draw_lines(self.ctx, lines, stroke)

            for a, b in lines:
                a, b = restrict_line(a, b, baseframe.shape[1]-1, baseframe.shape[0]-1)
                ax, ay = map(int, map(round, a))
                bx, by = map(int, map(round, b))
                a = (ax, ay)
                b = (bx, by)

                #pix_a = (baseframe[ay, ax, :] / 255) * lines_alpha
                #pix_b = (baseframe[by, bx, :] / 255) * lines_alpha
                pix_a = tuple(list(baseframe[ay, ax, :] / 255) + [lines_alpha])
                pix_b = tuple(list(baseframe[by, bx, :] / 255) + [lines_alpha])
                stroke = gz.ColorGradient('linear', ((0, pix_a), (1, pix_b)), a, b)
                draw_lines(self.ctx, [(a, b)], stroke)

            lines_alpha -= lines_alpha_decay
            if lines_alpha > 0:
                tmp_vor_lines.append((lines, lines_alpha, lines_alpha_decay))

        self.vor_lines = tmp_vor_lines



with open('tmp/onsets.pickle', 'rb') as f:
    samplerate, onsets, onset_max_ampl, _ = pickle.load(f)
    assert len(onsets) == len(onset_max_ampl)

# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

frame_gen = VideoFrameGenerator(onset_frame_ampl)

clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=10)

audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')
audioclip = audioclip.set_duration(clip.duration)

clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
