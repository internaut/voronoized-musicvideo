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


np.random.seed(123)

CLIP_FPS = 24
CLIP_W = 640
CLIP_H = 480


class VideoFrameGenerator:
    def __init__(self, scenes, onset_frame_ampl):
        assert scenes
        assert onset_frame_ampl

        self.clips = []
        self.cur_scene = None
        self.cur_clip = None
        self.clip_t = 0
        self.scenes = scenes
        self.w, self.h = self._setup_scenes()
        assert len(self.clips) == len(self.scenes)

        self.onset_frame_ampl = onset_frame_ampl

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        self.vor_lines = []

    def make_video_frame(self, t):
        fnum = int(round(t * CLIP_FPS))   # frame number

        self._update_cur_scene(t)

        onset_ampl = self.onset_frame_ampl.get(fnum, 0)

        jump = self.cur_scene.get('jump', None)

        if jump and onset_ampl >= jump['ampl']:
            if 'by' in jump:
                self.clip_t += jump['by']
                self.clip_t = max(self.clip_t, 0)
            elif 'by_random' in jump:
                self.clip_t += np.random.normal(0, jump['by_random'] * onset_ampl)
            else:
                self.clip_t = jump.get('to', 0)
        else:
            self.clip_t += 1 / CLIP_FPS

        in_frame = self.cur_clip.get_frame(self.clip_t)

        if in_frame.dtype != np.uint8:
            in_frame = in_frame.astype(np.uint8)

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)

        self.ctx = cairo.Context(self.surface)

        if self.cur_scene['mode'] == 'original':
            return in_frame
        elif self.cur_scene['mode'] == 'voronoi':
            return self._render_voronoi(in_frame, onset_ampl)
        else:
            return None

    def _render_voronoi(self, in_frame, onset_ampl):
        vor_opts = self.cur_scene['voronoi']

        _, bin_frame, features = features_from_img(in_frame,
                                                   blur_radius=5,
                                                   features_where=vor_opts.get('features_where', 0))

        base = self.cur_scene.get('base', None)
        if base == 'original':
            surface_base = in_frame[:, :, [2, 1, 0]]
            surface_base = np.dstack([surface_base, 255 * np.ones((self.h, self.w), dtype=np.uint8)])

            surface_data = np.frombuffer(self.surface.get_data(), np.uint8)
            surface_data += surface_base.flatten()
            self.surface.mark_dirty()
        else:
            self.ctx.set_source_rgba(0, 0, 0, 1)
            self.ctx.paint()

        if onset_ampl > 0:
            n_vor_features = int(round(vor_opts['lines_features_factor'] * onset_ampl))
            vor = voronoi_from_feature_samples(features, n_vor_features)
            lines = lines_for_voronoi(vor, self.w, self.h)

            alpha_decay = vor_opts['lines_alpha_decay_basefactor'] * (1.5-onset_ampl)
            initial_alpha = max(min(vor_opts['lines_initial_alpha_factor'] * onset_ampl, 1.0), 0.2)

            self.vor_lines.append((lines, initial_alpha, alpha_decay))

        self._update_voronoi_lines(in_frame)

        return nparray_from_surface(self.surface)

    def _update_voronoi_lines(self, baseframe):
        #self.ctx.set_operator(cairo.OPERATOR_MULTIPLY)    # TODO

        tmp_vor_lines = []
        for lines, lines_alpha, lines_alpha_decay in self.vor_lines:
            for a, b in lines:
                a, b = restrict_line(a, b, baseframe.shape[1]-1, baseframe.shape[0]-1)
                ax, ay = map(int, map(round, a))
                bx, by = map(int, map(round, b))
                a = (ax, ay)
                b = (bx, by)

                color = self.cur_scene['voronoi'].get('color', None)
                if color:
                    stroke = color + (lines_alpha, )
                else:
                    pix_a = tuple(list(baseframe[ay, ax, :] / 255) + [lines_alpha])
                    pix_b = tuple(list(baseframe[by, bx, :] / 255) + [lines_alpha])
                    stroke = gz.ColorGradient('linear', ((0, pix_a), (1, pix_b)), a, b)
                draw_lines(self.ctx, [(a, b)], stroke)

            lines_alpha -= lines_alpha_decay
            if lines_alpha > 0:
                tmp_vor_lines.append((lines, lines_alpha, lines_alpha_decay))

        self.vor_lines = tmp_vor_lines

    def _setup_scenes(self):
        base_size = None
        for sc_def in self.scenes:
            clip = VideoFileClip(os.path.join('video', sc_def['video']), audio=False)
            subclip_markers = sc_def.get('subclip')
            if subclip_markers:
                clip = clip.subclip(*subclip_markers)
            clip = clip.fx(vfx.resize, width=CLIP_W)
            if not base_size:
                base_size = clip.size
            elif base_size and (base_size[0] < clip.size[0] or base_size[1] < clip.size[1]):
                clip = clip.fx(vfx.crop,
                               x_center=clip.size[0]//2, y_center=clip.size[1]//2,
                               width=base_size[0], height=base_size[1])

            self.clips.append(clip)

        return base_size

    def _update_cur_scene(self, t):
        for i, sc_def in enumerate(self.scenes):
            scene_t = sc_def['t']
            if scene_t[0] <= t < scene_t[1] and self.cur_clip is not self.clips[i]:
                self.cur_scene = sc_def
                self.cur_clip = self.clips[i]
                self.clip_t = 0
                break



with open('tmp/onsets.pickle', 'rb') as f:
    samplerate, onsets, onset_max_ampl, _ = pickle.load(f)
    assert len(onsets) == len(onset_max_ampl)

scenes = [
    {
        'video': '00156.MTS',
        'mode': 'original',
        't': (0, 24.5),
        'jump': {
            'ampl': 0.02,
            'by_random': 20,
        }
    },
    {
        'video': '00151.MTS',
        'mode': 'voronoi',
        'base': 'original',
        't': (24.5, 58.212),
        'subclip': (5, 10),
        'jump': {
            'ampl': 0.05,
            'to': 0,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 10000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.01,
            'features_where': 0
        },
    },
]

# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

frame_gen = VideoFrameGenerator(scenes, onset_frame_ampl)

clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=58)

audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')
audioclip = audioclip.set_duration(clip.duration)

clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
