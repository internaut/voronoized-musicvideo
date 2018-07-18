"""
Main video rendering script.

2018, Markus Konrad <post@mkonrad.net>
"""

import sys
import pickle
import os

import cairocffi as cairo
import gizeh as gz
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip

from voronoize import features_from_img, voronoi_from_feature_samples, lines_for_voronoi
from graphics import draw_lines, nparray_from_surface
from helpers import restrict_line
from conf import SCENES, CLIP_FPS


INPUT_AUDIO = 'audio/kiriloff-fortschritt-master2.wav'
INPUT_ONSETS = 'tmp/onsets.pickle'
OUTPUT_VIDEO = 'out/kiriloff_fortschritt.mp4'

if len(sys.argv) >= 2:
    override_duration = int(sys.argv[1])
else:
    override_duration = None


np.random.seed(123)


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
        self.blank_frame = np.zeros((self.h, self.w, 3))
        assert len(self.clips) == len(self.scenes)

        self.onset_frame_ampl = onset_frame_ampl

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        self.vor_lines = []

    def make_video_frame(self, t):
        fnum = int(round(t * CLIP_FPS))   # frame number

        self._update_cur_scene(t)

        if self.cur_scene['mode'] is None:
            return self.blank_frame

        onset_ampl = self.onset_frame_ampl.get(fnum, 0)

        jump = self.cur_scene.get('jump', None)

        if jump and onset_ampl >= jump['ampl'] and self.onset_frame_ampl.get(fnum-1, 0) < jump['ampl'] and self.clip_t > 0:
            if 'by' in jump:
                self.clip_t += jump['by']
                self.clip_t = max(self.clip_t, 0)
            elif 'by_random' in jump:
                rand_jump = np.random.normal(0, jump['by_random'] * onset_ampl)
                if abs(rand_jump) < 1 / CLIP_FPS:
                    rand_jump = 1 / CLIP_FPS
                self.clip_t += rand_jump
            else:
                self.clip_t = jump.get('to', 0)
        else:
            self.clip_t += 1 / CLIP_FPS

        if self.clip_t > self.cur_clip.duration:
            self.clip_t = 0

        in_frame = self.cur_clip.get_frame(self.clip_t)

        if in_frame.dtype != np.uint8:
            in_frame = in_frame.astype(np.uint8)

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)

        self.ctx = cairo.Context(self.surface)

        if self.cur_scene['mode'] == 'original':
            out_frame = in_frame
        elif self.cur_scene['mode'] == 'voronoi':
            out_frame = self._render_voronoi(in_frame, onset_ampl, t)
        else:
            out_frame = None

        if out_frame is not None:
            fade = self.cur_scene.get('fade', None)
            if fade and t >= fade['start_t']:
                fade_color = fade.get('color', 'black')
                fade_dt = (t - fade['start_t']) / (fade['end_t'] - fade['start_t'])
                if fade_color == 'white':
                    fade_frame = out_frame.astype(np.float) + np.full((self.h, self.w, 3), fade_dt * 255,
                                                                      dtype=np.float)
                    out_frame = fade_frame.clip(0, 255).astype(np.uint8)
                else:
                    out_frame = (out_frame.astype(np.float) * (1-fade_dt)).astype(np.uint8)

        return out_frame

    def _render_voronoi(self, in_frame, onset_ampl, t):
        vor_opts = self.cur_scene['voronoi']

        _, bin_frame, features = features_from_img(in_frame,
                                                   blur_radius=5,
                                                   features_where=vor_opts.get('features_where', 0))

        base = self.cur_scene.get('base', None)
        if base in ('original', 'bin'):
            if base == 'original':
                surface_base = in_frame[:, :, [2, 1, 0]]   # correct channel order
            else:   # bin
                # copy binary image data to all 3 channels
                surface_base = np.repeat(bin_frame[:, :, np.newaxis], 3, axis=2)

            # add alpha channel
            surface_base = np.dstack([surface_base, 255 * np.ones((self.h, self.w), dtype=np.uint8)])

            # set data
            surface_data = np.frombuffer(self.surface.get_data(), np.uint8)
            surface_data += surface_base.flatten()
            self.surface.mark_dirty()
        else:
            if isinstance(base, tuple):
                base_color = base + (1, )
            else:
                base_color = (0, 0, 0, 1)

            self.ctx.set_source_rgba(*base_color)
            self.ctx.paint()

        if onset_ampl > 0:
            if 'lines_features_factor_fade' in vor_opts:
                ff_fade = vor_opts['lines_features_factor_fade']
                ff_fade_from, ff_fade_to = ff_fade['from_to']
                ff_fade_delta = ff_fade_to - ff_fade_from
                ff_fade_dt = (t - ff_fade['start_t']) / (ff_fade['end_t'] - ff_fade['start_t'])
                features_factor = ff_fade_from + ff_fade_dt * ff_fade_delta
            else:
                features_factor = vor_opts['lines_features_factor']

            n_vor_features = int(round(features_factor * onset_ampl))
            vor = voronoi_from_feature_samples(features, n_vor_features)
            lines = lines_for_voronoi(vor, self.w, self.h)

            alpha_decay = vor_opts['lines_alpha_decay_basefactor'] * (1.5-onset_ampl)
            initial_alpha = max(min(vor_opts['lines_initial_alpha_factor'] * onset_ampl, 1.0), 0.2)

            self.vor_lines.append((lines, initial_alpha, alpha_decay))

        self._update_voronoi_lines(in_frame)

        return nparray_from_surface(self.surface)

    def _update_voronoi_lines(self, baseframe):
        tmp_vor_lines = []
        for lines, lines_alpha, lines_alpha_decay in self.vor_lines:
            for a, b in lines:
                a, b = restrict_line(a, b, baseframe.shape[1]-1, baseframe.shape[0]-1)
                a, b = map(np.array, (a, b))
                a[np.isnan(a)] = baseframe.shape[1]-1
                b[np.isnan(b)] = baseframe.shape[0]-1
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
        for i, sc_def in enumerate(self.scenes):
            video_basename, _ = os.path.splitext(sc_def['video'])
            inputfile = os.path.join('video', str(i + 1).zfill(2) + '_' + video_basename + '.mp4')
            clip = VideoFileClip(inputfile, audio=False)
            # subclip_markers = sc_def.get('subclip')
            # if subclip_markers:
            #     clip = clip.subclip(*subclip_markers)
            # clip = clip.fx(vfx.resize, width=CLIP_W)
            if not base_size:
                base_size = clip.size
            # elif base_size and (base_size[0] < clip.size[0] or base_size[1] < clip.size[1]):
            #     clip = clip.fx(vfx.crop,
            #                    x_center=clip.size[0]//2, y_center=clip.size[1]//2,
            #                    width=base_size[0], height=base_size[1])

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


with open(INPUT_ONSETS, 'rb') as f:
    samplerate, onsets, onset_max_ampl, _ = pickle.load(f)
    assert len(onsets) == len(onset_max_ampl)

# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

audioclip = AudioFileClip(INPUT_AUDIO)

if override_duration:
    duration = override_duration
else:
    duration = audioclip.duration

print('will generate %d sec. of video' % duration)
print('using audio %s' % INPUT_AUDIO)
print('using onsets %s' % INPUT_ONSETS)

frame_gen = VideoFrameGenerator(SCENES, onset_frame_ampl)
gen_clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=duration)

introtext = "kiriloff – fortschritt"
#videomaterial: nico stelljes, martin schultze
#programmierung: markus konrad (https://github.com/internaut/fortschritt)"""
introtext_clip = TextClip(introtext,
                          color='white',
                          font='Menlo-Bold',
                          fontsize=20,
                          method='caption',
                          size=(frame_gen.w, frame_gen.h))

main_clip = CompositeVideoClip([
    gen_clip,
    introtext_clip.set_start(0.5).set_end(6.5).crossfadein(0.5).crossfadeout(0.5)
])

main_clip = main_clip.set_audio(audioclip).set_duration(gen_clip.duration)
main_clip.write_videofile(OUTPUT_VIDEO, fps=CLIP_FPS)
