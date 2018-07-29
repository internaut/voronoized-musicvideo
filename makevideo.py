"""
Main video rendering script.

Run as: makevideo.py [clip duration]

2018, Markus Konrad <post@mkonrad.net>
"""

import sys
import pickle
import os
import math

import cairocffi as cairo
import gizeh as gz
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip

from voronoize import features_from_img, voronoi_from_feature_samples, lines_for_voronoi
from helpers import restrict_line, draw_lines, nparray_from_surface
from conf import SCENES, CLIP_FPS, STROKE_WIDTH


INPUT_AUDIO = 'audio/kiriloff-fortschritt-master2.wav'
INPUT_ONSETS = 'tmp/onsets.pickle'
OUTPUT_VIDEO = 'out/kiriloff_fortschritt.mp4'


# parse one optional parameter: video duration up to which the video should be rendered

if len(sys.argv) >= 2:
    try:
        override_duration = int(sys.argv[1])
    except ValueError:
        override_duration = None
else:
    override_duration = None

if len(sys.argv) >= 3:
    if ',' in sys.argv[2]:
        override_only_scene = set(s-1 for s in map(int, sys.argv[2].split(',')))
    else:
        override_only_scene = {int(sys.argv[2]) - 1}
else:
    override_only_scene = None

np.random.seed(123)


class VideoFrameGenerator:
    """
    Video frame generator class. Renders video frames according the a scene definition.
    """

    def __init__(self, scenes, onset_frame_ampl):
        """
        Initialize with a scene definition and onset amplitudes
        :param scenes: scene definition (as in conf.py)
        :param onset_frame_ampl: onset amplitudes (generated in onsets.py) -- a dict that holds the onset amplitude for
        a frame with an onset
        """

        assert scenes
        assert onset_frame_ampl

        # initializations
        self.clips = []         # holds the input video clip for each frame
        self.cur_scene = None   # current scene definition (dict)
        self.cur_scene_idx = None
        self.cur_clip = None    # input video clip used in current scene
        self.clip_t = 0         # current input frame time
        self.scenes = scenes    # scene definitions
        self.w, self.h = self._setup_scenes()               # output video width and height
        self.blank_frame = np.zeros((self.h, self.w, 3))    # a generic black frame
        assert len(self.clips) == len(self.scenes)

        self.onset_frame_ampl = onset_frame_ampl

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)  # the cairo surface to draw each frame
        self.ctx = cairo.Context(self.surface)      # the cairo context for the surface

        self.vor_lines = []     # holds voronoi lines that are currently rendered

    def make_video_frame(self, t):
        """
        Render a video frame for clip at time `t`.
        :param t: current output clip time
        :return: rendered frame as NumPy array or None (if blank output)
        """

        fnum = int(round(t * CLIP_FPS))   # frame number

        # set current scene settings for time `t`
        self._update_cur_scene(t)

        # shortcut if scene should not be rendered
        if self.cur_scene['mode'] is None or \
                (override_only_scene is not None and self.cur_scene_idx not in override_only_scene):
            return self.blank_frame

        # get current frame's onset amplitude
        # if there's no onset, the amplitude is set to 0
        onset_ampl = self.onset_frame_ampl.get(fnum, 0)

        # handle onset frame jumping:
        # if enabled for this scene, we can jump to a different input clip frame on an onset
        jump = self.cur_scene.get('jump', None)
        if jump and onset_ampl >= jump['ampl']\
                and self.onset_frame_ampl.get(fnum-1, 0) < jump['ampl']\
                and self.clip_t > 0:
            if 'by' in jump:   # jump forward/backward
                self.clip_t += jump['by']
                self.clip_t = max(self.clip_t, 0)
            elif 'by_random' in jump:   # jump randomly
                rand_jump = np.random.normal(0, jump['by_random'] * onset_ampl)
                if abs(rand_jump) < 1 / CLIP_FPS:
                    rand_jump = 1 / CLIP_FPS
                self.clip_t += rand_jump
            else:   # jump to first input clip frame
                self.clip_t = jump.get('to', 0)
        else:
            self.clip_t += 1 / CLIP_FPS

        if self.clip_t > self.cur_clip.duration:  # rewind if necessary
            self.clip_t = 0

        # retrieve the current input frame
        in_frame = self.cur_clip.get_frame(self.clip_t)

        # cast if necessary
        if in_frame.dtype != np.uint8:
            in_frame = in_frame.astype(np.uint8)

        # recreate surface and context for this frame
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        # render the frame depending on the "mode"
        if self.cur_scene['mode'] == 'original':        # use original input frame
            out_frame = in_frame
        elif self.cur_scene['mode'] == 'voronoi':       # render voronoi lines
            out_frame = self._render_voronoi(in_frame, onset_ampl, t)
        else:
            out_frame = None        # blank output

        # if enabled, apply fading
        if out_frame is not None:
            fade = self.cur_scene.get('fade', None)
            if fade and t >= fade['start_t']:
                # fade to color
                fade_color = fade.get('color', 'black')
                # time into fade
                fade_dt = (t - fade['start_t']) / (fade['end_t'] - fade['start_t'])
                if fade_color == 'white':
                    # add white frame and clip output
                    fade_frame = out_frame.astype(np.float) + np.full((self.h, self.w, 3), fade_dt * 255,
                                                                      dtype=np.float)
                    out_frame = fade_frame.clip(0, 255).astype(np.uint8)
                else:
                    # fade to black
                    out_frame = (out_frame.astype(np.float) * (1-fade_dt)).astype(np.uint8)

        return out_frame

    def _render_voronoi(self, in_frame, onset_ampl, t):
        """
        Render voronoi lines for an input frame with a given onset amplitude at clip time `t`.
        :param in_frame: input frame
        :param onset_ampl: onset amplitude
        :param t: clip time
        :return: rendered frame as NumPy array
        """
        # get the current scene's voronoi rendering options
        vor_opts = self.cur_scene['voronoi']

        # retrieve "features" from input frame
        # these are either the coordinates of black or white pixels (depending on the "features_where" parameter) in
        # the binarized input frame
        _, bin_frame, features = features_from_img(in_frame,
                                                   blur_radius=5,
                                                   features_where=vor_opts.get('features_where', 0))

        # get the base, i.e. background of the frame to be rendered
        base = self.cur_scene.get('base', None)
        if base in ('original', 'bin'):   # original or binarized version of input frame
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

            base_color = None
        else:   # solid color background
            if isinstance(base, tuple):
                base_color = base
            else:
                base_color = (0, 0, 0)

            alternating_base = self.cur_scene.get('alternating_base', None)
            if alternating_base:
                base_color = np.array(base_color)
                alt_base_freq = alternating_base['freq']
                alt_base_col = np.array(alternating_base['color_b'])
                assert len(base_color) == len(alt_base_col)
                alt_base_y = (1 + math.sin(alt_base_freq * self.clip_t)) / 2
                base_color = alt_base_y * base_color + (1-alt_base_y) * alt_base_col
                base_color = tuple(base_color)

            base_color = base_color + (1, )

            self.ctx.set_source_rgba(*base_color)
            self.ctx.paint()

        # if we have an onset, let's generate new voronoi lines from the features
        if onset_ampl > 0:
            # get the "features_factor" which controls the numbers of lines to be drawn
            if 'lines_features_factor_fade' in vor_opts:
                ff_fade = vor_opts['lines_features_factor_fade']
                ff_fade_from, ff_fade_to = ff_fade['from_to']
                ff_fade_delta = ff_fade_to - ff_fade_from
                ff_fade_dt = (t - ff_fade['start_t']) / (ff_fade['end_t'] - ff_fade['start_t'])
                features_factor = ff_fade_from + ff_fade_dt * ff_fade_delta
            else:
                features_factor = vor_opts['lines_features_factor']

            # sample the features and generate voronoi regions (stored in object `vor`)
            n_vor_features = int(round(features_factor * onset_ampl))
            vor = voronoi_from_feature_samples(features, n_vor_features)

            # generate voronoi lines from voronoi object `vor`
            lines = lines_for_voronoi(vor, self.w, self.h)

            # set initial alpha (transparency) value
            alpha_decay = vor_opts['lines_alpha_decay_basefactor'] * (1.5-onset_ampl)
            initial_alpha = max(min(vor_opts['lines_initial_alpha_factor'] * onset_ampl, 1.0), 0.2)

            # add these lines with their transparency settigns to the current list of of voronoi lines)
            self.vor_lines.append((lines, initial_alpha, alpha_decay))

        # update the current "alive" voronoi lines and their transparency and draw them to the input frame
        self._update_voronoi_lines(in_frame)

        out_frame = nparray_from_surface(self.surface)

        posteffect_opts = self.cur_scene.get('posteffect_destroy', None)
        if posteffect_opts and base_color is not None:
            # works only with monochrome bg:
            #line_px_mask = out_frame[:, :, 0] != int(base_color[0] * 255)
            line_px_mask = ~np.isclose(out_frame[:, :, 0], int(base_color[0] * 255), atol=1)
            line_pixels = out_frame[line_px_mask]
            px_destroy_factor = posteffect_opts['offset']\
                                + posteffect_opts['ampl'] * (1 + math.sin(posteffect_opts['freq'] * self.clip_t)) / 2
            out_frame[line_px_mask] = line_pixels * px_destroy_factor

        return out_frame

    def _update_voronoi_lines(self, baseframe):
        """
        update the current "alive" voronoi lines and their transparency and draw them to the input frame
        :param baseframe: frame on which the voronoi lines will be drawn
        """

        # def apply_alternating_color(color, t, opts):
        #     color = np.array(color)
        #     y = opts['offset'] + opts['ampl'] * (1 + math.sin(opts['freq'] * t)) / 2
        #     color = y * color
        #     if opts['clip']:
        #         color = np.clip(color, 0, 1)
        #     return tuple(color)

        # go through all the current sets of voronoi lines, construct the a-b-lines for the input frame,
        # set the color and draw the lines
        tmp_vor_lines = []
        for lines, lines_alpha, lines_alpha_decay in self.vor_lines:
            for a, b in lines:
                # restrict end points to current frame size
                a, b = restrict_line(a, b, baseframe.shape[1]-1, baseframe.shape[0]-1)
                a, b = map(np.array, (a, b))
                a[np.isnan(a)] = baseframe.shape[1]-1
                b[np.isnan(b)] = baseframe.shape[0]-1
                ax, ay = map(int, map(round, a))
                bx, by = map(int, map(round, b))
                a = (ax, ay)
                b = (bx, by)

                # get the color setting
                color = self.cur_scene['voronoi'].get('color', None)
                # alternating_color_opts = self.cur_scene['voronoi'].get('alternating_color', None)
                if color:   # add current alpha value for solid color
                    # if alternating_color_opts:
                    #     color = apply_alternating_color(color, self.clip_t, alternating_color_opts)

                    stroke = color + (lines_alpha, )
                else:       # make a color gradient between the pixels at the respective end points of the line
                    pix_a = tuple(baseframe[ay, ax, :] / 255)
                    pix_b = tuple(baseframe[by, bx, :] / 255)
                    # if alternating_color_opts:
                    #     pix_a = apply_alternating_color(pix_a, self.clip_t, alternating_color_opts)
                    #     pix_b = apply_alternating_color(pix_b, self.clip_t, alternating_color_opts)
                    # else:
                    #     pix_a = tuple(pix_a)
                    #     pix_b = tuple(pix_b)

                    pix_a = pix_a + (lines_alpha,)
                    pix_b = pix_b + (lines_alpha,)
                    stroke = gz.ColorGradient('linear', ((0, pix_a), (1, pix_b)), a, b)

                # draw the lines
                draw_lines(self.ctx, [(a, b)], stroke, stroke_width=STROKE_WIDTH)

            # decrease line transparency
            lines_alpha -= lines_alpha_decay

            # only retain lines that are visible (i.e. transparency above 0)
            if lines_alpha > 0:
                tmp_vor_lines.append((lines, lines_alpha, lines_alpha_decay))

        # retain lines for next frame
        self.vor_lines = tmp_vor_lines

    def _setup_scenes(self):
        """
        Setup the scenes and set input clips for each scene. Return output frame size.
        :return: output frame size
        """
        base_size = None
        for i, sc_def in enumerate(self.scenes):
            video_basename, _ = os.path.splitext(sc_def['video'])

            # input clip
            inputfile = os.path.join('video', str(i + 1).zfill(2) + '_' + video_basename + '.mp4')
            clip = VideoFileClip(inputfile, audio=False)

            if not base_size:
                base_size = clip.size

            self.clips.append(clip)

        return base_size

    def _update_cur_scene(self, t):
        """
        Update scene settings depending on current output clip time `t`.
        :param t: output clip time
        """
        for i, sc_def in enumerate(self.scenes):
            scene_t = sc_def['t']
            if scene_t[0] <= t < scene_t[1] and self.cur_clip is not self.clips[i]:
                self.cur_scene = sc_def
                self.cur_scene_idx = i
                self.cur_clip = self.clips[i]
                self.clip_t = 0
                break


# load the onsets
with open(INPUT_ONSETS, 'rb') as f:
    samplerate, onsets, onset_max_ampl, _ = pickle.load(f)
    assert len(onsets) == len(onset_max_ampl)

# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

# load the audio clip
audioclip = AudioFileClip(INPUT_AUDIO)

if override_duration:
    duration = override_duration
else:
    duration = audioclip.duration

print('will generate %d sec. of video' % duration)
if override_only_scene:
    print('will only render scenes %s' % (','.join(map(str, [s+1 for s in override_only_scene]))))
print('using audio %s' % INPUT_AUDIO)
print('using onsets %s' % INPUT_ONSETS)

# setup the frame generator
frame_gen = VideoFrameGenerator(SCENES, onset_frame_ampl)
gen_clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=duration)

# setup intro text
introtext = "kiriloff – fortschritt"
introtext_clip = TextClip(introtext,
                          color='white',
                          font='Menlo-Bold',
                          fontsize=20 if gen_clip.size[0] <= 640 else 45,
                          method='caption',
                          size=(frame_gen.w, frame_gen.h))

# create full clip as composite of generate frames and intro text
main_clip = CompositeVideoClip([
    gen_clip,
    introtext_clip.set_start(0.5).set_end(6.5).crossfadein(0.5).crossfadeout(0.5)
])

# generate frames
main_clip = main_clip.set_audio(audioclip).set_duration(gen_clip.duration)
main_clip.write_videofile(OUTPUT_VIDEO, fps=CLIP_FPS)

print('done.')
