import pickle
import os

import cv2
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, AudioClip
import moviepy.video.fx.all as vfx

from voronoize import features_from_img, voronoi_from_feature_samples, lines_for_voronoi
from helpers import create_surface, draw_lines

CLIP_FPS = 24
CLIP_W = 640
#CLIP_H = 480


class VideoFrameGenerator:
    def __init__(self, scenes, onset_frame_ampl, fps=CLIP_FPS):
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
        self.fps = fps

        #self.frame_layers_alpha_decay_basefactor = 0.05
        #self.frame_layers = []

        self.vor_lines = []    # holds tuples (voronoi lines frame, lines mask, current alpha, alpha decay factor)

    def make_video_frame(self, t):
        fnum = int(round(t * self.fps))   # frame number

#         onset_ampl = self.onset_frame_ampl.get(fnum, 0)
#         self._update_cur_scene(t)
#
#         self.clip_t += 1/self.fps
# #        self.clip_t -= onset_ampl * 0.5
# #        self.clip_t = max(self.clip_t, 0)
#
#         if self.clip_t > self.cur_clip.end:
#             self.clip_t = 0
#
#         in_frame = self.cur_clip.get_frame(self.clip_t)
#
#         if in_frame.dtype != np.uint8:
#             in_frame = in_frame.astype(np.uint8)
#
#         _, bin_frame, features = features_from_img(in_frame,
#                                                    blur_radius=5,
#                                                    features_where=self.cur_scene.get('features_where', 0))
        # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        # gray_frame = gray_frame.astype(np.float) / 255
        # out_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)   # use mask as output base
        # out_frame = out_frame.astype(np.float) / 255

        # if self.cur_scene['base'] == 'bin':
        #    out_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR).astype(np.float) / 255
        #    if self.cur_scene.get('base_invert'):
        #        out_frame = 1 - out_frame
        # else:
        #    out_frame = create_frame(in_frame.shape[1], in_frame.shape[0], dtype=np.float)
        out_surface = create_surface(in_frame.shape[1], in_frame.shape[0])

        # fade_in = self.cur_scene.get('fade_in')
        # if fade_in:
        #     base_frame_alpha = min(self.clip_t / fade_in, 1)
        # else:
        #     base_frame_alpha = 1.0
        #
        # # gray_frame = out_frame.copy()
        #
        # # out_frame = in_frame.astype(np.float) / 255        # use orig. frame as output base
        # # out_frame_mask = (255-bin_frame)[:, :, np.newaxis]    # use masked orig. frame as output base
        # # out_frame = out_frame_mask * (in_frame.astype(np.float) / 255)
        #
        # if onset_ampl > 0:
        #     n_vor_features = int(round(self.cur_scene['vor_lines_features_factor'] * onset_ampl))
        #     vor = voronoi_from_feature_samples(features, n_vor_features)
        #     lines = lines_for_voronoi(vor, self.w, self.h)
        #     alpha_decay = self.cur_scene['vor_lines_alpha_decay_basefactor'] * (1.5-onset_ampl)
        #     initial_alpha = max(min(self.cur_scene['vor_lines_initial_alpha_factor'] * onset_ampl, 1.0), 0.2)
        #
        #     self.vor_lines.append((lines, initial_alpha, alpha_decay))
        #     #self.frame_layers.append((gray_frame, 1.0, alpha_decay))
        #
        # self._update_voronoi_lines(out_surface, base_frame_alpha)
        # print(fnum, len(self.vor_lines), sum(map(len, [lines for lines, _, _ in self.vor_lines])))

        return out_surface.get_npimage()

    def _update_voronoi_lines(self, surface, surface_alpha):
        tmp_vor_lines = []
        for lines, lines_alpha, lines_alpha_decay in self.vor_lines:
            draw_lines(surface, lines, color=(1, 1, 1, lines_alpha))

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
    # {
    #    'video': 'VID_20180110_233008.mp4',
    #    'base': None,
    #    't': (0, 10),
    #    #'subclip': (3, 13),
    #    # 't': (0, 5),
    #    'features_where': 255,
    #    'vor_lines_features_factor': 40000,
    #    'vor_lines_initial_alpha_factor': 10.0,
    #    'vor_lines_alpha_decay_basefactor': 0.01,
    #},
    {
         'video': 'augen.mp4',
         'base': None,
         't': (0, 58.212),
         #'t': (0, 5),
         'vor_lines_features_factor': 20000,
         'vor_lines_initial_alpha_factor': 8.0,
         'vor_lines_alpha_decay_basefactor': 0.01,
    },
    # {
    #     'video': 'flug.mp4',
    #     'base': 'bin',
    #     'base_invert': True,
    #     #'fade_in': 0.5,
    #     't': (58.212, 81.769),
    #     #'t': (5, 10),
    #     'vor_lines_features_factor': 15000,
    #     'vor_lines_initial_alpha_factor': 6.0,
    #     'vor_lines_alpha_decay_basefactor': 0.01,
    # },
    # {
    #     'video': 'live.3gp',
    #     'base': None,
    #     #'fade_in': 0.5,
    #     't': (81.769, 120),
    #     #'t': (5, 10),
    #     'features_where': 255,
    #     'vor_lines_features_factor': 20000,
    #     'vor_lines_initial_alpha_factor': 2.0,
    #     'vor_lines_alpha_decay_basefactor': 0.01,
    # }
]

#input_clip = VideoFileClip('video/stockvideotest.mp4', audio=False).subclip(0, 0 + CLIP_SEC)
#input_clip = input_clip.fx(vfx.resize, width=CLIP_W)


# convert onset audio sample markers to frame numbers
onset_frames = np.round(onsets / samplerate * CLIP_FPS).astype(np.int)
onset_frame_ampl = dict(zip(onset_frames, onset_max_ampl))

frame_gen = VideoFrameGenerator(scenes, onset_frame_ampl)

#audioclip = AudioFileClip('audio/kiriloff-fortschritt-unmastered.wav')

clip = VideoClip(lambda t: frame_gen.make_video_frame(t), duration=10)
#audioclip = audioclip.set_duration(clip.duration)
#clip = clip.set_audio(audioclip)
clip.write_videofile('out/moviepy_video_test.mp4', fps=CLIP_FPS)
