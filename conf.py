"""
Common configuration

2018, Markus Konrad <post@mkonrad.net>
"""

import sys

CLIP_FPS = 25

if len(sys.argv) > 1 and sys.argv[1] == 'lowres':
    CLIP_W = 640
    CLIP_H = 360
else:
    CLIP_W = 1920
    CLIP_H = 1080


# scene definitions used in video_preproc.py and makevideo.py
# chronological order of scenes
# each with dict of settings:
# - video: video file to use
# - mode: "original" or "voronoi" (or None to show only black frames)
# - base: underlying frame ("original", "bin" or color tuple)
# - t: time in sec. interval the scene is shown
# - jump: jump at onset amplitude to frames within scene
# - voronoi: voronoi cells rendering configuration
SCENES = [
    {
        'video': '00156.MTS',
        'mode': 'voronoi',
        #'mode': 'original',
        #'mode': None,
        't': (0, 25.089),
        'jump': {
            'ampl': 0.06,
            'by_random': 20,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 20000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.01,
            'features_where': 0
        },
        'fade': {
            'start_t': 23.0,
            'end_t': 25.089,
        },
    },
    {
        'video': '00151.MTS',
        'mode': 'voronoi',
        #'mode': None,
        #'base': 'original',
        't': (25.089, 31),
        'subclip': (5, 10),
        'jump': {
            'ampl': 0.05,
            'to': 0,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 17500,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.01,
            'features_where': 0
        },
    },
    {
        'video': '00151.MTS',
        'mode': 'voronoi',
        #'mode': None,
        #'base': 'original',
        't': (31, 58.5),
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 17500,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.01,
            'features_where': 0
        },
    },
    {
        'video': '00155.MTS',
        'mode': 'voronoi',
        #'mode': 'original',
        #'mode': None,
        't': (58.5, 82.0),
        'subclip': (12, None),
        'jump': {
            'ampl': 0.2,
            'by_random': 8,
        },
        'fade': {
            'start_t': 80.0,
            'end_t': 82.0,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 10000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.01,
            'features_where': 0
        },
    },
    {
        'video': 'live.3gp',
        'mode': 'voronoi',
        #'mode': None,
        't': (82.0, 120.01),
        'voronoi': {
            'lines_features_factor_fade': {
                'from_to': (2000, 15000),
                'start_t': 82.0,
                'end_t': 101.0
            },
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.1,
            'features_where': 0
        },
    },
    {
        'video': '00121.MTS',
        'subclip': (203, None),
        'mode': 'voronoi',
        #'mode': None,
        't': (120.01, 143.449),
        'voronoi': {
            'lines_features_factor': 15000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.05,
            'features_where': 0
        },
    },
    {
        'video': '00122.MTS',
        'subclip': (8*60+5, None),
        'mode': 'voronoi',
        #'mode': None,
        't': (143.449, 2*60+38.523),
        'jump': {
            'ampl': 0.1,
            'by_random': 8,
        },
        'fade': {
            'start_t': 2*60+36,
            'end_t': 2*60+38.523,
            'color': 'white',
        },
        'voronoi': {
            'lines_features_factor': 15000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.05,
            'features_where': 0
        },
    },
    {
        'video': '00160.MTS',
        'mode': 'voronoi',
        'base': (1, 1, 1),
        't': (2*60+38.523, 2*60+58.5),
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 7500,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.025,
            'features_where': 0
        },
    },
    {
        'video': '00151.MTS',
        'mode': 'voronoi',
#        'mode': None,
        'base': (1, 1, 1),
        't': (2*60+58.5, 3*60+43.48),
        'subclip': (5, 10),
        'jump': {
            'ampl': 0.05,
            'to': 0,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 10000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.025,
            'features_where': 0
        },
    },
    {
        'video': 'live.3gp',
        'mode': 'voronoi',
        'base': (1, 1, 1),
        't': (3*60+43.48, 4*60+5.365),
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 10000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.025,
            'features_where': 255
        },
        'fade': {
            'start_t': 4*60+4.365,
            'end_t': 4*60+5.365,
            'color': 'black',
        },
    },
    {
        'video': '00155.MTS',
        'subclip': (17, None),
        'mode': 'voronoi',
        'base': 'bin',
        't': (4*60+5.365, 5),
        'jump': {
            'ampl': 0.5,
            'by_random': 8,
        },
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 5000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.025,
            'features_where': 0
        },
    },
]
