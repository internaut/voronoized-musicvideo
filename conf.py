CLIP_FPS = 24
CLIP_W = 640
CLIP_H = 480

SCENES = [
    {
        'video': '00156.MTS',
        'mode': 'original',
        #'mode': None,
        't': (0, 24.5),
        'jump': {
            'ampl': 0.02,
            'by_random': 20,
        }
    },
    {
        'video': '00151.MTS',
        'mode': 'voronoi',
        #'mode': None,
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
    {
        'video': '00155.MTS',
        'mode': 'original',
        #'mode': None,
        't': (58.212, 81.0),
        'subclip': (12, None),
        'jump': {
            'ampl': 0.1,
            'by_random': 8,
        },
        'fade': {
            'start_t': 79.0,
            'end_t': 81.0,
        }
    },
    {
        'video': 'live.3gp',
        'mode': 'voronoi',
        #'mode': None,
        't': (81.0, 120.5),
        'voronoi': {
            'lines_features_factor_fade': {
                'from_to': (2000, 15000),
                'start_t': 81.0,
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
        't': (120.5, 143),
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
        't': (143, 2*60+38),
        'jump': {
            'ampl': 0.1,
            'by_random': 8,
        },
        'fade': {
            'start_t': 2*60+36,
            'end_t': 2*60+38.5,
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
        't': (2*60+38.5, 2*60+58.5),
        'voronoi': {
            'color': (0, 0, 0),
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
        't': (2*60+58.5, 3*60+42.8),
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
        't': (3*60+42.8, 4*60+5),
        'voronoi': {
#            'color': (0, 0, 0),
            'lines_features_factor': 10000,
            'lines_initial_alpha_factor': 8.0,
            'lines_alpha_decay_basefactor': 0.025,
            'features_where': 255
        },
        'fade': {
            'start_t': 4*60+4,
            'end_t': 4*60+5,
            'color': 'black',
        },
    },
    {
        'video': '00155.MTS',
        'subclip': (17, None),
        'mode': 'voronoi',
        'base': 'bin',
        't': (4*60+5, 5),
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