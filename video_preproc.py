"""
Video preprocessing script.
Takes raw videos from ./video/raw and preprocesses them according to the SCENES definition from conf.py.

Run as: video_preproc.py [lowres]

2018, Markus Konrad <post@mkonrad.net>
"""

import os

from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

from conf import SCENES, CLIP_H, CLIP_W, CLIP_FPS


print('video preprocessing to videos of size %dx%d, %d fps' % (CLIP_W, CLIP_H, CLIP_FPS))

base_size = None

# go through each scene
for i, scene in enumerate(SCENES):
    # get the input and output video clip paths
    inputfile = os.path.join('video/raw', scene['video'])
    video_basename, _ = os.path.splitext(scene['video'])
    outputfile = os.path.join('video', str(i+1).zfill(2) + '_' + video_basename + '.mp4')

    # load the clip
    clip = VideoFileClip(inputfile, audio=False, fps_source='fps')
    orig_clip_size = clip.size

    # get subclip if necessary
    subclip_markers = scene.get('subclip')
    if subclip_markers:
        clip = clip.subclip(*subclip_markers)
        apply_subclip = True
    else:
        apply_subclip = False

    # resize if necessary
    if clip.size[0] != CLIP_W or clip.size[1] != CLIP_H or clip.fps != CLIP_FPS:
        clip = clip.fx(vfx.resize, width=CLIP_W)
        apply_resize = True
    else:
        apply_resize = False

    # crop if necessary
    apply_crop = False
    if not base_size:
        base_size = clip.size
    elif base_size and (base_size[0] < clip.size[0] or base_size[1] < clip.size[1]):
        clip = clip.fx(vfx.crop,
                       x_center=clip.size[0] // 2, y_center=clip.size[1] // 2,
                       width=base_size[0], height=base_size[1])
        apply_crop = True

    # apply all transformations and write output clip
    print('[%d/%d] converting %s -> %s (%dx%d w/ %d fps -> %dx%d w/ %d fps | subclip: %d | resize: %d | crop: %d)'
          % ((i+1), len(SCENES), inputfile, outputfile,
             orig_clip_size[0], orig_clip_size[1], clip.fps,
             base_size[0], base_size[1], CLIP_FPS,
             apply_subclip, apply_resize, apply_crop))

    clip.write_videofile(outputfile, fps=CLIP_FPS)

print('done.')
