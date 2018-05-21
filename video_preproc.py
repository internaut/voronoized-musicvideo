import os

from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

from conf import SCENES, CLIP_H, CLIP_W, CLIP_FPS


base_size = None

for i, scene in enumerate(SCENES):
    inputfile = os.path.join('video/raw', scene['video'])
    video_basename, _ = os.path.splitext(scene['video'])
    outputfile = os.path.join('video', str(i+1).zfill(2) + '_' + video_basename + '.mp4')

    print('[%d/%d] converting %s -> %s (%dx%d, %d fps)'
          % ((i+1), len(SCENES), inputfile, outputfile, CLIP_W, CLIP_H, CLIP_FPS))

    clip = VideoFileClip(inputfile, audio=False, fps_source='fps')
    subclip_markers = scene.get('subclip')
    if subclip_markers:
        clip = clip.subclip(*subclip_markers)
    clip = clip.fx(vfx.resize, width=CLIP_W)
    if not base_size:
        base_size = clip.size
    elif base_size and (base_size[0] < clip.size[0] or base_size[1] < clip.size[1]):
        clip = clip.fx(vfx.crop,
                       x_center=clip.size[0] // 2, y_center=clip.size[1] // 2,
                       width=base_size[0], height=base_size[1])

    clip.write_videofile(outputfile, fps=CLIP_FPS)

