import cv2
from moviepy.editor import VideoFileClip


input_clip = VideoFileClip('video/stockvideotest.mp4', audio=False).subclip(0, 3)


def binarize(frame, blur_radius=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if blur_radius > 0:
        gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)


bin_clip = input_clip.fl_image(binarize)
# get_frame!

bin_clip.write_videofile('out/moviepy_bin_test.mp4')


# for frame in input_clip.iter_frames():
#     frame = binarize(frame, 5)

