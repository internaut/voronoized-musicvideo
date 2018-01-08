import cv2
import numpy as np

sec = 5
fps = 24
n_frames = fps*sec
w = 1000
h = 1000
#writer = cv2.VideoWriter('opencv_video.mkv', cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
writer = cv2.VideoWriter('out/opencv_video_test.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center = (int(round(w/2)), int(round(h/2)))
for i in range(n_frames):
    print('frame %d' % (i+1))

    frame = np.full((w, h, 3), (0, 0, 0), np.uint8)
    r = round(i * min(w, h)/n_frames)
    cv2.circle(frame, center, r, (255, 0, 0), -1)

    writer.write(frame)

