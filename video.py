import cv2
import numpy as np

sec = 10
fps = 24
n_frames = fps*sec
w = 1000
h = 1000
r_min = 1
r_max = h/2
impulse = 100
decay = 1        # linear decay of r per frame
#writer = cv2.VideoWriter('opencv_video.mkv', cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
writer = cv2.VideoWriter('out/opencv_video_test.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center = (int(round(w/2)), int(round(h/2)))
onset_sec  = np.array([1.0, 2.0, 5.0, 5.5])
onset_frames = onset_sec * fps
onset_frames = onset_frames.round().astype(np.int)
onset_ampl = np.array([0.3, 0.2, 1.0, 0.9])
onset_frame_ampl = dict(zip(onset_frames, onset_ampl))

r = 0
for i in range(n_frames):
    print('frame %d' % (i+1))

    frame = np.full((w, h, 3), (0, 0, 0), np.uint8)

    radd = onset_frame_ampl.get(i, 0) * impulse
    r = r + radd - decay
    r = max(min(r, r_max), r_min)

    cv2.circle(frame, center, int(round(r)), (255, 0, 0), -1)

    writer.write(frame)

