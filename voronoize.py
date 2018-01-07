import random

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def display_img(img):
    if len(img.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None
    plt.imshow(img, interpolation='bicubic', cmap=cmap)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def lines_for_voronoi(vor, img_w, img_h):
    center = np.array([img_w, img_h]) / 2
    max_dim_extend = max(img_w, img_h)

    lines = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            lines.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * max_dim_extend

            lines.append([vor.vertices[i], far_point])
    return lines


img = cv2.imread('city-silhouette2-960px.jpg')
img_w, img_h = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
display_img(gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
display_img(gray)
_, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
display_img(binimg)

#edges = cv2.Canny(gray, 100, 200)
#morph_open_kernel = np.ones((2, 2), np.uint8)
#edges = cv2.erode(edges, morph_open_kernel, iterations=1)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_open_kernel)

#display_img(edges)

#cv2.imwrite('city-silhouette-edges.jpg', edges)


features = np.where(binimg == 0)

outimg = cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)

pts = random.sample(list(zip(features[1], features[0])), 5000)
vor = Voronoi(pts)

#vimg = np.full((img.shape[0], img.shape[1], 3), (255, 255, 255), np.uint8)
#display_img(vimg)

#######

lines = lines_for_voronoi(vor, img_w, img_h)

for a, b in lines:
    a = tuple(a.round().astype(np.int32))
    b = tuple(b.round().astype(np.int32))
    cv2.line(outimg, a, b, (255, 0, 0))

# see https://github.com/scipy/scipy/blob/v1.0.0/scipy/spatial/_plotutils.py

# for p in vor.vertices:
#     cv2.circle(vimg, tuple(p.round().astype(np.int32)), 5, (255, 0, 0), -1)

#cv2.polylines(vimg, [lines_arr], False, (0, 0, 0))
display_img(outimg)


######

#voronoi_plot_2d(vor, show_points=False, show_vertices=True); plt.show()   # upside down

