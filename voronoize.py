import random

import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d

from helpers import display_img, draw_lines


def lines_for_voronoi(vor, img_w, img_h):
    # taken and adopted from https://github.com/scipy/scipy/blob/v1.0.0/scipy/spatial/_plotutils.py
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


def features_from_img(img, blur_radius, features_where=0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if blur_radius > 0:
        gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray, binimg, np.where(binimg == features_where)


def voronoi_from_feature_samples(features, sample_size):
    pts = random.sample(list(zip(features[1], features[0])), sample_size)
    return Voronoi(pts)

#
# img = cv2.imread('city-silhouette2-960px.jpg')
# img_w, img_h = img.shape[:2]
#
# binimg, features = features_from_img(img, blur_radius=5)
# outimg = cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)
#
# vor = voronoi_from_feature_samples(features, 1000)
# lines = lines_for_voronoi(vor, img_w, img_h)
# draw_lines(outimg, lines, (255, 0, 0))
#
# #vimg = np.full((img.shape[0], img.shape[1], 3), (255, 255, 255), np.uint8)
# #display_img(vimg)
#
# # for p in vor.vertices:
# #     cv2.circle(vimg, tuple(p.round().astype(np.int32)), 5, (255, 0, 0), -1)
#
# #cv2.polylines(vimg, [lines_arr], False, (0, 0, 0))
# display_img(outimg)
#
#
# ######
#
# #voronoi_plot_2d(vor, show_points=False, show_vertices=True); plt.show()   # upside down
#
