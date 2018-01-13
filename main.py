"""
Image processing
"""
from __future__ import division
import numpy as np
import cv2
import os

# OpenCV halves the H values to fit the range [0,255],
# so H value instead of being in range [0, 360], is in range [0, 180].
# S and V are still in range [0, 255].

SHOW_STEPS = False
SHOW_FINAL_RESULT = True


def compare_imgs():
    return
    # shape = closing
    # actualImage = getImage()
    #
    # _, contours, _= cv2.findContours(shape, 2, 1)
    # cnt1 = contours[0]
    # _, contours, _ = cv2.findContours(actualImage, 2, 1)
    # cnt2 = contours[0]
    # ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
    # print ret

    # cv2.imshow("img", actualImage)
    # cv2.waitKey(0)

    # return;

    # hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # lower_green = np.array([30, 50, 50])
    # upper_green = np.array([80, 255, 200])


def get_images_paths(dir):
    imgs = []
    imgsDir = dir
    valid_images = [".jpg", ".png"]

    cwd = os.getcwd()
    os.chdir(imgsDir)
    for f in os.listdir("."):
        filename, ext = os.path.splitext(f)
        if (ext.lower() not in valid_images):
            continue
        imgs.append(os.path.abspath(f))
    os.chdir(cwd)
    imgs.sort()
    return imgs


def resize_image(img):
    maxDim = max(img.shape)
    scaleFactor = 500 / maxDim
    return cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor)


def soften_image(img):
    return cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3)


def convert_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mask_out_green_pixels(img):
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    return cv2.inRange(img, lower_green, upper_green)


def remove_outside_noise(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def enclose_empty_holes(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def find_biggest_contour(img):
    _, cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    return max(cnts, key=cv2.contourArea)


def filter_out_small_cnts(cnts, min_area):
    filtered_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]
    if len(filtered_cnts) == 0:
        return filter_out_small_cnts(cnts, min_area / 2)
    else:
        return filtered_cnts


def outline_leaf(originalImg, cnt):
    print cnt
    epsilon = 0.003 * cv2.arcLength(cnt, True)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return cv2.drawContours(originalImg, [approx], -1, (0, 255, 0), 2)


def load_shapes():
    imgs = get_images_paths("./images/shapes")
    cnts = []
    for imgPath in imgs:
        cnts.append(cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE))
    return cnts


# def find_best_shape_match(cnt):
#     shape_cnts = load_shapes()
#
#     cur_win_shape = None
#     cur_win_similarity = float("inf")
#
#     for shape_cnt in shape_cnts:
#         sim = cv2.matchShapes(cnt, shape_cnt, 1, 0.0)
#         if sim < cur_win_similarity:
#             cur_win_similarity = sim
#             cur_win_shape = shape_cnt
#
#     return cur_win_shape


def find_best_shape_match(img):
    CNT_DEPTH = 10
    MIN_AREA = 10000

    _, cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    cnts = filter_out_small_cnts(cnts, MIN_AREA)

    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    max_sim = float("inf")
    best_shape = None
    best_cnt = None

    shapes = load_shapes()
    for i in range(len(shapes)):
        cnt_depth = len(sorted_cnts) if len(sorted_cnts) < CNT_DEPTH else CNT_DEPTH
        shape_cnt = find_biggest_contour(shapes[i])
        for j in range(cnt_depth):
            sim = cv2.matchShapes(shape_cnt, sorted_cnts[j], 1, 0.0)
            if sim < max_sim:
                max_sim = sim
                best_cnt = sorted_cnts[j]
                best_shape = shapes[i]

    return (best_cnt, best_shape)


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def save_img(img, path):
    return


def detect_leaf(imgPath):
    img = cv2.imread(imgPath)
    print imgPath

    img = resize_image(img)
    origin_img = img.copy()

    img = soften_image(img)

    img = convert_to_hsv(img)
    img = mask_out_green_pixels(img)

    best_cnt, shape = find_best_shape_match(img)

    if SHOW_STEPS: show_image(img)

    img = enclose_empty_holes(img)

    if SHOW_STEPS: show_image(img)

    img = remove_outside_noise(img)

    if SHOW_STEPS: show_image(img)

    origin_img = outline_leaf(origin_img, best_cnt)

    if SHOW_FINAL_RESULT: show_image(origin_img)


def main():
    """Entry pointt"""

    img_paths = get_images_paths("./images/leaves")
    for img_path in img_paths:
        detect_leaf(img_path)


if __name__ == "__main__":
    main()
