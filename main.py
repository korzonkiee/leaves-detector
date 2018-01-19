"""
Image processing
"""
from __future__ import division
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import os

# OpenCV halves the H values to fit the range [0,255],
# so H value instead of being in range [0, 360], is in range [0, 180].
# S and V are still in range [0, 255].

SHOW_STEPS = False
SHOW_FINAL_RESULT = False
SAVE_IMG = True


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


def save_categorized_images(labels):
    paths = get_images_paths("./images/processed")
    for i, path in enumerate(paths):
        filename, ext = os.path.splitext(path)
        os.rename(path, filename + "_type" + str(labels[i]) + ext)


def clean_processed_dir():
    filelist = [f for f in os.listdir("./images/processed")]
    for f in filelist:
        os.remove(os.path.join("./images/processed", f))


def resize_image(img):
    maxDim = max(img.shape)
    scaleFactor = 500 / maxDim
    return cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor)


def soften_image(img):
    return cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3)


def convert_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mask_out_green_pixels(img):
    lower_green = np.array([35, 25, 40])
    upper_green = np.array([75, 255, 255])

    return cv2.inRange(img, lower_green, upper_green)


def remove_outside_noise(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def enclose_empty_holes(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def find_biggest_contour(img):
    _, cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea)


def filter_out_small_cnts(cnts, min_area):
    filtered_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]
    if len(filtered_cnts) == 0:
        return filter_out_small_cnts(cnts, min_area / 2)
    else:
        return filtered_cnts


def find_concavities(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(approx, returnPoints=False)
    concavities = cv2.convexityDefects(approx, hull)
    if concavities is None:
        concavities = []

    return len(concavities)


def outline_leaf(original_img, cnt):
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 2)


def load_shapes():
    imgs = get_images_paths("./images/shapes")
    cnts = []
    for imgPath in imgs:
        cnts.append(cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE))
    return cnts


def compare_imgs(img1, img2):
    return cv2.matchShapes(img1, img2, 1, 0.0)


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
            sim = compare_imgs(shape_cnt, sorted_cnts[j])
            if sim < max_sim:
                max_sim = sim
                best_cnt = sorted_cnts[j]
                best_shape = shapes[i]

    return (best_cnt, best_shape, max_sim)


def leave_leaf(img, best_cnt):
    h, w = img.shape[:2]
    img = np.zeros((h, w, 3), np.uint8)
    epsilon = 0.003 * cv2.arcLength(best_cnt, True)
    perimeter = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, epsilon, True)
    cv2.drawContours(img, [approx], -1, (255, 255, 255), cv2.FILLED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def is_better_leaf(img, shape, current_sim):
    new_sim = compare_imgs(img, shape)
    if new_sim < current_sim:
        current_sim = new_sim
        return True, new_sim

    return False, -1


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def adjust_morph(img, shape, sim):
    empty_holes_removed = enclose_empty_holes(img)
    res, new_sim = is_better_leaf(empty_holes_removed, shape, sim)
    if res:
        img = empty_holes_removed
        sim = new_sim

    out_noise_removed = remove_outside_noise(img)
    res, new_sim = is_better_leaf(out_noise_removed, shape, sim)
    if res:
        img = out_noise_removed
        sim = new_sim

    return img


def save_img(img, path):
    cv2.imwrite(path, img)


def detect_leaf(imgPath, arr_concavities):
    img = cv2.imread(imgPath)
    print imgPath

    img = resize_image(img)
    origin_img = img.copy()

    img = soften_image(img)
    img = convert_to_hsv(img)
    img = mask_out_green_pixels(img)

    best_cnt, shape, sim = find_best_shape_match(img)
    if SHOW_STEPS: show_image(img)
    img = leave_leaf(img, best_cnt)
    if SHOW_STEPS: show_image(img)
    img = adjust_morph(img, shape, sim)
    if SHOW_STEPS: show_image(img)

    best_cnt = find_biggest_contour(img)
    n_concavities = find_concavities(best_cnt)

    # origin_img = outline_leaf(origin_img, best_cnt)

    # x_img = soften_image(origin_img)
    #
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_out = cv2.subtract(origin_img, img)
    mask_out = cv2.subtract(origin_img, mask_out)

    arr_concavities += [n_concavities]

    # show_image(avg_img)

    if SHOW_FINAL_RESULT: show_image(origin_img)

    if SAVE_IMG: save_img(mask_out, "./images/processed/r-" + os.path.split(imgPath)[1])

    return arr_concavities


def main():
    clean_processed_dir()
    img_paths = get_images_paths("./images/leaves")
    arr_concavities = []
    for img_path in img_paths:
        arr_concavities = detect_leaf(img_path, arr_concavities)

    arr_concavities = np.float32(arr_concavities)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    ret, labels, center = cv2.kmeans(arr_concavities, 2, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

    save_categorized_images(labels)


if __name__ == "__main__":
    main()
