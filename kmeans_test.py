# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

def compress_image(img, num_clusters):
    X = img.reshape((-1,1))
    kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=1)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    image_compress = np.choose(labels, centroids).reshape(img.shape)
    return image_compress

def plot_image(img):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    # plt.imshow(img, plt.cm.gray, vmin=vmin, vmax=vmax)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

def scipy_read_image(path):
    input_image = misc.imread(path)
    plot_image(input_image)
    return input_image

def PIL_read_image(path):
    input_image = Image.open(path)
    # 转化为灰度图
    input_image = input_image.convert('L')
    input_image = np.array(input_image)
    plot_image(input_image)
    return input_image

if __name__ == '__main__':
    image_path = 'E:/flower_image.jpg'
    # scipy_read_image(image_path)
    img = PIL_read_image(image_path)
    cp_img = compress_image(img, 2)
    plot_image(cp_img)


