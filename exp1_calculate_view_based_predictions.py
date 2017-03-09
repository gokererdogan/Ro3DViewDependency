"""
Representation of 3D Shape

This file contains the script for calculating the predictions from view-based model for experiment 1 (canonical view
experiment).

Created on May 26, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np

if __name__ == "__main__":
    stimulus1_name = 's1'
    train_view1 = 80
    stimulus2_name = 's5'
    train_view2 = 20

    train_img1 = np.load("stimuli/exp1/{0:s}_{1:d}.npy".format(stimulus1_name, train_view1))
    train_img2 = np.load("stimuli/exp1/{0:s}_{1:d}.npy".format(stimulus2_name, train_view2))

    viewpoints = range(0, 360, 20)
    distance_test1_train1 = np.zeros(len(viewpoints))
    distance_test2_train1 = np.zeros(len(viewpoints))
    distance_test1_train2 = np.zeros(len(viewpoints))
    distance_test2_train2 = np.zeros(len(viewpoints))
    for vp_i, vp in enumerate(viewpoints):
        print(vp)

        test_img1 = np.load("stimuli/exp1/{0:s}_{1:d}.npy".format(stimulus1_name, vp))
        test_img2 = np.load("stimuli/exp1/{0:s}_{1:d}.npy".format(stimulus2_name, vp))
        distance_test1_train1[vp_i] = np.sum(np.square(test_img1 - train_img1))
        distance_test2_train1[vp_i] = np.sum(np.square(test_img2 - train_img1))
        distance_test1_train2[vp_i] = np.sum(np.square(test_img1 - train_img2))
        distance_test2_train2[vp_i] = np.sum(np.square(test_img2 - train_img2))

    import matplotlib.pyplot as plt

    plt.style.use("classic")
    plt.figure()
    plt.plot(viewpoints, distance_test1_train2 - distance_test1_train1)
    plt.figure()
    plt.plot(viewpoints, distance_test2_train1 - distance_test2_train2)
    plt.show()
