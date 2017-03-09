"""
Representation of 3D Shape

This file contains the script for calculating the predictions of pixel-based model for experiment 2 on view-dependency.

Created on Jun 10, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.plot_viewpoint_results import plot_figure


if __name__ == '__main__':
    # stimuli = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    stimuli = ['s10']
    stimuli_count = len(stimuli)
    stimuli_folder = './stimuli/exp2'
    plot_folder = './plot/individual_stimuli/pixel_based'

    predictions = pd.DataFrame(index=np.arange(21), columns=['Condition', 'ViewpointDifference', 'Prediction'],
                               dtype=float)

    conditions = ['inter', 'extra', 'ortho']
    viewpoints = range(0, 91, 15)

    for stimulus in stimuli:
        print(stimulus)
        # read training images
        training_imgs = np.load('{0:s}/{1:s}_train.npy'.format(stimuli_folder, stimulus))

        row_id = 0
        for condition in conditions:
            filename = "{0:s}/{1:s}_{2:s}.npy".format(stimuli_folder, stimulus, condition)
            condition_imgs = np.load(filename)

            for i, vp in enumerate(viewpoints):
                dist = np.min(np.sum((training_imgs - condition_imgs[i])**2, axis=(1, 2)))
                predictions.iloc[row_id] = [condition, vp, dist]
                row_id += 1

        plot_figure(predictions)
        plt.savefig('{0:s}/{1:s}.png'.format(plot_folder, stimulus))

