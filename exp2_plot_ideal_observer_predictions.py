"""
Representation of 3D Shape

This file contains the script for calculating the predictions of a 3D ideal observer for experiment 2 on
view-dependency. Since for a 3D ideal observer would extract the true 3D shape regardless of viewpoint,
p(I_test|I_train) is the same for all viewpoints. Therefore, the prediction is driven by p(I_test|I!=I_train)
which can be approximated by the pixel mass in test image.

Created on Jun 10, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.plot_viewpoint_results import plot_figure


if __name__ == '__main__':
    stimuli = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    # stimuli = ['s10']
    stimuli_count = len(stimuli)
    stimuli_folder = './stimuli/exp2'
    plot_folder = './plot/individual_stimuli/ideal_3D'

    predictions = pd.DataFrame(index=np.arange(21), columns=['Condition', 'ViewpointDifference', 'Prediction'],
                               dtype=float)

    conditions = ['inter', 'extra', 'ortho']
    viewpoints = range(0, 91, 15)

    for stimulus in stimuli:
        row_id = 0
        print(stimulus)
        for condition in conditions:
            filename = "{0:s}/{1:s}_{2:s}.npy".format(stimuli_folder, stimulus, condition)
            condition_imgs = np.load(filename)
            condition_imgs /= 255.0

            for i, vp in enumerate(viewpoints):
                mass = np.sum(condition_imgs[i]**2) / condition_imgs.size
                predictions.iloc[row_id] = [condition, vp, np.exp(-mass)]
                row_id += 1

        plot_figure(predictions)
        plt.savefig('{0:s}/{1:s}.png'.format(plot_folder, stimulus))

