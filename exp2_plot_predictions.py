"""
Representation of 3D Shape

This file contains the script for plotting the predictions of our model for each stimuli individually.

Created on Jun 17, 2016

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
    plot_folder = './plot/individual_stimuli/our_model'

    predictions = pd.read_csv('predictions.csv', index_col=0)
    predictions['Prediction'] = np.exp(predictions['LogProbability_Image'] - predictions['LogProbability_best'])

    for stimulus in stimuli:
        stim_preds = predictions.loc[predictions['Stimulus Id'] == stimulus, :]
        print(stimulus)

        plot_figure(stim_preds)
        plt.savefig('{0:s}/{1:s}.png'.format(plot_folder, stimulus))

