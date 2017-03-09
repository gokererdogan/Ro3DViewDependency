"""
Representation of 3D Shape

This file contains the script for calculating the predictions of our model for experiment 2 on view-dependency.

Created on Mar 17, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import itertools
import cPickle as pkl
import numpy as np
import pandas as pd

import joblib

from Infer3DShape.similarity.calculate_similarity import calculate_probability_image_given_hypothesis
import Infer3DShape.vision_forward_model as vfm
import Infer3DShape.paperclip_shape as paperclip_shape


def predict_stimulus_condition(stim_folder, stim, condition, random_shapes):
    filename = "{0:s}/{1:s}_{2:s}.npy".format(stim_folder, stim, condition)
    data = np.load(filename)

    fwm = vfm.VisionForwardModel(render_size=(200, 200), custom_lighting=False)
    for shape in random_shapes:
        shape.forward_model = fwm

    log_probs = []
    for i in range(7):
        print(stim, condition, i)
        # calculate logp(image_test|I != image_train) by random shapes
        logp_test_no_train = 0.0
        logp_test_no_train_best = 0.0
        for shape in random_shapes:
            logp_avg, logp_max = calculate_probability_image_given_hypothesis(data[i], shape)
            logp_test_no_train += logp_avg
            logp_test_no_train_best += logp_max

        logp_test_no_train /= len(random_shapes)
        logp_test_no_train_best /= len(random_shapes)

        log_probs.append([stim, condition, i*15, logp_test_no_train, logp_test_no_train_best])

    del fwm
    pkl.dump(log_probs, open("{0:s}_{1:s}_temp.pkl".format(stim, condition), "w"))
    return log_probs


if __name__ == '__main__':
    stimuli = ['s1', 's2', 's3', 's4', 's5']
    # stimuli = ['s1']
    stimuli_count = len(stimuli)
    stimuli_folder = './stimuli/exp2'
    results_folder = './results/exp2'

    conditions = ['inter', 'extra', 'ortho']
    # conditions = ['ortho']

    # LogProbability_weighted_best: similar to LogProbability_best and LogProbability_weighted_best
    # LogProbability_Image: An approximation of p(I_test|I != I_train)
    predictions = pd.DataFrame(index=np.arange(21*stimuli_count),
                               columns=['Stimulus Id', 'Condition', 'ViewpointDifference',
                                        'LogProbability_test_no_train', 'LogProbability_test_no_train_best'])

    shape_space_size = 20
    shape_space = []

    for i in range(shape_space_size):
        stimulus = paperclip_shape.PaperClipShape(forward_model=None, viewpoint=[np.array([np.sqrt(8.0), 0.0, 45.0])],
                                                  min_joints=2, max_joints=10, joint_count=8, min_angle=30.0,
                                                  max_angle=120.0, params={'SEGMENT_LENGTH_VARIANCE': 0.0001,
                                                                           'MAX_PIXEL_VALUE': 255.0,
                                                                           'LL_VARIANCE': 0.01})
        shape_space.append(stimulus)

    args = list(itertools.product(stimuli, conditions))
    args = [('s3', 'inter')]
    results = joblib.Parallel(n_jobs=1, verbose=5)(joblib.delayed(predict_stimulus_condition)(stimuli_folder, s, c, shape_space) for s, c in args)

    row_id = 0
    for r in results:
        predictions.iloc[row_id:(row_id+7)] = r
        row_id += 7

    predictions.to_csv("logp_test_no_train.csv")

