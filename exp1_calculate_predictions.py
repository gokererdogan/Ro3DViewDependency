"""
Representation of 3D Shape

This file contains the script for calculating the predictions of our model for experiment 1 (canonical view experiment).

Created on May 26, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import os

import fnmatch
import warnings

import numpy as np
import pandas as pd

from Infer3DShape.similarity.calculate_similarity import calculate_similarity_image_given_image
import Infer3DShape.vision_forward_model as vfm
import mcmclib.mcmc_run as mcmc_run


def probability_test_given_training(samples, sample_log_probs, data, viewpoints):
    # reset viewpoints
    for sid, sample in enumerate(samples):
        sample.viewpoint = [viewpoints[sid]]

    # calculate similarities
    logp, logp_w, logp_best, logp_wbest = calculate_similarity_image_given_image(data, samples, sample_log_probs)

    # calculate logp(image_test), (as an approximation of logp(image_test|I != image_train), which is approximately
    #   N(||image_test||; 0, sigma)
    mse = np.sum(np.square(data / samples[0].params['MAX_PIXEL_VALUE'])) / data.size
    logp_image = -mse / (2 * samples[0].params['LL_VARIANCE'])

    return logp, logp_best, logp_w, logp_wbest, logp_image


if __name__ == '__main__':
    train_stimuli = ['s1_80', 's5_20']
    test_stimuli = ['s1', 's5']
    stimuli_folder = './stimuli/exp1'
    results_folder = './results/exp1'

    test_viewpoints = range(0, 360, 20)
    use_best_samples = False

    fwm = vfm.VisionForwardModel(render_size=(200, 200), custom_lighting=False)

    # LogProbability: p(I_test|I_train) with p(I_test|S) calculated by integrating out viewpoint
    # LogProbability_best: p(I_test|I_train) with p(I_test|S) calculated by picking the best viewpoint
    # LogProbability_weighted: p(I_test|I_train) with each sample from p(S|I_train) weighted by its posterior prob.
    #   Note that this is not per se a proper way of estimation but it is an approximation and in cases of where a
    #   single sample dominates the posterior, it is quite close to MAP estimate.
    # LogProbability_weighted_best: similar to LogProbability_best and LogProbability_weighted_best
    # LogProbability_Image: An approximation of p(I_test|I != I_train)
    predictions = pd.DataFrame(index=np.arange(len(test_viewpoints) * len(test_stimuli) * len(train_stimuli)),
                               columns=['TrainingStimulus', 'TestStimulus', 'TestViewpoint', 'LogProbability',
                                        'LogProbability_best', 'LogProbability_weighted',
                                        'LogProbability_weighted_best', 'LogProbability_Image'],
                               dtype=float)

    row_id = 0
    for train_stimulus in train_stimuli:
        print train_stimulus
        run_list_file = '{0:s}/Exp1_{1:s}.csv'.format(results_folder, train_stimulus)
        run_list = pd.read_csv(run_list_file, index_col=0)

        # find the runs with minimum mse, i.e., max. probability
        run_ids = run_list.groupby('input_file').apply(lambda x: x['run_id'][np.argmin(x['mse'])])
        # get the filenames of run results
        run_files = os.listdir(results_folder)

        # read the samples
        run_id = run_ids[train_stimulus]
        run_file_pattern = "{0:s}_*_{1:06d}.pkl".format(train_stimulus, run_id)
        run_file = fnmatch.filter(run_files, run_file_pattern)
        if len(run_file) > 1:
            warnings.warn("Multiple run files for {0:s}".format(train_stimulus))

        # load run
        run = mcmc_run.MCMCRun.load("{0:s}/{1:s}".format(results_folder, run_file[0]))

        if use_best_samples:
            run_samples = run.best_samples.samples
            run_sample_log_probs = run.best_samples.log_probs
        else:
            run_samples = run.samples.samples
            run_sample_log_probs = run.samples.log_probs

        run_viewpoints = []
        for sample in run_samples:
            sample.params['LL_VARIANCE'] = 0.01
            sample.params['MAX_PIXEL_VALUE'] = 255.0
            sample.forward_model = fwm
            run_viewpoints.append(sample.viewpoint[0])

        for test_stimulus in test_stimuli:
            print "\t{0:s}".format(test_stimulus)
            for vp in test_viewpoints:
                print "\t\t{0:d}".format(vp)
                filename = "{0:s}/{1:s}_{2:d}.npy".format(stimuli_folder, test_stimulus, vp)
                test_img = np.load(filename)

                vp_predictions = probability_test_given_training(run_samples, run_sample_log_probs, test_img,
                                                                 run_viewpoints)

                predictions.iloc[row_id] = [train_stimulus, test_stimulus, vp] + list(vp_predictions)
                row_id += 1
            print

    """
    # for prediction of recognition accuracies, we use P(I_test|I = I_train) - P(I_test|I != I_train)
    model_prediction = np.exp(predictions['LogProbability_weighted_best']) - np.exp(predictions['LogProbability_Image'])
    predictions['ModelAccuracyPrediction_weighted_best'] = model_prediction
    model_prediction = np.exp(predictions['LogProbability_weighted']) - np.exp(predictions['LogProbability_Image'])
    predictions['ModelAccuracyPrediction_weighted'] = model_prediction
    model_prediction = np.exp(predictions['LogProbability_best']) - np.exp(predictions['LogProbability_Image'])
    predictions['ModelAccuracyPrediction_best'] = model_prediction
    model_prediction = np.exp(predictions['LogProbability']) - np.exp(predictions['LogProbability_Image'])
    predictions['ModelAccuracyPrediction'] = model_prediction

    predictions.to_csv("{0:s}/predictions{1:s}.csv".format(results_folder, "_best" if use_best_samples else ""))
    """
