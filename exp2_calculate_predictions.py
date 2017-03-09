"""
Representation of 3D Shape

This file contains the script for calculating the predictions of our model for experiment 2 on view-dependency.

Created on Mar 17, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import os
import fnmatch

import numpy as np
import pandas as pd

from Infer3DShape.similarity.calculate_similarity import calculate_similarity_image_given_image
import Infer3DShape.vision_forward_model as vfm
import mcmclib.mcmc_run as mcmc_run


def predict_condition(stimulus, condition, samples, sample_log_probs, data, viewpoints):
    log_probs = []
    for i in range(7):
        # reset viewpoints
        for sid, sample in enumerate(samples):
            sample.viewpoint = [viewpoints[sid]]

        # calculate similarities
        logp, logp_w, logp_best, logp_wbest = calculate_similarity_image_given_image(data[i], samples, sample_log_probs)

        # calculate logp(image_test), (as an approximation of logp(image_test|I != image_train), which is approximately
        #   N(||image_test||; 0, sigma)
        mse = np.sum(np.square(data[i] / sample.params['MAX_PIXEL_VALUE'])) / data[i].size
        logp_image = -mse / (2 * sample.params['LL_VARIANCE'])

        log_probs.append([stimulus, condition, i*15, logp, logp_best, logp_w, logp_wbest, logp_image])

    return log_probs


def read_samples(forward_model, folder, files, ids, stimulus, use_best=False):
    samples = []
    sample_log_probs = []
    viewpoints = []

    # read the samples
    for run_id in ids[stimulus]:
        run_file_pattern = "{0:s}_*_*_{1:06d}.pkl".format(stimulus, run_id)
        run_file = fnmatch.filter(files, run_file_pattern)

        # load run
        run = mcmc_run.MCMCRun.load("{0:s}/{1:s}".format(folder, run_file[0]))

        if use_best:
            run_samples = run.best_samples.samples
            run_sample_log_probs = run.best_samples.log_probs
        else:
            run_samples = run.samples.samples
            run_sample_log_probs = run.samples.log_probs

        for sample, log_prob in zip(run_samples, run_sample_log_probs):
            samples.append(sample)
            sample_log_probs.append(log_prob)
            sample.params['LL_VARIANCE'] = 0.01
            sample.params['MAX_PIXEL_VALUE'] = 255.0
            sample.forward_model = forward_model
            # store the theta=0.0 viewpoint. 5th viewpoint is the theta=0.0 view.
            viewpoints.append(sample.viewpoint[4])

    return samples, sample_log_probs, viewpoints


def main(stimulus):
    stimuli_folder = './stimuli/exp2'
    results_folder = './results/exp2'

    conditions = ['inter', 'extra', 'ortho']

    use_best_samples = False

    run_list_file = './results/exp2/20160616.csv'
    run_list = pd.read_csv(run_list_file, index_col=0)

    # find the runs with minimum mse, i.e., max. probability
    run_list['stimulus_name'] = run_list['input_file'].apply(lambda x: x.split('_')[0])
    # run_ids = run_list.groupby('stimulus_name').apply(lambda x: x['run_id'][np.argmin(x['mse'])])
    run_ids = run_list.groupby('stimulus_name').apply(lambda x: x['run_id'])
    # get the filenames of run results
    run_files = os.listdir(results_folder)

    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)

    # LogProbability: p(I_test|I_train) with p(I_test|S) calculated by integrating out viewpoint
    # LogProbability_best: p(I_test|I_train) with p(I_test|S) calculated by picking the best viewpoint
    # LogProbability_weighted: p(I_test|I_train) with each sample from p(S|I_train) weighted by its posterior prob.
    #   Note that this is not per se a proper way of estimation but it is an approximation and in cases of where a
    #   single sample dominates the posterior, it is quite close to MAP estimate.
    # LogProbability_weighted_best: similar to LogProbability_best and LogProbability_weighted_best
    # LogProbability_Image: An approximation of p(I_test|I != I_train)
    predictions = pd.DataFrame(index=np.arange(21),
                               columns=['Stimulus Id', 'Condition', 'ViewpointDifference', 'LogProbability',
                                        'LogProbability_best', 'LogProbability_weighted',
                                        'LogProbability_weighted_best', 'LogProbability_Image'],
                               dtype=float)

    row_id = 0
    print(stimulus)

    stim_samples, stim_sample_log_probs, stim_viewpoints = read_samples(fwm, results_folder, run_files, run_ids,
                                                                        stimulus, use_best_samples)

    for condition in conditions:
        print condition
        filename = "{0:s}/{1:s}_{2:s}.npy".format(stimuli_folder, stimulus, condition)
        condition_data = np.load(filename)

        condition_predictions = predict_condition(stimulus, condition, stim_samples, stim_sample_log_probs,
                                                  condition_data, stim_viewpoints)

        predictions.iloc[row_id:(row_id+7)] = condition_predictions
        row_id += 7

    predictions.to_csv("predictions{0:s}_{1:s}.csv".format("_best" if use_best_samples else "", stimulus))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stimulus_name", help="Name of the stimulus to make predictions for, e.g., s1")
    args = parser.parse_args()
    # print args.stimulus_name
    main(args.stimulus_name)
