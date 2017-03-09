"""
Representation of 3D Shape

This file contains the script for generating stimuli for canonical view experiment.
This is similar to Experiment 1 in Edelman and Bulthoff (1992)

Created on May 21, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl
import numpy as np

import Infer3DShape.vision_forward_model as vfm


def save_and_render_stimulus(folder, name, stimulus):
    for theta in range(0, 360, 20):
        stimulus.viewpoint[0][1] = theta
        stimulus.forward_model.save_render("{0:s}/{1:s}_{2:d}.png".format(folder, name, theta), stimulus)


def save_numpy_arrays(folder, name, stimulus):
    for theta in range(0, 360, 20):
        stimulus.viewpoint[0][1] = theta
        r = stimulus.forward_model.render(stimulus)
        np.save("{0:s}/{1:s}_{2:d}.npy".format(folder, name, theta), r)


if __name__ == "__main__":
    stimuli_folder = "stimuli/exp1/"

    # stimuli names
    names = ['s1']
    # names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']

    # render_size = (400, 400)
    render_size = (200, 200)

    fwm = vfm.VisionForwardModel(render_size=render_size, offscreen_rendering=False, custom_lighting=False)

    for stimulus_name in names:
        shape = pkl.load(open("{0:s}/{1:s}.pkl".format(stimuli_folder, stimulus_name)))
        shape.viewpoint = [[np.sqrt(8.0), 0.0, 45.0]]
        shape.forward_model = fwm
        # save_and_render_stimulus(stimuli_folder, stimulus_name, shape)
        save_numpy_arrays(stimuli_folder, stimulus_name, shape)

