"""
Representation of 3D Shape

This file contains the script for generating stimuli for experiment 2 (view generalization experiment in
Bulthoff & Edelman, 1992).
These images are used for running the MCMC chains and calculating model predictions.

Created on Apr 6, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl
import numpy as np

import Infer3DShape.vision_forward_model as vfm


def save_and_render_stimulus(name, shape):
    r, theta, _ = shape.viewpoint[0]
    # during training, subjects see views: theta = 270, 285, 300, 345, 0, 15
    shape.viewpoint = []
    shape.viewpoint.append((r, (theta + 270) % 360, 45))
    shape.viewpoint.append((r, (theta + 285) % 360, 45))
    shape.viewpoint.append((r, (theta + 300) % 360, 45))
    shape.viewpoint.append((r, (theta + 345) % 360, 45))
    shape.viewpoint.append((r, (theta + 0) % 360, 45))
    shape.viewpoint.append((r, (theta + 15) % 360, 45))
    fwm.save_render("stimuli/exp2/{0:s}_train.png".format(name), shape)
    img = fwm.render(shape)
    np.save("stimuli/exp2/{0:s}_train.npy".format(name), img)

    # there are three conditions: inter, extra, and ortho
    # in inter, we test views theta = 0, 345, 330, 315, 300, 285, 270
    shape.viewpoint = []
    shape.viewpoint.append((r, (theta + 0) % 360, 45))
    shape.viewpoint.append((r, (theta + 345) % 360, 45))
    shape.viewpoint.append((r, (theta + 330) % 360, 45))
    shape.viewpoint.append((r, (theta + 315) % 360, 45))
    shape.viewpoint.append((r, (theta + 300) % 360, 45))
    shape.viewpoint.append((r, (theta + 285) % 360, 45))
    shape.viewpoint.append((r, (theta + 270) % 360, 45))
    fwm.save_render("stimuli/exp2/{0:s}_inter.png".format(name), shape)
    img = fwm.render(shape)
    np.save("stimuli/exp2/{0:s}_inter.npy".format(name), img)

    # in extra, we test views theta = 0, 15, 30, 45, 60, 75, 90
    shape.viewpoint = []
    shape.viewpoint.append((r, (theta + 0) % 360, 45))
    shape.viewpoint.append((r, (theta + 15) % 360, 45))
    shape.viewpoint.append((r, (theta + 30) % 360, 45))
    shape.viewpoint.append((r, (theta + 45) % 360, 45))
    shape.viewpoint.append((r, (theta + 60) % 360, 45))
    shape.viewpoint.append((r, (theta + 75) % 360, 45))
    shape.viewpoint.append((r, (theta + 90) % 360, 45))
    fwm.save_render("stimuli/exp2/{0:s}_extra.png".format(name), shape)
    img = fwm.render(shape)
    np.save("stimuli/exp2/{0:s}_extra.npy".format(name), img)

    # in ortho , we test views phi = 45, 30, 15, 0, -15, -30, -45
    # note that we need to be careful when phi becomes negative.
    shape.viewpoint = []
    shape.viewpoint.append((r, theta, 45))
    shape.viewpoint.append((r, theta, 30))
    shape.viewpoint.append((r, theta, 15))
    shape.viewpoint.append((r, theta, 0))
    shape.viewpoint.append((r, (theta + 180) % 360, 15))
    shape.viewpoint.append((r, (theta + 180) % 360, 30))
    shape.viewpoint.append((r, (theta + 180) % 360, 45))
    fwm.save_render("stimuli/exp2/{0:s}_ortho.png".format(name), shape)
    img = fwm.render(shape)
    np.save("stimuli/exp2/{0:s}_ortho.npy".format(name), img)


if __name__ == "__main__":
    stimuli_folder = "stimuli/exp2"

    # stimuli names
    names = ['s10']
    # names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    # names = ['s11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20']

    # theta=0.0 viewpoint
    viewpoint_thetas = pkl.load(open("{0:s}/viewpoint_thetas.pkl".format(stimuli_folder)))

    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)

    for name in names:
        shape = pkl.load(open("{0:s}/{1:s}.pkl".format(stimuli_folder, name)))
        shape.viewpoint = [[np.sqrt(8.0), viewpoint_thetas[name], 45.0]]
        save_and_render_stimulus(name, shape)


