"""
Representation of 3D Shape

This file contains the script for generating paperclip stimuli.

Created on Mar 3, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl
import numpy as np

import Infer3DShape.vision_forward_model as vfm
import Infer3DShape.paperclip_shape as paperclip_shape


def get_random_viewpoint():
    theta = (np.random.rand() * 360.0) - 180.0
    return np.array((np.sqrt(8.0), theta, 45.0))


def save_and_render_stimulus(stimulus_id, stimulus):
    pkl.dump(stimulus, open("stimuli/s{0:d}.pkl".format(stimulus_id), "w"), 2)

    for theta in range(0, 360, 10):
        stimulus.viewpoint[0][1] = theta
        r = fwm.render(stimulus)
        np.save("stimuli/s{0:d}_{1:d}.npy".format(stimulus_id, theta), r)
        fwm.save_render("stimuli/s{0:d}_{1:d}.png".format(stimulus_id, theta), stimulus)


def check_stimulus(stimulus):
    """
    for j in range(1, len(stimulus.joint_positions)-1):
        if stimulus._get_joint_angle(j) < 45.0:
            return False
    """

    moi = stimulus.calculate_moment_of_inertia()
    if moi > 1.0 or moi < 0.5:
        return False

    return True


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)

    fwm_view = vfm.VisionForwardModel(render_size=(400, 400), offscreen_rendering=False, custom_lighting=False)

    stimulus_id = 11

    while stimulus_id < 21:
        stimulus_confirmed = False
        while not stimulus_confirmed:
            stimulus_ok = False
            while not stimulus_ok:
                stimulus = paperclip_shape.PaperClipShape(forward_model=fwm, viewpoint=[get_random_viewpoint()],
                                                          min_joints=2, max_joints=10, joint_count=8, min_angle=45.0,
                                                          max_angle=120.0, params={'SEGMENT_LENGTH_VARIANCE': 0.001})
                if check_stimulus(stimulus):
                    stimulus_ok = True

            print(stimulus.calculate_moment_of_inertia())
            fwm_view._view(stimulus)

            confirm = raw_input("Accept the stimulus? (y/n) ")
            if confirm == 'y':
                stimulus_confirmed = True
                save_and_render_stimulus(stimulus_id, stimulus)

        stimulus_id += 1

