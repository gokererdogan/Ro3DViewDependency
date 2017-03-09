"""
Representation of 3D Shape

This file contains the script for calculating the mutual information between an image and a shape
which we use as a measure of view diagnosticity.

Created on Apr 14, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl
import numpy as np

import Infer3DShape.paperclip_shape as paperclip_shape
import Infer3DShape.vision_forward_model as vfm
import Infer3DShape.similarity.calculate_similarity as shape_similarity


def entropy(p, axis):
    return -np.sum(p * np.log(p), axis=axis)


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)

    shape = pkl.load(open('stimuli/canonical_view/s1.pkl'))
    shape.forward_model = fwm
    shape.viewpoint = [np.array([np.sqrt(8.0), 0.0, 45.0])]
    shape.params['MAX_PIXEL_VALUE'] = 255.0
    shape.params['LL_VARIANCE'] = 0.01

    shape_space = [shape]

    """
    # randomly generate candidate objects
    shape_space_size = 50

    for i in range(shape_space_size-1):
        stimulus = paperclip_shape.PaperClipShape(forward_model=fwm, viewpoint=[np.array([np.sqrt(8.0), 0.0, 45.0])],
                                                  min_joints=2, max_joints=10, joint_count=8, min_angle=30.0,
                                                  max_angle=120.0, params={'SEGMENT_LENGTH_VARIANCE': 0.0001,
                                                                           'MAX_PIXEL_VALUE': 255.0,
                                                                           'LL_VARIANCE': 0.01})
        shape_space.append(stimulus)
    """

    # use shapes from the experiment as candidate objects
    for oid in range(2, 11):
        s = pkl.load(open('stimuli/canonical_view/s{0:d}.pkl'.format(oid)))
        s.forward_model = fwm
        s.viewpoint = [np.array([np.sqrt(8.0), 0.0, 45.0])]
        s.params['MAX_PIXEL_VALUE'] = 255.0
        s.params['LL_VARIANCE'] = 0.01
        shape_space.append(s)

    viewpoints = np.arange(0.0, 360.0, 20.0)
    shape_space_size = len(shape_space)
    shape_log_probabilities_avg = np.zeros((shape_space_size, viewpoints.size))
    shape_log_probabilities_max = np.zeros((shape_space_size, viewpoints.size))
    for vp_i, vp in enumerate(viewpoints):
        print(vp)
        shape.viewpoint[0][1] = vp
        img = fwm.render(shape)

        for s_i, s in enumerate(shape_space):
            print(s_i)
            logp_avg, logp_max = shape_similarity.calculate_probability_image_given_hypothesis(img, s)
            shape_log_probabilities_avg[s_i, vp_i] = logp_avg
            shape_log_probabilities_max[s_i, vp_i] = logp_max

    shape_probs_avg = np.exp(shape_log_probabilities_avg)
    shape_probs_avg /= np.sum(shape_probs_avg, 0)

    shape_probs_max = np.exp(shape_log_probabilities_max)
    shape_probs_max /= np.sum(shape_probs_max, 0)

    mi_avg = np.log(shape_space_size) - entropy(shape_probs_avg, axis=0)
    mi_max = np.log(shape_space_size) - entropy(shape_probs_max, axis=0)

    np.save('shape_log_probs_avg.npy', shape_log_probabilities_avg)
    np.save('shape_log_probs_max.npy', shape_log_probabilities_max)
    np.save('shape_probs_avg.npy', shape_probs_avg)
    np.save('shape_probs_max.npy', shape_probs_max)
    np.save('mi_avg.npy', mi_avg)
    np.save('mi_max.npy', mi_max)

