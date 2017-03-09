"""
Representation of 3D Shape

This file contains the script for calculating the total pixel mass in different views of a shape.
We would like to see if canonical view predictions are driven mainly by total pixel mass. 

Created on May 23, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl
import numpy as np

import Infer3DShape.paperclip_shape as paperclip_shape
import Infer3DShape.vision_forward_model as vfm


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)

    shape = pkl.load(open('stimuli/canonical_view/s1.pkl'))
    shape.forward_model = fwm
    shape.viewpoint = [np.array([np.sqrt(8.0), 0.0, 45.0])]
    shape.params['MAX_PIXEL_VALUE'] = 255.0
    shape.params['LL_VARIANCE'] = 0.01

    viewpoints = np.arange(0.0, 360.0, 20.0)
    tot_pixel_mass = np.zeros(viewpoints.size)
    for vp_i, vp in enumerate(viewpoints):
        print(vp)
        shape.viewpoint[0][1] = vp
        img = fwm.render(shape)
        tot_pixel_mass[vp_i] = np.sum(np.square(img))

