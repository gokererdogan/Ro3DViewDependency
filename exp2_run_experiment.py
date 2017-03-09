"""
Representation of 3D Shape

This file contains the script for the famous view dependency experiment with paperclip objects presented in
Bulthoff, H. H., & Edelman, S. (1992). Psychophysical support for a two-dimensional view interpolation theory
of object recognition. Proceedings of the National Academy of Sciences of the United States of America, 89(1), 604.

Created on Mar 16, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import time

import gmllib.experiment as exp
from Infer3DShape.run_chain import run_chain


def run_experiment2(**kwargs):
    """This method runs the chain with a PaperClipShape hypothesis and given parameters for the second view-dependency
    experiment we look at.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `Infer3DShape.run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): name of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            change_viewpoint_variance (float): variance for the change viewpoint move
            max_joint_count (int): maximum number of joints in the shape. required if shape_type is 'paperclip'
            move_joint_variance (float): variance for the move joint move. required if shape_type is 'paperclip'
            max_new_segment_length (float): maximum new segment length for add joint move. required if shape_type is
                'paperclip'
            max_segment_length_change (float): maximum change ratio for change segment length move. required if
                shape_type is 'paperclip'
            rotate_midsegment_variance (float): variance for the rotate midsegment move. required if shape_type is
                'paperclip'
            burn_in (int): see `run_chain` function
            sample_count (int): see `run_chain` function
            best_sample_count (int): see `run_chain` function
            thinning_period (int): see `run_chain` function
            report_period (int): see `run_chain` function
            temperatures (list): see `run_chain` function

    Returns:
        dict: run results
    """
    try:
        input_file = kwargs['input_file']
        results_folder = kwargs['results_folder']
        data_folder = kwargs['data_folder']
        sampler = kwargs['sampler']

        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']

        max_joint_count = kwargs['max_joint_count']
        move_joint_variance = kwargs['move_joint_variance']
        max_new_segment_length = kwargs['max_new_segment_length']
        max_segment_length_change = kwargs['max_segment_length_change']
        rotate_midsegment_variance = kwargs['rotate_midsegment_variance']

        burn_in = kwargs['burn_in']
        sample_count = kwargs['sample_count']
        best_sample_count = kwargs['best_sample_count']
        thinning_period = kwargs['thinning_period']
        report_period = kwargs['report_period']

        # pid is set by Experiment class
        chain_id = kwargs['pid']

        temperatures = None
        if sampler == 'pt':
            temperatures = kwargs['temperatures']

    except KeyError as e:
        raise ValueError("All experiment parameters should be provided. Missing parameter {0:s}".format(e.message))

    import numpy as np

    # seed using chain_id to prevent parallel processes from getting the same random seed
    np.random.seed(int((time.time() * 1000) + chain_id) % 2**32)

    # read training data. subjects are trained on two sets of views 75 degrees apart.
    data = np.load("{0:s}/{1:s}_train.npy".format(data_folder, input_file))
    render_size = data.shape[1:]

    # if shape_type == 'paperclip':
    custom_lighting = False

    import Infer3DShape.vision_forward_model as i3d_vfm
    fwm = i3d_vfm.VisionForwardModel(render_size=render_size, offscreen_rendering=True, custom_lighting=custom_lighting)

    shape_params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value, 'SEGMENT_LENGTH_VARIANCE': 0.0001}

    # construct viewpoint. during training, subjects see views: theta = 270, 285, 300, 345, 0, 15
    theta = np.random.rand() * 360.0
    viewpoint1 = np.array((np.sqrt(8.0), theta + 270.0, 45.0))
    viewpoint2 = np.array((np.sqrt(8.0), theta + 285.0, 45.0))
    viewpoint3 = np.array((np.sqrt(8.0), theta + 300.0, 45.0))
    viewpoint4 = np.array((np.sqrt(8.0), theta + 345.0, 45.0))
    viewpoint5 = np.array((np.sqrt(8.0), theta + 0.0, 45.0))
    viewpoint6 = np.array((np.sqrt(8.0), theta + 15.0, 45.0))
    viewpoint = [viewpoint1, viewpoint2, viewpoint3, viewpoint4, viewpoint5, viewpoint6]

    # construct initial hypothesis and kernel
    import Infer3DShape.i3d_proposal as i3d_proposal
    kernel_params = {'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}
    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z}

    import Infer3DShape.paperclip_shape as i3d_pc
    h = i3d_pc.PaperClipShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params,
                              min_joints=2, max_joints=max_joint_count, joint_count=6, mid_segment_id=2)

    kernel_params['MOVE_JOINT_VARIANCE'] = move_joint_variance
    kernel_params['MAX_NEW_SEGMENT_LENGTH'] = max_new_segment_length
    kernel_params['MAX_SEGMENT_LENGTH_CHANGE'] = max_segment_length_change
    kernel_params['ROTATE_MIDSEGMENT_VARIANCE'] = rotate_midsegment_variance

    moves['paperclip_move_joints'] = i3d_pc.paperclip_shape_move_joint
    moves['paperclip_move_branch'] = i3d_pc.paperclip_shape_move_branch
    moves['paperclip_change_segment_length'] = i3d_pc.paperclip_shape_change_segment_length
    moves['paperclip_change_branch_length'] = i3d_pc.paperclip_shape_change_branch_length
    moves['paperclip_add_remove_joint'] = i3d_pc.paperclip_shape_add_remove_joint
    moves['paperclip_rotate_midsegment'] = i3d_pc.paperclip_shape_rotate_midsegment

    import mcmclib.proposal as mcmc_proposal
    kernel = mcmc_proposal.RandomMixtureProposal(moves=moves, params=kernel_params)

    results = run_chain(name=input_file, sampler=sampler, initial_h=h, data=data, kernel=kernel, burn_in=burn_in,
                        thinning_period=thinning_period, sample_count=sample_count, best_sample_count=best_sample_count,
                        report_period=report_period, results_folder=results_folder, temperatures=temperatures)

    return results


if __name__ == "__main__":
    experiment = exp.Experiment(name="20160616", experiment_name='exp2', experiment_method=run_experiment2,
                                # grouped_params=['change_viewpoint_variance', 'change_size_variance'],
                                # grouped_params=['move_joint_variance', 'change_viewpoint_variance',
                                #                'rotate_midsegment_variance'],
                                sampler=['mh'],
                                input_file=['s19', 's19', 's19', 's19',
                                            's20', 's20', 's20', 's20'],
                                results_folder='./results/exp2/',
                                data_folder='./stimuli/exp2/',
                                max_pixel_value=255.0,  # for shape_type = paperclip
                                ll_variance=[0.0001],
                                change_viewpoint_variance=[10.0],
                                max_joint_count=12,  # for shape_type = paperclip
                                move_joint_variance=[0.0001],  # for shape_type = paperclip
                                max_new_segment_length=[0.6],  # for shape_type = paperclip
                                max_segment_length_change=[0.6],  # for shape_type = paperclip
                                rotate_midsegment_variance=[10.0],  # for shape_type = paperclip
                                burn_in=20000,
                                sample_count=2,
                                best_sample_count=2,
                                thinning_period=5000,
                                report_period=5000)

    experiment.run(parallel=True, num_processes=-1)

    print(experiment.results)
    experiment.save('./results/exp2/')
    experiment.append_csv('./results/exp2/20160616.csv')
