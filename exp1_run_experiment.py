"""
Representation of 3D Shape

This file contains the script for the canonical view effect experiment

Created on Mar 3, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp
from Infer3DShape.run_chain import run_chain


def run_experiment1(**kwargs):
    """This method runs the chain with a PaperClipShape hypothesis and given parameters for the canonical view effect
    experiment.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `Infer3DShape.run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): mame of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            max_joint_count (int): maximum number of joints in the shape
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            move_joint_variance (float): variance for the move joint move
            max_new_segment_length (float): maximum new segment length for add joint move
            max_segment_length_change (float): maximum change ratio for change segment length move
            rotate_midsegment_variance (float): variance for the rotate midsegment move
            change_viewpoint_variance (float): variance for the change viewpoint move
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
        max_joint_count = kwargs['max_joint_count']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        move_joint_variance = kwargs['move_joint_variance']
        max_new_segment_length = kwargs['max_new_segment_length']
        max_segment_length_change = kwargs['max_segment_length_change']
        rotate_midsegment_variance = kwargs['rotate_midsegment_variance']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
        burn_in = kwargs['burn_in']
        sample_count = kwargs['sample_count']
        best_sample_count = kwargs['best_sample_count']
        thinning_period = kwargs['thinning_period']
        report_period = kwargs['report_period']
        temperatures = None
        if 'temperatures' in kwargs:
            temperatures = kwargs['temperatures']
    except KeyError as e:
        raise ValueError("All experiment parameters should be provided. Missing parameter {0:s}".format(e.message))

    import numpy as np

    # read the data file
    data = np.load("{0:s}/{1:s}.npy".format(data_folder, input_file))
    render_size = data.shape[1:]

    import Infer3DShape.vision_forward_model as i3d_vfm
    fwm = i3d_vfm.VisionForwardModel(render_size=render_size, offscreen_rendering=True, custom_lighting=False)

    shape_params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value, 'SEGMENT_LENGTH_VARIANCE': 0.0001}

    viewpoint = np.array((np.sqrt(8.0), np.random.rand() * 360.0, 45.0))

    import Infer3DShape.paperclip_shape as i3d_pc
    h = i3d_pc.PaperClipShape(forward_model=fwm, viewpoint=[viewpoint], params=shape_params, min_joints=2,
                              max_joints=max_joint_count, joint_count=6, mid_segment_id=2)

    kernel_params = {'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance,
                     'MOVE_JOINT_VARIANCE': move_joint_variance,
                     'MAX_NEW_SEGMENT_LENGTH': max_new_segment_length,
                     'MAX_SEGMENT_LENGTH_CHANGE': max_segment_length_change,
                     'ROTATE_MIDSEGMENT_VARIANCE': rotate_midsegment_variance}

    import Infer3DShape.i3d_proposal as i3d_proposal
    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z, # in exp1, only rotations around z are allowed.
             'paperclip_move_joints': i3d_pc.paperclip_shape_move_joint,
             'paperclip_move_branch': i3d_pc.paperclip_shape_move_branch,
             'paperclip_change_segment_length': i3d_pc.paperclip_shape_change_segment_length,
             'paperclip_change_branch_length': i3d_pc.paperclip_shape_change_branch_length,
             'paperclip_add_remove_joint': i3d_pc.paperclip_shape_add_remove_joint,
             'paperclip_rotate_midsegment': i3d_pc.paperclip_shape_rotate_midsegment}

    import mcmclib.proposal as mcmc_proposal
    kernel = mcmc_proposal.RandomMixtureProposal(moves=moves, params=kernel_params)

    results = run_chain(name=input_file, sampler=sampler, initial_h=h, data=data, kernel=kernel, burn_in=burn_in,
                        thinning_period=thinning_period, sample_count=sample_count, best_sample_count=best_sample_count,
                        report_period=report_period, results_folder=results_folder, temperatures=temperatures)

    return results

if __name__ == "__main__":
    MAX_PIXEL_VALUE = 255.0

    experiment = exp.Experiment(name="Exp1_s5", experiment_name='exp1', experiment_method=run_experiment1,
                                grouped_params=['ll_variance', 'move_joint_variance', 'change_viewpoint_variance',
                                                'rotate_midsegment_variance'],
                                sampler=['mh'],
                                input_file=['s5_20'],
                                results_folder='./results/exp1/',
                                data_folder='./stimuli/exp1/',
                                max_joint_count=12,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],
                                change_viewpoint_variance=[10.0],
                                move_joint_variance=[0.0001],
                                max_new_segment_length=[0.6],
                                max_segment_length_change=[0.6],
                                rotate_midsegment_variance=[10.0],
                                burn_in=10000,
                                sample_count=10,
                                best_sample_count=10,
                                thinning_period=5000,
                                report_period=5000)

    experiment.run(parallel=True, num_processes=1)

    print(experiment.results)
    experiment.save('./results/exp1/')
    experiment.append_csv('./results/exp1/Exp1_s5.csv')

    """
    's1_0', 's1_20', 's1_40', 's1_60', 's1_80', 's1_100', 's1_120', 's1_140',
                                            's1_160', 's1_180', 's1_200', 's1_220', 's1_240', 's1_260', 's1_280',
                                            's1_300', 's1_320', 's1_340'
    """
