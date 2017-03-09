"""
Representation of 3D Shape

This file contains the script for plotting model predictions and experimental data from
Bulthoff and Edelman (1992).

Created on Apr 24, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.xmargin'] = 0.001
# mpl.rcParams['axes.ymargin'] = 0.01
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.alpha'] = 0.6
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['figure.figsize'] = (10, 8)
# mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'png'


def plot_figure(data, sem=True):
    # average all stimuli
    avg_pred = data.groupby(['Condition', 'ViewpointDifference']).mean()
    avg_pred_sem = data.groupby(['Condition', 'ViewpointDifference']).sem()
    
    inter = avg_pred.loc['inter']
    inter_sem = avg_pred_sem.loc['inter']
    extra = avg_pred.loc['extra']
    extra_sem = avg_pred_sem.loc['extra']
    ortho = avg_pred.loc['ortho']
    ortho_sem = avg_pred_sem.loc['ortho']
    x = inter.index.get_level_values('ViewpointDifference')

    plt.figure()
    plt.errorbar(x, inter['Prediction'], yerr=inter_sem['Prediction'])
    plt.errorbar(x, extra['Prediction'], yerr=extra_sem['Prediction'])
    plt.errorbar(x, ortho['Prediction'], yerr=ortho_sem['Prediction'])
    plt.legend(['interpolation', 'extrapolation', 'orthogonal'], loc='best')

    plt.xlabel("Viewpoint difference")
    plt.ylabel("Posterior ratio")

    plt.xlim([-5, 95])
    plt.xticks(range(0, 91, 15))

    
if __name__ == "__main__":
    import pandas as pd

    use_best_samples = False
    use_best_str = '_best' if use_best_samples else ''
    predictions = pd.read_csv('../predictions{0:s}.csv'.format(use_best_str), index_col=0)
        
    plot_column = 'LogProbability_best'

    # difference between posterior probabilities
    # predictions['Prediction'] = np.exp(predictions['LogProbability_Image']) - np.exp(predictions[plot_column])
    # Bayes factor, i.e., posterior ratio
    predictions['Prediction'] = np.exp(predictions['LogProbability_Image'] - predictions[plot_column])
    # Log Bayes factor
    # predictions['Prediction'] = predictions['LogProbability_Image'] - predictions[plot_column]

    # set the error rate for view-0 to be the same across all objects
    # predictions = set_view0_error_equal(predictions)

    plot_figure(predictions)
    
    plt.savefig("model_results{0:s}.png".format(use_best_str))

    # ------------------------------------------------------------------------------------ #
    #
    # plot bulthoff and edelman (1992) results
    exp_results = pd.read_csv('bulthoff_edelman_1992.csv', index_col=0)
    exp_results = exp_results.groupby(['Condition', 'ViewpointDifference']).mean()

    inter = exp_results.loc['inter']
    extra = exp_results.loc['extra']
    ortho = exp_results.loc['ortho']
    x = inter.index.get_level_values('ViewpointDifference')

    plt.figure()
    plt.errorbar(x, inter['ErrorRate'], yerr=inter['ErrorSD'])
    plt.errorbar(x, extra['ErrorRate'], yerr=extra['ErrorSD'])
    plt.errorbar(x, ortho['ErrorRate'], yerr=ortho['ErrorSD'])
    plt.legend(['interpolation', 'extrapolation', 'orthogonal'], loc='best')

    plt.xlabel("Viewpoint difference")
    plt.ylabel("Error rate")

    plt.xlim([-5, 95])
    plt.xticks(range(0, 91, 15))

    plt.savefig("bulthoff_edelman_1992.png")
