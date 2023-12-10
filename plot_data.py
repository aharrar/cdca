import plotly.graph_objs as go

'''
Created on Sep 15, 2016
@author: urishaham
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib


# matplotlib.use('TkAgg')

def plot_scatter(src_pca, target_pca, labels_b1, labels_b2, plot_dir, title='before-calibrationp'):
    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title=f"{title}-batch",
                name1='src', name2='target', to_plot_labels=False, plots_dir=plot_dir)

    scatterHist(src_pca[:, 0],
                src_pca[:, 1],
                target_pca[:, 0],
                target_pca[:, 1],
                labels_b1, labels_b2,
                "pc1", "pc2", title=f"{title}-labels",
                name1='src', name2='target', to_plot_labels=True, plots_dir=plot_dir)


# matplotlib.use('TkAgg')

def scatterHist(x1, x2, y1, y2, l1, l2, axis1='', axis2='', title='', name1='', name2='',
                plots_dir='', to_plot_labels=True):
    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    if to_plot_labels:

        l1_mask0 = l1 == 0
        l1_mask1 = l1 == 1
        l2_mask0 = l2 == 0
        l2_mask1 = l2 == 1

        x1_, x2_ = np.hstack((x1[l1_mask0], y1[l2_mask0])), np.hstack((x2[l1_mask0], y2[l2_mask0]))
        y1_, y2_ = np.hstack((x1[l1_mask1], y1[l2_mask1])), np.hstack((x2[l1_mask1], y2[l2_mask1]))
        axScatter.scatter(x1_, x2_, color='blue', s=3)
        axScatter.scatter(y1_, y2_, color='red', s=3)
    else:
        x1_, x2_ = x1, x2
        y1_, y2_ = y1, y2
        axScatter.scatter(x1_, x2_, color='blue', s=3)
        axScatter.scatter(y1_, y2_, color='red', s=3)

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = np.max([np.max(np.fabs(x1_)), np.max(np.fabs(x2_))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x1_, bins=bins, color='blue', density=True, stacked=True, histtype='step')
    axHisty.hist(x2_, bins=bins, orientation='horizontal', color='blue', density=True, stacked=True, histtype='step')
    axHistx.hist(y1_, bins=bins, color='red', density=True, stacked=True, histtype='step')
    axHisty.hist(y2_, bins=bins, orientation='horizontal', color='red', density=True, stacked=True, histtype='step')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axHistx.set_xticklabels([])
    axHistx.set_yticklabels([])
    axHisty.set_xticklabels([])
    axHisty.set_yticklabels([])
    axScatter.set_xlabel(axis1, fontsize=18)
    axScatter.set_ylabel(axis2, fontsize=18)

    axHistx.set_title(title, fontsize=18)
    axHistx.set_rasterized(True)

    axScatter.legend([name1, name2], fontsize=18)
    if not plots_dir == '':
        fig.savefig(plots_dir + '/' + title + '.eps', format='eps')
        fig.savefig(plots_dir + '/' + title + '.png', format='png')
    else:
        plt.show()


def plot2d(sample1, sample2):
    # Define the traces for each set of data
    trace1 = go.Scatter(x=sample1[:, 0], y=sample1[:, 1], mode='markers', name='sample-1', line=dict(color='blue'))
    trace2 = go.Scatter(x=sample2[:, 0], y=sample2[:, 1], mode='markers', name='sample-2', line=dict(color='red'))

    # Create a figure with the traces and layout
    fig = go.Figure(data=[trace1, trace2], layout=go.Layout(title='Multiple Resources of Data'))

    # Show the plot
    fig.show()
    plt.show()


from sklearn import decomposition


def plot_data(sample1, sample2):
    pca = decomposition.PCA(n_components=2)
    pca.fit(sample1)
    pc1 = 0
    pc2 = 1
    axis1 = 'PC' + str(pc1)
    axis2 = 'PC' + str(pc2)

    # plot data before calibration
    sample1_pca = pca.transform(sample1)
    sample2_pca = pca.transform(sample2)
    # plot2d(sample1_pca,sample2_pca)
    scatterHist(sample1_pca[:, pc1],
                sample1_pca[:, pc2],
                sample2_pca[:, pc1],
                sample2_pca[:, pc2],
                axis1,
                axis2,
                title="Data before calibration",
                name1='sample1',
                name2='sample2')
    plt.show()


def get_pca_data(sample):
    pca = decomposition.PCA(n_components=2)
    pca.fit(sample)

    return pca.transform(sample)


def plot_pca_data_cdca(src_pca, tgt_pca, src_labels, tgt_labels):
    import plotly.express as px

    # Add labels to each data point
    import plotly.graph_objects as go
    src_pca_p = np.array([[value[0], value[1]] for index, value in enumerate(src_pca) if src_labels[index] == 1])
    src_pca_n = np.array([[value[0], value[1]] for index, value in enumerate(src_pca) if src_labels[index] == 0])
    tgt_pca_p = np.array([[value[0], value[1]] for index, value in enumerate(tgt_pca) if tgt_labels[index] == 1])
    tgt_pca_n = np.array([[value[0], value[1]] for index, value in enumerate(tgt_pca) if tgt_labels[index] == 0])
    print(src_pca_p.shape)
    print(src_pca_n.shape)

    # Determine marker colors for source data

    # Determine marker colors for target data

    # Create the figure
    fig = go.Figure()

    # Add source scatter trace
    fig.add_trace(
        go.Scatter(
            x=src_pca_p[:, 0],
            y=src_pca_p[:, 1],
            mode='markers',
            marker=dict(color='green'),
            name="src positive"
        )
    )

    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=tgt_pca_p[:, 0],
            y=tgt_pca_p[:, 1],
            mode='markers',
            marker=dict(color='yellow'),
            name="target positive"
        )
    )
    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=src_pca_n[:, 0],
            y=src_pca_n[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name="src negative"
        )
    )
    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=tgt_pca_n[:, 0],
            y=tgt_pca_n[:, 1],
            mode='markers',
            marker=dict(color='red')
            , name="target negative"
        )
    )

    # Show the figure
    fig.show()

def plot_pca_data(src_pca, tgt_pca):
    import plotly.express as px

    # Add labels to each data point
    import plotly.graph_objects as go
    # Create the figure
    fig = go.Figure()

    # Add source scatter trace
    fig.add_trace(
        go.Scatter(
            x=src_pca[:, 0],
            y=src_pca[:, 1],
            mode='markers',
            marker=dict(color='green'),
            name="src"
        )
    )

    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=tgt_pca[:, 0],
            y=tgt_pca[:, 1],
            mode='markers',
            marker=dict(color='red')
            , name="target"
        )
    )

    # Show the figure
    fig.show()


def plot_list(lst, title):
    plt.plot(lst)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.show()

def plot_pca_data1(src_pca, tgt_pca, src_labels, tgt_labels):
    import plotly.express as px

    # Add labels to each data point
    import plotly.graph_objects as go
    src_pca_p = np.array([[value[0], value[1]] for index, value in enumerate(src_pca) if src_labels[index] == 1])
    src_pca_n = np.array([[value[0], value[1]] for index, value in enumerate(src_pca) if src_labels[index] == 0])
    tgt_pca_p = np.array([[value[0], value[1]] for index, value in enumerate(tgt_pca) if tgt_labels[index] == 1])
    tgt_pca_n = np.array([[value[0], value[1]] for index, value in enumerate(tgt_pca) if tgt_labels[index] == 0])

    # Determine marker colors for source data

    # Determine marker colors for target data

    # Create the figure
    fig = go.Figure()

    # Add source scatter trace
    fig.add_trace(
        go.Scatter(
            x=src_pca_p[:, 0],
            y=src_pca_p[:, 1],
            mode='markers',
            marker=dict(color='green'),
            name="src positive"
        )
    )

    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=tgt_pca_p[:, 0],
            y=tgt_pca_p[:, 1],
            mode='markers',
            marker=dict(color='yellow'),
            name="target positive"
        )
    )
    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=src_pca_n[:, 0],
            y=src_pca_n[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            name="src negative"
        )
    )
    # Add target scatter trace
    fig.add_trace(
        go.Scatter(
            x=tgt_pca_n[:, 0],
            y=tgt_pca_n[:, 1],
            mode='markers',
            marker=dict(color='red')
            , name="target negative"
        )
    )

    # Show the figure
    fig.show()
