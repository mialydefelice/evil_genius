import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import scipy
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard

def make_correlation_plot(df, file_name):
    corr = binary_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin = -1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Plot ' + file_name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'correlation_plot_' + file_name + '.jpg'))
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'correlation_plot_' + file_name + '.pdf'))
    plt.clf()
    return

def make_jaccard_similarity_plot(df, file_name):
    res = 1- pdist(df.T, 'jaccard')
    squareform(res)
    distance = pd.DataFrame(squareform(res), index=df.columns, columns= df.columns)
    #jac_sim = 1 - pairwise_distances(df.T, metric = "jaccard")
    #jac_sim = pd.DataFrame(jac_sim, index=df.columns, columns=df.columns)

    mask = np.triu(np.ones_like(distance, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(distance, mask=mask, cmap=cmap, vmax=1, vmin = -1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Jaccard Similarity Plot ' + file_name)

    plt.tight_layout()
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'jaccard_similarity' + file_name + '.jpg'))
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'jaccard_similarity' + file_name + '.pdf'))
    plt.clf()
    return

def create_bar_plot(df, file_name):
    fig = df.sum(axis = 0, skipna = True).plot.bar(figsize=(20,10))
    plt.title('Primary drug or cormorb: ' + file_name)
    plt.tight_layout()
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'barplot_' + file_name + '.jpg'))
    plt.savefig(os.path.join(base_file_path, region, 'plots', 'barplot_' + file_name + '.pdf'))
    plt.clf()
    return