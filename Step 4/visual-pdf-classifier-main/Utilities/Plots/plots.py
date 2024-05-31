import numpy as np
from matplotlib import pyplot as plt


def plot_distance_matrix(shift_matrix, labels, path=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.matshow(shift_matrix, cmap=plt.cm.Blues)

    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(labels))]
    ax.set_prop_cycle('color', colors)

    grid_labels = labels
    ax.set_xticks([i for i in range(len(grid_labels))])
    ax.set_yticks([i for i in range(len(grid_labels))])
    ax.set_xticklabels(grid_labels, size=5)
    ax.set_yticklabels(grid_labels, size=5)

    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45, size=5,
             ha="left", va="center", rotation_mode="anchor")

    for i in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, i, "{:.1f}".format(shift_matrix[i, c]), va='center', ha='center', size=5)
    plt.gcf().set_dpi(300)

    if path:
        plt.savefig(path,dpi=500)

    plt.show()

def plot_embeddings(embeddings_dict, xlim=None, ylim=None, path=None):
    plt.figure(figsize=(10, 10))
    for sub_class, class_embeddings in embeddings_dict.items():
        plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], alpha=0.5)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(embeddings_dict.keys())

    if path:
        plt.savefig(path,dpi=500)

    plt.show()
