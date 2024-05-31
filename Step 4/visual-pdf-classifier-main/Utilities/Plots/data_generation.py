from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def compute_embeddings_raw(model,dataloader):
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, target in tqdm(dataloader):

            if isinstance(images,list):
                images = torch.tensor(images)

            if torch.cuda.is_available():
                images = images.cuda()

            embeddings += list(model(images).cpu().numpy())
    return embeddings

def compute_embeddings(model, dataloader):

    sublcass_embeddings_dict = defaultdict(lambda: [])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        for images, target in tqdm(dataloader):

            if isinstance(images,list):
                images = torch.tensor(images)

            if torch.cuda.is_available():
                images = images.cuda()

            b_embeddings = model(images)
            b_embeddings = list(b_embeddings.cpu().numpy())
            for embedding, subclass_name in zip(b_embeddings, target):
                sublcass_embeddings_dict[subclass_name].append(embedding)

    np_results_dict = dict()
    for sublcass_key, subclass_embeddings in sublcass_embeddings_dict.items():
        np_results_dict[sublcass_key] = np.array(subclass_embeddings)

    return np_results_dict


def compute_subclass_centroids(subclass_embeddings):
    centroids_dict = dict()

    for subclass_name, subclass_embedding in subclass_embeddings.items():
        centroid = np.mean(subclass_embedding, axis=0)
        centroids_dict[subclass_name] = centroid

    return centroids_dict


def compute_shift_matrix(subclass_embeddings):
    subclass_centroids = compute_subclass_centroids(subclass_embeddings)
    n_categories = len(subclass_centroids.keys())
    shift_matrix = np.zeros(shape=(n_categories, n_categories))

    for i, centroid_subclass_name in enumerate(subclass_centroids):

        reference_centroid = subclass_centroids[centroid_subclass_name]

        for c, embeddings_subclass_name in enumerate(subclass_embeddings):

            sum_distances = 0

            for element in subclass_embeddings[embeddings_subclass_name]:
                distance = np.linalg.norm(element - reference_centroid)
                sum_distances += distance

            shift_matrix[i, c] = round(sum_distances / len(subclass_embeddings[embeddings_subclass_name]), 2)

    return shift_matrix


def compact_sublcasses(dict_embeddins):
    return np.concatenate(list(dict_embeddins.values()))
