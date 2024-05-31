import datetime
import functools
import json
import os
import pathlib
import random
import sys
from itertools import chain
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np
import requests
import fitz
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from Data.Datasets.SimpleMonoclassDataset import SimpleMonoclassDataset
from Utilities.Confs.Configs import Configs
import datetime as dt
from torch.utils.data import DataLoader
from Data.SampleProcessors.BaseSampleProcessor import BaseSampleProcessor
from Models.FeatureExtraction.EmbeddingNet import EmbeddingNet
from Utilities.Plots.data_generation import compute_embeddings_raw
from pprint import pprint
from Utilities.Plots.plots import plot_embeddings
from sklearn import decomposition
from collections import defaultdict, Counter

# sample a color set to use in the scatter plots
cmap = plt.cm.get_cmap('hsv', 1000)
from PIL import Image
import imagehash as imagehash

sys.path.append('../')

from Data.Datalake.Phishing.Phishing import file_hash_decode, prepare_phishing_datalake
from pprint import pprint
from Utilities.Plots.plots import plot_embeddings
from sklearn import decomposition
from collections import defaultdict, Counter

###  PARAMETERS ###
# define the maximum amount of samples to process in a single sweep
batch_size = 2000

# sample a color set to use in the scatter plots
cmap = plt.cm.get_cmap('hsv', 1000)


def get_new_samples(db_cursor, date_from=dt.date.today(), date_to=dt.date.today()):
    """
    Return the newly received samples that have not yet been tested against our seo poisoning metric
    and have not yet been tested for seo_poisoning
    :param db_cursor: connection cursor to use to execute the operation
    :param date_from: date from which to get samples
    :param date_to: date up-to-which get samples
    :return: A list of samples each one with 1 attributes: [filehash]
    """

    query = """Select distinct filehash from imported_samples left join samplecluster s using (filehash)
                where is_seo is NULL and (to_ignore is NULL or to_ignore = FALSE)
                and upload_date between %s::date AND %s::date
                AND mimetype = 'application/pdf'"""

    db_cursor.execute(query, (date_from, date_to))
    res = db_cursor.fetchall()
    return [r[0] for r in res]


# In[85]:


def get_new_samples_no_ft_vector(db_cursor, date_from=dt.date.today(), date_to=dt.date.today()):
    """
    Return the newly received samples that have not yet been processed by the feature extraction model
    and have not yet been tested for seo_poisoning
    :param db_cursor: connection cursor to use to execute the operation
    :param date_from: date from which to get samples
    :param date_to: date up-to-which get samples
    :return: A list of samples each one with 1 attributes: [filehash]
    """

    query = """Select filehash from imported_samples inner join samplecluster s using (filehash)
                where ft_vector is NULL and (to_ignore is NULL or to_ignore = FALSE)
                and upload_date between %s::date AND %s::date
                AND mimetype = 'application/pdf'"""

    db_cursor.execute(query, (date_from, date_to))
    res = db_cursor.fetchall()
    return [r[0] for r in res]


# In[86]:


def get_samples_to_cluster_visually(db_cursor, date_from, date_to):
    """
    Return the set of samples that have already been processed by the feature extraction model and are ready to be clustered
    :param db_cursor: connection cursor to use to execute the operation
    :param date_from: date from which to get samples
    :param date_to: date up-to-which get samples
    :return: A list of samples each one with 2 attributes: [filehash,ft_vector]
    """

    query = """SELECT filehash,ft_vector from samplecluster inner join imported_samples c using (filehash)

            where (is_seo = True OR c.provider = 'FromUrl')
            AND (fk_cluster = -1 or fk_cluster is NULL) and ft_vector is not null AND upload_date between %s  and %s"""

    db_cursor.execute(query, (date_from, date_to))
    res = db_cursor.fetchall()
    return res


# In[87]:


def get_anchors_of_clusters(db_cursor):
    """
    Returns a dictionary containing an association cluster_key - anchor features of the cluster
    :param db_cursor: connection cursor to use to execute the operation
    :return: Dicrionary of lists
    """

    query = """ SELECT  fk_cluster,json_agg(json_build_object('filehash',filehash,'ft_vector', ft_vector))
            FROM ( SELECT ROW_NUMBER() OVER (PARTITION BY fk_cluster ORDER BY random()) AS r,
                fk_cluster,filehash,ft_vector
              FROM samplecluster INNER JOIN clusters on fk_cluster = clusters.id WHERE fk_cluster is not null and fk_cluster > -1 and (clusters.to_ignore is NULL or clusters.to_ignore = FALSE) and is_seo = TRUE 
                ) x WHERE x.r <= 20 group by fk_cluster
    """

    db_cursor.execute(query)
    res = db_cursor.fetchall()

    cluster_samples_dict = {}

    for fk_cluster, samples in list(res):
        cluster_samples_dict[fk_cluster] = [(item["filehash"], item["ft_vector"]) for item in samples]

    return cluster_samples_dict


# In[88]:


def get_centroids_of_clusters(cluster_samples_dict):
    """
    From the anchors of the clusters compute the current centroid
    """

    centroids = dict()

    for cluster_key, anchor_samples in cluster_samples_dict.items():
        anchors = [sample[1] for sample in anchor_samples]
        anchors_array = np.asarray(anchors)
        center = np.mean(anchors_array, axis=0)

        centroids[cluster_key] = np.array(center)

    return centroids


def compute_closest_cluster(embeddings, filehashes, centroids, allowed_centroid_keys):
    allowed_centroids = [centroid for key, centroid in centroids.items() if key in allowed_centroid_keys]
    selected_cluster_indexes, _ = pairwise_distances_argmin_min(embeddings, allowed_centroids)

    results = defaultdict(lambda: [])

    for i, selected_cluster in enumerate(selected_cluster_indexes):
        results[allowed_centroid_keys[selected_cluster]].append(filehashes[i])

    return results


# In[89]:


def save_new_samples_data(db_cursor, filehashes, embeddings, is_seos, original_anchor=None):
    """
    Return the set of samples that have already been processed by the feature extraction mode, are positive for poisoning, have been labelled as noise
    :param db_cursor: connection cursor to use to execute the operation
    :param filehashes: list[str]
        list of filehashes
    :param embeddings: list[list[float]]
        list of embeddings
    :param is_seos: list[bool]
        list of bits
    :return: None
    """

    if not filehashes:
        return

    if not embeddings:
        embeddings = [None] * len(filehashes)
    else:
        embeddings = [embeddings[i].tolist() for i in range(len(embeddings))]

    if not is_seos:
        is_seos = [None] * len(filehashes)

    query = """INSERT INTO samplecluster(filehash,is_seo,ft_vector,original_anchor) VALUES(%s,%s,%s,%s) 
               ON CONFLICT (filehash) DO UPDATE 
               SET 
               is_seo = COALESCE(EXCLUDED.is_seo,samplecluster.is_seo), 
               ft_vector = COALESCE(EXCLUDED.ft_vector,samplecluster.ft_vector), 
               original_anchor = COALESCE(EXCLUDED.original_anchor,samplecluster.original_anchor);"""

    for i, filehash in enumerate(filehashes):
        db_cursor.execute(query, (filehash, is_seos[i], embeddings[i], original_anchor))


# In[90]:


def mark_samples_to_ignore(db_cursor, filehashes):
    """
    Mark samples to ignore in the clustering procedure. These are all samples that are not possible to process.
    :param db_cursor: connection cursor to use to execute the operation
    :param filehashes: list[str]
        list of filehashes
    :param embeddings: list[list[float]]
        list of embeddings
    :param is_seos: list[bool]
        list of bits
    :return: None
    """

    if not filehashes:
        return

    query = """INSERT INTO samplecluster(filehash,to_ignore) VALUES(%s,%s) ON CONFLICT (filehash) DO UPDATE SET to_ignore = EXCLUDED.to_ignore"""

    for i, filehash in enumerate(filehashes):
        db_cursor.execute(query, (filehash, True))


# In[91]:


def create_new_cluster(db_cursor, fk_clustering_step, cluster_name=''):
    """
    Create a new cluster and return its unique ide
    """

    query = """INSERT INTO clusters(id,fk_clustering_step,name) VALUES((Select coalesce(max(id)+1,1) from clusters),%s,%s) RETURNING id"""
    db_cursor.execute(query, (fk_clustering_step, cluster_name))
    res = db_cursor.fetchone()
    return res[0]


# In[92]:


def save_samples_in_cluser(db_cursor, filehashes, fk_cluster, fk_clustering_step, method):
    """
    Save the passed samples in the given cluster
    """

    assert (method in ["visual", "manual", None])

    query = """UPDATE samplecluster SET fk_cluster = %s, method=%s,fk_clustering_step=%s where filehash in %s and (fk_cluster is NULL or fk_cluster = -1)"""
    db_cursor.execute(query, (fk_cluster, method, fk_clustering_step, tuple(filehashes)))

    return True


# In[93]:


def get_campaigns_of_clusters(db_cursor):
    """
    Return a dictionary containing an associated campaign key -> list of associated clusters
    :param db_cursor: connection cursor to use to execute the operation
    :return: a dictionary containing an associated campaign key -> list of associated clusters
    """

    query = """select c.fk_campaign, array_agg(c.id) from clusters c where c.fk_campaign is not null and c.fk_campaign > -1 group by c.fk_campaign"""

    db_cursor.execute(query)
    res = db_cursor.fetchall()

    result_dict = {}
    for campaign_id, clusters in res:
        result_dict[campaign_id] = list(clusters)
    return result_dict


def is_sample_seo(filehash, root_folder_samples):
    sample_path = os.path.join(root_folder_samples, file_hash_decode(filehash), filehash)

    try:
        doc = fitz.open(sample_path)

        n_pages = doc.page_count

        url_list = []
        for i in range(doc.page_count):
            page = doc.load_page(i)

            for url in page.get_links():
                if '.pdf' in url['uri']:
                    url_list.append(url['uri'])

        n_pdf_links = len(url_list)

        n_pdf_per_page = n_pdf_links / n_pages
        return filehash, n_pdf_per_page > 1 and n_pdf_links >= 5
    except:
        return filehash, False


def is_sample_valid(filehash, root_folder_screenshots):
    """
    Returns true if the given filehash corresponds to a sample in the dataset ready to be processed
    """

    screenshot_path = os.path.join(root_folder_screenshots, file_hash_decode(filehash), filehash + ".png")

    return filehash, Path(screenshot_path).exists()


def load_dataset(phishing_dataset, json_file_path):
    clustering_step_id = log_clustering_step(phishing_dataset._db_cursor, datetime.date(2020, 12, 16), -1)

    with open(json_file_path) as f:
        data = json.load(f)

        model = EmbeddingNet(1, 32, 0.3, 3)

        checkpoint_path = os.path.join(pathlib.Path(__file__).parent.resolve().parent,
                                       "Weights/new/128_3.ckpt")
        model = EmbeddingNet.load_from_checkpoint(checkpoint_path=checkpoint_path)

        for class_key, filehashes in data.items():

            filehashes_tmp = []

            for filehash in filehashes:
                if str(filehash).count('copy') == 0:
                    filehashes_tmp.append(filehash)

            filehashes = filehashes_tmp

            results = are_documents_seo(phishing_dataset, filehashes)

            filehashes = [r[0] for r in results]
            seo_results = [r[1] for r in results]

            print(f"Processing template{class_key}")

            # check if these original centroids respect our definition of seo poisoning
            if seo_results.count(True) == 0:
                # no sample of the proposed cluster respects the definition of seo_poisoning
                # do not create a cluster
                continue

            screenshots_paths = [
                os.path.join(phishing_dataset.screenshots_files_endpoint, file_hash_decode(filehash), filehash + ".png")
                for filehash in filehashes]

            # create new cluster
            new_cluster_id = create_new_cluster(phishing_dataset._db_cursor, clustering_step_id, str(class_key).lower())

            print(f"New seo cluster with {seo_results.count(True)} anchors! Id:{new_cluster_id}")

            sample_processor = BaseSampleProcessor(192, 1, advanced_augmentations=False)

            # extract the visual features from the samples
            dataset = SimpleMonoclassDataset([(path,) for path in screenshots_paths], sample_processor=sample_processor)
            dataloader = DataLoader(dataset, pin_memory=False, num_workers=48, batch_size=8)
            embeddings = compute_embeddings_raw(model, dataloader)

            # save samples in the new cluster
            save_new_samples_data(phishing_dataset._db_cursor, [str(f) for f in filehashes], embeddings,
                                  is_seos=seo_results,
                                  original_anchor=True)

            save_samples_in_cluser(phishing_dataset._db_cursor, [str(f) for f in filehashes], new_cluster_id,
                                   clustering_step_id,
                                   'manual')


def are_documents_seo(phishing_dataset, filehashes):
    with Pool(100) as pool:
        results = list(
            tqdm(pool.imap(functools.partial(is_sample_seo, root_folder_samples=phishing_dataset.root_folder),
                           filehashes), total=len(filehashes)))

    return results


def identify_seo_documents(phishing_dataset, from_date, to_date):
    """
        Get from the db all the newly inserted samples, use our seao metric to identify the ones
        likelly containing a SEO poisoning attack
    """
    # Get new samples not yet processed
    new_samples_filehashes = get_new_samples(phishing_dataset._db_cursor, from_date, to_date)

    if not new_samples_filehashes:
        return

    print(f"Identified {len(new_samples_filehashes)} new samples")

    # Check that foreach sample there is a corresponding screesnhot, if not the sample is considered invalid
    with Pool(16) as pool:
        results = list(tqdm(pool.imap(
            functools.partial(is_sample_valid, root_folder_screenshots=phishing_dataset.screenshots_files_endpoint),
            new_samples_filehashes), total=len(new_samples_filehashes)))

    filehashes = []
    filehashes_to_ignore = []

    for filehash, is_valid in results:

        if is_valid:
            filehashes.append(filehash)
        else:
            filehashes_to_ignore.append(filehash)

    print(f"Found {len(filehashes_to_ignore)} samples not having a screenshot and that will be ignored")

    results = are_documents_seo(phishing_dataset, filehashes)

    result_filehashes = []
    is_seo_results = []
    for filehash, is_seo_r in results:
        result_filehashes.append(filehash)
        is_seo_results.append(is_seo_r)

    assert len(is_seo_results) == len(result_filehashes)

    print(f"Found {is_seo_results.count(True)} SEO positive and {is_seo_results.count(False)} SEO negative samples")

    # mark samples having no screenshot as samples to ignore
    mark_samples_to_ignore(phishing_dataset._db_cursor, filehashes_to_ignore)

    # save the seo_poisoning results in the db
    save_new_samples_data(phishing_dataset._db_cursor, result_filehashes, None, is_seo_results)


# In[97]:
def generate_hashes_sample(filehash, phishing_screenshots_entrypoint):
    # compute the screenshot of the image
    path_sample_screenshot = os.path.join(phishing_screenshots_entrypoint, file_hash_decode(filehash),
                                          filehash + ".png")

    # load the image
    image = Image.open(path_sample_screenshot)

    # compute the hashes
    phash = str(imagehash.phash(image))
    whash = str(imagehash.whash(image))
    dhash = str(imagehash.dhash(image))
    average_hash = str(imagehash.average_hash(image))

    return phash, whash, dhash, average_hash


def extract_visual_hashes_from_sample(filehash, phishing_dataset, phishing_screenshots_entrypoint):
    phash, whash, dhash, average_hash = generate_hashes_sample(filehash, phishing_screenshots_entrypoint)

    with phishing_dataset._db_cursor as cur:
        cur.execute(f"""Update samplecluster
                    SET
                    screens_phash = %s,
                    screens_whash = %s,
                    screens_dhash = %s,
                    screens_ahash = %s
                    WHERE
                    filehash = %s
                    """, (phash, whash, dhash, average_hash, filehash))


def prepare_samples_for_clustering(phishing_dataset, from_date, to_date):
    """
        This function prepare new samples for being clustered, checking if they respecte
        the seo_poisoning url definition, extracting their embeddings and saving everything in the db
    """

    # get samples not having an associated feature vector
    filehashes = get_new_samples_no_ft_vector(phishing_dataset._db_cursor, from_date, to_date)

    print(f"Found {len(filehashes)} samples to extract the embeddings from")
    screenshots_paths = []

    # form a list of all the screenshots'paths
    for filehash in filehashes:
        screenshots_paths.append(
            os.path.join(phishing_dataset.screenshots_files_endpoint, file_hash_decode(filehash), filehash + ".png"))

    print("Using model: Weights/new/128_3.ckpt")
    # Load model and sample processonr
    checkpoint_path = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "Weights/new/128_3.ckpt")

    model = EmbeddingNet.load_from_checkpoint(checkpoint_path=checkpoint_path)
    sample_processor = BaseSampleProcessor(128, 3, advanced_augmentations=False)

    # extract the visual features from the samples
    dataset = SimpleMonoclassDataset([(path,) for path in screenshots_paths], sample_processor=sample_processor)
    dataloader = DataLoader(dataset, pin_memory=False, num_workers=32, batch_size=8)
    embeddings = compute_embeddings_raw(model, dataloader)

    # save the extracted embeddings and seo_poisoning results in the db
    save_new_samples_data(phishing_dataset._db_cursor, filehashes, embeddings, None)

    return

    with Pool(50) as p:
        r = list(tqdm(p.imap(functools.partial(extract_visual_hashes_from_sample, phishing_dataset=phishing_dataset,
                                               phishing_screenshots_entrypoint=phishing_dataset.screenshots_files_endpoint),
                             filehashes), total=len(filehashes), desc="Extracting visual hashes"))


def group_by_visual_hashes(phishing_dataset, filehashes):
    with phishing_dataset._db_cursor as cur:
        cur.execute(f"""    update samplecluster 
                            set fk_cluster  = t.fk_cluster
                            from (select distinct s2.fk_cluster,s.filehash
                            from samplecluster s 
                            inner join samplecluster s2 on s.screens_phash = s2.screens_phash and s.screens_whash = s2.screens_whash 
                            where s2.fk_cluster is not null and s2.fk_cluster > -1) t 
                            where t.fk_cluster is not null  
                            and t.filehash = samplecluster.filehash 
                            and samplecluster.filehash in %s
                        """, (tuple(filehashes)))

        return cur.rowcount


def cluster_using_hashes(phishing_dataset, date_from, date_to):
    samples_to_cluster_tot = get_samples_to_cluster_visually(phishing_dataset._db_cursor, date_from, date_to)

    filehashes = [f[0] for f in samples_to_cluster_tot]

    return group_by_visual_hashes(phishing_dataset, filehashes)


def cluster_samples_visually(phishing_dataset, date_from, date_to, possible_eps, visualize=False):
    # flag marking the creation of a new cluste
    discovered_new_cluster = False

    for eps in possible_eps:

        print(f'Using eps={eps}')

        # flag indicating if the clustering process resulted in a clean set of clusters or if we should try
        # with a smaller epsilon
        perfect_clustering = True

        # create a record recording this clustering step in the db
        clustering_step_id = log_clustering_step(phishing_dataset._db_cursor, date_to, eps)

        # retrieve all the samples with no assigned cluster from the db
        samples_to_cluster_tot = get_samples_to_cluster_visually(phishing_dataset._db_cursor, date_from, date_to)

        print(f"Found {len(samples_to_cluster_tot)} new samples to cluster")

        # If no new samples to cluster are found then return and do not try with a different eps
        if not samples_to_cluster_tot:
            return False

        # retrieve a set of labelled samples to use as anchors of the respective clusters
        cluster_anchors = get_anchors_of_clusters(phishing_dataset._db_cursor)

        # compute clusters centroid
        clusters_centroids = get_centroids_of_clusters(cluster_anchors)

        # shuffle the order of the samples to add variability before dividing them in batches
        random.shuffle(samples_to_cluster_tot)

        if len(samples_to_cluster_tot) > batch_size:
            print("The number of samples to cluster is excessive, clustering them by batch")

        for batch_i in range(0, len(samples_to_cluster_tot), batch_size):

            if len(samples_to_cluster_tot) > batch_size:
                print(f"Batch:{batch_i}")

            samples_to_cluster = samples_to_cluster_tot[batch_i:batch_i + batch_size]

            # create a list of all the samples we have to cluster
            samples = [sample for sample in samples_to_cluster] + [s for s in
                                                                   chain.from_iterable(cluster_anchors.values())]

            # create a list containing all the embeddings to cluster
            embeddings = [sample[1] for sample in samples]

            # create a list of the initial labels
            original_labels = [-1] * len(samples_to_cluster)
            for key, cluster_samples in cluster_anchors.items():
                original_labels += [key] * len(cluster_samples)

            # cluster the embeddings
            clustering_algorithm = DBSCAN(eps=eps, min_samples=30, n_jobs=8)

            # assign to each embedding a label
            predicted_cluster_labels = clustering_algorithm.fit_predict(np.array(embeddings))

            # datastructure to save to which cluster each sample has been assigned
            cluster_filehash_association = defaultdict(lambda: [])

            # datastructure to save
            cluster_anchors_assignments = defaultdict(lambda: [])
            cluster_anchors_assignments_filehashes = defaultdict(lambda: [])

            cluster_embeddings = defaultdict(lambda: [])

            # foreach sample processed from dbscan
            for i, predicted_label in enumerate(predicted_cluster_labels):

                # if it is an anchor sample
                if original_labels[i] > -1:

                    # add the anchor label to a list of labels associated to this cluster
                    cluster_anchors_assignments[predicted_label].append(original_labels[i])
                    cluster_anchors_assignments_filehashes[predicted_label].append(samples[i][0])


                # store the filehash to assign the sample to this cluster
                cluster_filehash_association[predicted_label].append(samples[i][0])

                # store the embeddings of the samples associated to each cluster
                cluster_embeddings[predicted_label].append(samples[i][1])

            campaigns_clusters = get_campaigns_of_clusters(phishing_dataset._db_cursor)

            for cluster_key, cluster_samples_hashes in cluster_filehash_association.items():

                # if this is the noise cluster mark the elements directly and continue
                if cluster_key == -1:
                    save_samples_in_cluser(phishing_dataset._db_cursor, cluster_samples_hashes, fk_cluster=-1,
                                           method="visual",
                                           fk_clustering_step=clustering_step_id)
                    continue

                # if this is not the noise cluster count how many associated labels it has
                counter_anchor_labels = Counter(cluster_anchors_assignments[cluster_key])
                anchors_filehashes = cluster_anchors_assignments_filehashes[cluster_key]

                # if there is no associated label this is a new cluster
                if len(counter_anchor_labels) == 0:

                    # create new cluster
                    new_cluster_id = create_new_cluster(phishing_dataset._db_cursor, clustering_step_id)

                    # save samples in the new cluster
                    save_samples_in_cluser(phishing_dataset._db_cursor, cluster_samples_hashes,
                                           fk_cluster=new_cluster_id,
                                           method="visual", fk_clustering_step=clustering_step_id)
                    discovered_new_cluster = True

                    print(f"Created cluster {new_cluster_id}")

                # if there is just one label associated to the cluster
                elif len(counter_anchor_labels) == 1:

                    # Save the samples into the pre-existing cluser
                    save_samples_in_cluser(phishing_dataset._db_cursor, cluster_samples_hashes,
                                           fk_cluster=list(counter_anchor_labels.keys())[0], method="visual",
                                           fk_clustering_step=clustering_step_id)
                else:
                    # This cluster contains anchors from multiple original clusters
                    # If the candidate clusters belong to the same campaign: merge them
                    # if they do not belong to the same category leave them unclustered and try to lunch dbscan
                    #   with a smaller eps

                    # datastructure to store the clusters that are in conflict with a specific campaign
                    conflicting_clusters_per_campaign = defaultdict(lambda: [])

                    # Flag indicating if it was possible to resolve this conflict automatically
                    resolved = False


                    print("ANALYZING conflict: ")
                    print("\n Anchors in cluster:")

                    # Foreach campaign and its belonging clusters
                    for campaign_id, clusters_of_campaign in campaigns_clusters.items():

                        # get which cluster of this campaign is connected to the conflict
                        campaign_clusters_involved = list(set(counter_anchor_labels.keys()).intersection(clusters_of_campaign))

                        # check if this campaign is connected to the conflict
                        if not campaign_clusters_involved:
                            # if not continue, testing the next campaign
                            continue

                        print(f"Clusters of campaign {campaign_id} involved in the conflict: {campaign_clusters_involved}")

                        # save the which clusters associated to this campaign are causing the conflict
                        conflicting_clusters_per_campaign[campaign_id] = campaign_clusters_involved

                        # If all labels associated to a predicted cluster belong to the same campaign
                        if set(clusters_of_campaign).issuperset(set(counter_anchor_labels.keys())):

                            # split the samples into the already existing clusters of the campaign

                            # foreach element find its closest cluster-centroid
                            closer_cluster_filehash_association = compute_closest_cluster(
                                cluster_embeddings[cluster_key],
                                cluster_samples_hashes, clusters_centroids,
                                list(counter_anchor_labels.keys()))

                            # foreach cluster and its corresponding associated filehashes
                            for clean_cluster_key, clean_filehashes in closer_cluster_filehash_association.items():
                                # save the elements in the cluster with the closet centroid
                                save_samples_in_cluser(phishing_dataset._db_cursor, clean_filehashes,
                                                       fk_cluster=clean_cluster_key, method="visual",
                                                       fk_clustering_step=clustering_step_id)

                            # The clustering is resolved, quick testing new campaigns
                            resolved = True
                            print("Conflict resolved!")
                            break

                    if not resolved:

                        # assert that there are 2 campaigns causing the exception
                        assert len(conflicting_clusters_per_campaign.keys()) > 1

                        # The conflict cannot be resolved automatically
                        # Signal that a conflict has happened
                        perfect_clustering = False

                        # save the samples in the noise cluster
                        save_samples_in_cluser(phishing_dataset._db_cursor, cluster_samples_hashes, fk_cluster=-1,
                                               method='visual',
                                               fk_clustering_step=clustering_step_id)

                        # Save a log of the conflict in the db
                        """
                        # foreach campaign involved in the cluster
                        for z, class_key in enumerate(sorted(conflicting_clusters_per_campaign.keys())):

                            conflicting_clusters = conflicting_clusters_per_campaign[class_key]

                            # foreach second campaing involved in the cluster
                            for c, second_class_key in enumerate(sorted(conflicting_clusters_per_campaign.keys())):

                                # make sure that the campaigns are different
                                if z > c:
                                    continue

                                second_conflicting_cluster = conflicting_clusters_per_campaign[second_class_key]

                                # foreach 2 clusters involved in the conflict belonging to different campaigns
                                for fk_cluster_1 in conflicting_clusters:
                                    for fk_cluster_2 in second_conflicting_cluster:

                                        if fk_cluster_1 >= fk_cluster_2:
                                            continue

                                        print(fk_cluster_1, fk_cluster_2)

                                        # log a conflig in the db
                                        log_conflict(phishing_dataset._db_cursor, clustering_step_id, fk_cluster_1,
                                                     fk_cluster_2)
                        """
                        # Hashes causing the conflicts
                        for f in anchors_filehashes:
                            print(f)

            if visualize:
                pca = decomposition.PCA(n_components=2)
                pca.fit(np.array(embeddings))

                embeddings_2_plot = {}
                for subclass, data in cluster_embeddings.items():
                    subclass_embeddings = np.array(data)
                    embeddings_2_plot[subclass] = pca.transform(subclass_embeddings)
                plot_embeddings(embeddings_2_plot)

            if discovered_new_cluster:
                return discovered_new_cluster

        if perfect_clustering:
            print(f"Eps: {eps} resulted in a perfect clustering! No need to process the samples further")
            break

    return discovered_new_cluster


def log_clustering_step(db_cursor, date, eps):
    """
        Log the results of the clustering step into the db.
    """
    query = """INSERT INTO clustering_step(clustering_date,eps,execution_date) VALUES(%s,%s,NOW()) ON CONFLICT (clustering_date,EPS) DO UPDATE SET execution_date = EXCLUDED.execution_date RETURNING id"""
    db_cursor.execute(query, (date, eps))
    res = db_cursor.fetchone()
    return res[0]


def log_conflict(db_cursor, fk_clustering_step, fk_cluster_1, fk_cluster_2):
    """
        Log the results of the clustering step into the db.
    """
    query = """INSERT INTO clustering_conflicts(fk_clustering_step,fk_cluster_1,fk_cluster_2) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING"""
    db_cursor.execute(query, (fk_clustering_step, fk_cluster_1, fk_cluster_2))


def cluster_samples(phishing_dataset, from_date, to_date):
    possible_eps = [0.6, 0.4, 0.2]

    print(f"Processing {to_date}")

    # Run the is_seo_metric on all the new samples
    identify_seo_documents(phishing_dataset, from_date, to_date)

    # Extract embeddings from the screenshot
    prepare_samples_for_clustering(phishing_dataset, from_date, to_date)

    # Run the entire clustering procedure with the specified epsilon
    discovered_new_cluster = cluster_samples_visually(phishing_dataset, from_date, to_date, possible_eps)

    if discovered_new_cluster:
        print("A new cluster has been discovered")


def send_notification(configs, date):
    webhook_url = configs["monitor"]["mattermost"]["webhook"]
    enabled = configs["monitor"]["mattermost"]["enabled"]

    if str(enabled).lower() != 'true':
        return

    # Load the Phishing datalake management class
    phishing_dataset = prepare_phishing_datalake(configs)
    db_cursor = phishing_dataset._db_cursor

    # Get all clusters created today
    query = """Select clusters.id from clusters inner join clustering_step cs on clusters.fk_clustering_step = cs.id
               where clustering_date = %s::date"""
    db_cursor.execute(query, (date,))
    res = list(db_cursor.fetchall())

    if not res:
        return

    text = ":warning: **A new cluster has been created:** \n"
    text += " \n".join([f"New cluster with id:{id}" for id in res])

    payload = {
        "text": text
    }

    if date.date() == (datetime.datetime.today() - datetime.timedelta(days=1)).date():
        # report only when files are not present
        r = requests.post(webhook_url, json=payload)
