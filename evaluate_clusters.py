import collections

import joblib
import numpy as np
from sklearn import metrics


def evaluate_clusters(clusters, min_cluster_size=None, max_cluster_size=None,
                      charges=None):
    clusters = clusters.copy()
    if charges is not None:
        clusters = clusters[clusters['precursor_charge'].isin(charges)]

    # Use consecutive cluster labels, skipping the noise points.    
    cluster_map = clusters['cluster'].value_counts(dropna=False)
    if -1 in cluster_map.index:
        cluster_map = cluster_map.drop(index=-1)
    cluster_map = (cluster_map.to_frame().reset_index().reset_index()
                   .rename(columns={'index': 'old', 'level_0': 'new'})
                   .set_index('old')['new'])
    cluster_map = cluster_map.to_dict(collections.defaultdict(lambda: -1))
    clusters['cluster'] = clusters['cluster'].map(cluster_map)

    # Reassign noise points to singleton clusters.
    noise_mask = clusters['cluster'] == -1
    num_clusters = clusters['cluster'].max() + 1
    clusters.loc[noise_mask, 'cluster'] = np.arange(
        num_clusters, num_clusters + noise_mask.sum())
    
    # Only consider clusters with specific minimum (inclusive) and/or
    # maximum (exclusive) size.
    cluster_counts = clusters['cluster'].value_counts(dropna=False)
    if min_cluster_size is not None:
        clusters.loc[clusters['cluster'].isin(cluster_counts[
            cluster_counts < min_cluster_size].index), 'cluster'] = -1
    if max_cluster_size is not None:
        clusters.loc[clusters['cluster'].isin(cluster_counts[
            cluster_counts >= max_cluster_size].index), 'cluster'] = -1

    # Compute cluster evaluation measures.
    noise_mask = clusters['cluster'] == -1
    num_noise = noise_mask.sum()
    num_clustered = len(clusters) - num_noise
    prop_clustered = (len(clusters) - num_noise) / len(clusters)

    clusters_ident = clusters.dropna(subset=['sequence'])
    clusters_ident_non_noise = (clusters[~noise_mask]
                                .dropna(subset=['sequence']))

    # The number of incorrectly clustered spectra is the number of PSMs that
    # differ from the majority PSM. Unidentified spectra are not considered.
    prop_clustered_incorrect = sum(joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_count_majority_label_mismatch)(clust['sequence'])
        for _, clust in clusters[~noise_mask].groupby('cluster')))
    prop_clustered_incorrect /= len(clusters_ident_non_noise)

    # Homogeneity measures whether clusters contain only identical PSMs.
    # This is only evaluated on non-noise points, because the noise cluster
    # is highly non-homogeneous by definition.
    homogeneity = metrics.homogeneity_score(
        clusters_ident_non_noise['sequence'],
        clusters_ident_non_noise['cluster'])
    
    # Completeness measures whether identical PSMs are assigned to the same
    # cluster.
    # This is evaluated on all PSMs, including those clustered as noise.
    completeness = metrics.completeness_score(
        clusters_ident['sequence'], clusters_ident['cluster'])

    return (num_clustered, num_noise,
            prop_clustered, prop_clustered_incorrect,
            homogeneity, completeness) 


def _count_majority_label_mismatch(labels):
    labels_assigned = labels.dropna()
    if len(labels_assigned) <= 1:
        return 0
    else:
        return len(labels_assigned) - labels_assigned.value_counts().iat[0]
