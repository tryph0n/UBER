"""Clustering utilities for hot-zone detection."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from .config import RANDOM_STATE

import logging

logger = logging.getLogger(__name__)


def evaluate_kmeans(X: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    """Run KMeans for each k value and return a metrics DataFrame.

    Metrics: silhouette score, Calinski-Harabasz index, inertia, avg cluster size.
    """
    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(30000, len(X)), random_state=RANDOM_STATE)
        ch = calinski_harabasz_score(X, labels)
        results.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "inertia": km.inertia_,
            "avg_cluster_size": len(X) / k,
        })
        logger.debug(f"k={k:>3d}: silhouette={sil:.4f}, CH={ch:,.0f}, inertia={km.inertia_:,.0f}")
    return pd.DataFrame(results)


def evaluate_dbscan(
    X: np.ndarray, eps_values: list[float], min_samples_values: list[int]
) -> pd.DataFrame:
    """Run DBSCAN grid search and return a metrics DataFrame.

    Skips configurations with fewer than 2 clusters or too few non-noise points.
    """
    results = []
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(labels) * 100

            if n_clusters < 2:
                continue

            non_noise_mask = labels != -1
            non_noise_labels = labels[non_noise_mask]
            X_non_noise = X[non_noise_mask]

            if len(set(non_noise_labels)) < 2 or len(non_noise_labels) < 100:
                continue

            cluster_counts = pd.Series(non_noise_labels).value_counts()
            largest_pct = cluster_counts.iloc[0] / len(non_noise_labels) * 100

            sil = silhouette_score(
                X_non_noise, non_noise_labels,
                sample_size=min(30_000, len(X_non_noise)),
                random_state=RANDOM_STATE,
            )
            ch = calinski_harabasz_score(X_non_noise, non_noise_labels)

            results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "noise_pct": noise_pct,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "largest_cluster_pct": largest_pct,
                "avg_cluster_size": len(non_noise_labels) / n_clusters,
            })
            logger.debug(
                f"eps={eps:.4f}, min_samples={ms:>3d}: "
                f"{n_clusters} clusters, noise={noise_pct:.1f}%, "
                f"sil={sil:.4f}, largest={largest_pct:.1f}%"
            )

    if not results:
        logger.warning("No valid DBSCAN configurations found.")
        return pd.DataFrame()
    return pd.DataFrame(results)


def fit_final_kmeans(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """Fit final KMeans model on full data.

    Returns (labels, cluster_centers, fitted_model).
    """
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    logger.info(f"KMeans fitted: k={k}, inertia={km.inertia_:,.0f}")
    cluster_sizes = pd.Series(labels).value_counts()
    logger.info(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
                f"mean={cluster_sizes.mean():.0f}, median={cluster_sizes.median():.0f}")
    return labels, km.cluster_centers_, km
