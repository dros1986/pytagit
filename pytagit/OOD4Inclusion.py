import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import pairwise_distances
from tqdm.rich import tqdm

class OOD4Inclusion:
    def __init__(self, distance_function='cosine', use_lad=True, label_error_detection=False):
        self.clean_distribution = OrderedDict()
        self.distance_function = distance_function
        self.use_lad = use_lad
        self.label_error_detection = label_error_detection

    def set_clean_distribution(self, class_id, feature_vectors):
        """
        Store a clean distribution of feature vectors for a given class.
        :param class_id: Unique identifier for the class.
        :param feature_vectors: A 2D tensor representing clean feature vectors.
        """
        self.clean_distribution[class_id] = feature_vectors

    def evaluate_new_samples(self, class_id, feature_vectors, threshold=None):
        """
        Evaluate whether new feature vectors belong to the clean distribution.
        :param class_id: Unique identifier for the class.
        :param feature_vectors: A 2D tensor representing new feature vectors.
        :param threshold: Threshold above which a sample is considered an outlier.
        :return: Boolean tensor indicating whether each sample is an inlier (True) or outlier (False), and their scores.
        """
        if class_id not in self.clean_distribution:
            raise ValueError(f"Class ID {class_id} not found in clean distribution. Please set it first.")

        clean_features = self.clean_distribution[class_id]
        distances = pairwise_distances(feature_vectors.numpy(), clean_features.numpy(), metric=self.distance_function)
        irrelevant_scores = np.mean(distances, axis=1) if self.use_lad else np.zeros(distances.shape[0])

        if self.label_error_detection:
            intra_class_dists = np.min(distances, axis=1)
            inter_class_dists = np.min(pairwise_distances(feature_vectors.numpy(), metric=self.distance_function), axis=1)
            label_error_scores = inter_class_dists - intra_class_dists
        else:
            label_error_scores = np.zeros(distances.shape[0])

        outlier_scores = irrelevant_scores + label_error_scores
        outlier_scores = torch.from_numpy(outlier_scores).float()
        # if threshold is not specified, return the outlier scores
        if threshold is None:
            return None, outlier_scores
        # otherwise, return a boolean mask indicating inliers and the scores
        inlier_mask = outlier_scores <= threshold  # True for inliers, False for outliers
        return inlier_mask, outlier_scores
