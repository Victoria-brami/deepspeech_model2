import numpy as np
import sklearn.metrics
import torch.nn as nn
import torch
import sklearn
from pytorch_lightning.metrics import Metric


# Core calculation of label precisions for one test sample.

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


# All-in-one calculation of per-class lwlrap.

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# Calculate the overall lwlrap using sklearn.metrics function.

def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


# Accumulator object version.

class lwlrap_accumulator(object):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self):
        self.num_classes = 0
        self.total_num_samples = 0

    def accumulate_samples(self, batch_truth, batch_scores):
        """Cumulate a new batch of samples into the metric.

        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes,
                                                        dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = (
                _one_sample_positive_class_precisions(scores, truth))
            self._per_class_cumulative_precision[pos_class_indices] += (
                precision_at_hits)
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return (self._per_class_cumulative_precision /
                np.maximum(1, self._per_class_cumulative_count))

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return (self._per_class_cumulative_count /
                float(np.sum(self._per_class_cumulative_count)))

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())


# torch nn version

class LwLRAPLossbis(nn.Module):

    def _one_sample_positive_class_precisions(self, scores, truth):
        """Calculate precisions for each true class for a single sample.

        Args:
          scores: np.array of (num_classes,) giving the individual classifier scores.
          truth: np.array of (num_classes,) bools indicating which classes are true.

        Returns:
          pos_class_indices: np.array of indices of the true classes for this sample.
          pos_class_precisions: np.array of precisions corresponding to each of those
            classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = torch.flatnonzero(truth > 0)
        # Only calculate precisions if there are some true classes.
        if not len(pos_class_indices):
            return pos_class_indices, torch.zeros(0)
        # Retrieval list of classes for this sample.
        retrieved_classes = torch.argsort(scores)[::-1]
        # class_rankings[top_scoring_class_index] == 0 etc.
        class_rankings = torch.zeros(num_classes, dtype=torch.int)
        class_rankings[retrieved_classes] = range(num_classes)
        # Which of these is a true label?
        retrieved_class_true = torch.zeros(num_classes, dtype=torch.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        # Num hits for every truncated retrieval list.
        retrieved_cumulative_hits = torch.cumsum(retrieved_class_true)
        # Precision of retrieval list truncated at each hit, in order of pos_labels.
        precision_at_hits = (
                retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
                (1 + class_rankings[pos_class_indices].astype(torch.float)))
        return pos_class_indices, precision_at_hits

    def calculate_per_class_lwlrap(self, truth, scores):
        """Calculate label-weighted label-ranking average precision.

        Arguments:
          truth: np.array of (num_samples, num_classes) giving boolean ground-truth
            of presence of that class in that sample.
          scores: np.array of (num_samples, num_classes) giving the classifier-under-
            test's real-valued score for each class for each sample.

        Returns:
          per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
            class.
          weight_per_class: np.array of (num_classes,) giving the prior of each
            class within the truth labels.  Then the overall unbalanced lwlrap is
            simply np.sum(per_class_lwlrap * weight_per_class)
        """
        assert truth.shape == scores.shape
        num_samples, num_classes = scores.shape
        # Space to store a distinct precision value for each class on each sample.
        # Only the classes that are true for each sample will be filled in.
        precisions_for_samples_by_classes = torch.zeros((num_samples, num_classes))
        for sample_num in range(num_samples):
            pos_class_indices, precision_at_hits = (
                self._one_sample_positive_class_precisions(scores[sample_num, :],
                                                           truth[sample_num, :]))
            precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
                precision_at_hits)
        labels_per_class = torch.sum(truth > 0, axis=0)
        weight_per_class = labels_per_class / float(torch.sum(labels_per_class))
        # Form average of each column, i.e. all the precisions assigned to labels in
        # a particular class.
        per_class_lwlrap = (torch.sum(precisions_for_samples_by_classes, axis=0) /
                            torch.max(1, labels_per_class).values)
        # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
        #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
        #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
        #                = np.sum(per_class_lwlrap * weight_per_class)
        return per_class_lwlrap, weight_per_class

    def forward(self, x, labels):
        return None


class LwLRAPLoss(nn.Module):

    def forward(self, preds, labels):
        """Calculate the overall lwlrap using sklearn.metrics.lrap."""
        # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
        """
        sample_weight = np.sum((truth > 0).detach().numpy(), axis=1)
        nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

        # nonzero_weight_sample_indices = torch.nonzero((sample_weight > 0).view(-1), as_tuple=True)[0]
        overall_lwlrap = torch.tensor(sklearn.metrics.label_ranking_average_precision_score(
            truth[nonzero_weight_sample_indices, :] > 0,
            scores[nonzero_weight_sample_indices, :],
            sample_weight=sample_weight[nonzero_weight_sample_indices]))
        return overall_lwlrap.item()
        """
        # print('devices: preds {} labels {}'.format(preds.get_device(), labels.get_device()))
        # Ranks of the predictions
        ranked_classes = torch.argsort(preds, dim=-1, descending=True)
        # i, j corresponds to rank of prediction in row i
        class_ranks = torch.zeros_like(ranked_classes)
        for i in range(ranked_classes.size(0)):
            for j in range(ranked_classes.size(1)):
                class_ranks[i, ranked_classes[i][j]] = j + 1
        # Mask out to only use the ranks of relevant GT labels
        ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
        # All the GT ranks are in front now
        sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
        # Number of GT labels per instance
        num_labels = labels.sum(-1)
        pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))]), device='cuda:0').unsqueeze(0)
        score_matrix = pos_matrix / sorted_ground_truth_ranks
        score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
        scores = score_matrix * score_mask_matrix
        score = scores.sum() / labels.sum()
        return score


# pytorch lightning computation: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418

class LWLRAP(Metric):
    def __init__(self, precision=16, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scores_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("labels_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.precision = precision

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        scores, labels = self._batch_compute(preds, target)
        self.scores_sum += scores
        self.labels_sum += labels

    def compute(self):
        res = self.scores_sum.float() / self.labels_sum.float()
        return 1 - res

    def __call__(self, preds, labels):
        scores, labels = self._batch_compute(preds, labels)
        self.scores_sum += scores
        self.labels_sum += labels
        res = scores.float() / labels.float()
        return 1 - res

    # label-level average
    # Assume float preds [BxC], labels [BxC] of 0 or 1
    def _batch_compute(self, preds, labels):
        device = preds.device
        # Ranks of the predictions
        ranked_classes = torch.argsort(preds, dim=-1, descending=True)
        # i, j corresponds to rank of prediction in row i
        class_ranks = torch.zeros_like(ranked_classes)
        for i in range(ranked_classes.size(0)):
            for j in range(ranked_classes.size(1)):
                class_ranks[i, ranked_classes[i][j]] = j + 1
        # Mask out to only use the ranks of relevant GT labels
        ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
        # All the GT ranks are in front now
        sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
        # Number of GT labels per instance
        num_labels = labels.sum(-1)
        if self.precision == 16:
            pos_matrix = torch.tensor(np.array([i + 1 for i in range(labels.size(-1))]), dtype=torch.float16, device=device, requires_grad=True).unsqueeze(0)
        elif self.precision == 32:
            pos_matrix = torch.tensor(np.array([i + 1 for i in range(labels.size(-1))]), dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        elif self.precision == 64:
            pos_matrix = torch.tensor(np.array([i + 1 for i in range(labels.size(-1))]), dtype=torch.float64, device=device, requires_grad=True).unsqueeze(0)
        score_matrix = pos_matrix / sorted_ground_truth_ranks
        score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
        scores = score_matrix * score_mask_matrix

        return scores.sum(), labels.sum()


# Kaggle proposition
def LWLRAPKaggle(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score



if __name__ == '__main__':
    input = torch.tensor([[0.1, 0.4, 0.3, 0.8], [0.6, 0.1, 0.9, 0.3], [0.5, 0.0, 0.9, 0.4]])
    target = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]])
    loss = LwLRAPLoss()
    print(loss(target, input))
    print(LWLRAPKaggle(input, target))

    new_loss = LWLRAP()
    scores, labels = new_loss._batch_compute(input, target)
    print(scores/labels)
