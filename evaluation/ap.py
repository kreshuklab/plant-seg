import numpy as np
from skimage import measure


class AveragePrecision:
    def __init__(self, iou_range=(0.5, 1.0), ignore_index=0, min_instance_size=None):
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size

    def __call__(self, input, target):
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray)
        assert input.ndim == target.ndim == 3

        target, target_instances = self._filter_instances(target)

        return self._calculate_average_precision(input, target, target_instances)

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)  # insert 0.0 at beginning of list
        recall.append(1.0)  # insert 1.0 at end of list
        precision.insert(0, 0.0)  # insert 0.0 at beginning of list
        precision.append(0.0)  # insert 0.0 at end of list
        # make the precision(recall) piece-wise constant and monotonically decreasing
        # by iterating backwards starting from the last precision value (0.0)
        # see: https://www.jeremyjordan.me/evaluating-image-segmentation-models/ e.g.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # compute the area under precision recall curve by simple integration of piece-wise constant function
        ap = 0.0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i - 1]) * precision[i]
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)

        # compute precision/recall curve points for various IoU values from a given range
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            # initialize false negatives set
            false_negatives = set(target_instances)
            # initialize false positives set
            false_positives = set(predicted_instances)
            # initialize true positives set
            true_positives = set()

            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(
                    pred_label, predicted, target, min_iou
                )
                if target_label is not None:
                    # update TP, FP and FN
                    if target_label == self.ignore_index:
                        # ignore if 'ignore_index' is the biggest overlapping
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            ROC.append((recall, precision))

        # sort points by recall
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        # return recall and precision values
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        """
        Return ground truth label which overlaps by at least 'min_iou' with a given input label 'p_label'
        or None if such ground truth label does not exist.
        """
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        # retrieve the biggest overlapping label
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        # return target label if IoU greater than 'min_iou'; since we're starting from 0.5 IoU there might be
        # only one target label that fulfill this criterion
        mask_target = target == target_label
        # return target_label if IoU > min_iou
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        """
        Computes intersection over union
        """
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / np.sum(union)

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 'ignore_index'
        :param input: input instance segmentation
        :return: tuple: (instance segmentation with small instances filtered, set of unique labels without the 'ignore_index')
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = input == label
                    input[mask] = self.ignore_index

        labels = set(np.unique(input))
        labels.discard(self.ignore_index)
        return input, labels

    @staticmethod
    def _dt_to_cc(distance_transform, threshold):
        """
        Threshold a given distance_transform and returns connected components.
        :param distance_transform: 3D distance transform matrix
        :param threshold: threshold energy level
        :return: 3D segmentation volume
        """
        boundary = (distance_transform > threshold).astype(np.uint8)
        return measure.label(boundary, background=0, connectivity=1)


def ap(segmentation, ground_truth, ignore_index=0, min_instance_size=10000):
    """
    Returns Average Precision score between the given segmentation and the ground_truth segmentation.

    Args:
        segmentation (ndarray): input segmentation
        ground_truth (ndarray): ground truth segmentation
        ignore_index (int): label to be ignored during AP computation
        min_instance_size (int): minimum instance size used for AP computation; use in order to make the metric robust
            to small instances present either in the input or ground truth segmentation; if 'None' all instances are
            taken into account during AP computation
    Returns:
        Average Precision between segmentation and ground_truth
    """
    ap = AveragePrecision(
        ignore_index=ignore_index, min_instance_size=min_instance_size
    )
    return ap(segmentation, ground_truth)
