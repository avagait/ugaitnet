# Metric functions

__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'March 2020'

import numpy as np
from sklearn.metrics import roc_curve


def mj_eerVerifDist(gt_labels, distances):
    """
    Computes Equal Error Rate
    :param gt_labels: numpy vector with values {0,1}
    :param distances: numpy vector with distances (lower distance should correspond to +1 label)
    :return: EER, threshold
    """

    fpr, tpr, threshold = roc_curve(gt_labels, -distances, pos_label=1)  # LOOK: negative distance
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER, -eer_threshold


# =============================== MAIN ============================
if __name__ == "__main__":
    y = np.array([1,1,1,1,1,0,0,0,0])
    y_pred = np.array([0.01, 0.02, 0.015, 0.08, 0.05, 0.07, 0.2, 0.15, 0.18])

    EER, thr = mj_eerVerifDist(y, y_pred)

    print("EER: {}".format(EER))
    print(thr)