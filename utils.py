import numpy as np
import pandas as pd
from scipy.stats import skew


def cal_h_acc(record, general=False):
    tr = record['test_seen_g']['acc'] if general else record['test_seen']['acc']
    ts = record['test_unseen_g']['acc'] if general else record['test_unseen']['acc']

    return 2 * tr * ts / (tr + ts)


def cal_acc(metric, general=False):

    labels = metric['correct_g'] if general else metric['correct']
    totals = metric['total_g'] if general else metric['total']

    labels = np.asarray(labels)
    totals = np.asarray(totals)

    tp = np.logical_and(labels, totals).sum(axis=0).reshape(-1)
    totals = totals.sum(axis=0).reshape(-1)

    classes = [np.nan_to_num(l / t) for l, t in zip(tp, totals)]
    acc = sum(classes) / len(np.asarray(metric['total']).sum(axis=0).reshape(-1))

    return classes, acc


def skewness(metric, general=False):

    labels = metric['correct_g'] if general else metric['correct']
    totals = metric['total_g'] if general else metric['total']

    # idx, count = np.unique(np.asarray(labels), return_counts=True)
    # labels = dict(zip(idx, count))
    # labels = pd.DataFrame.from_dict(labels, orient='index')

    labels = pd.DataFrame(np.asarray(labels))

    return labels.skew().values[0]
