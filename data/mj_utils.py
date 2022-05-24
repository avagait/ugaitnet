# (c) MJMJ/2020

import os.path as osp
import numpy as np

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

def mj_splitTrainValGait(datadir, perc=0.1):
    """
    Distributed by subject
    :param datadir: path
    :return: allRecords_tr, allRecords_val, sets_tr, sets_val, labmap
    """
    if datadir.split('/')[-1] == '':
        datadir = datadir[:len(datadir) - 1]
    npy_file = osp.join(datadir + ".npy")
    dbinfo = np.load(npy_file)

    allRecords = ["{:06d}".format(int(r)) for r in dbinfo[:, 0]]
    if dbinfo.shape[1] > 3:
        allSets = ["{:d}".format(int(r)) for r in dbinfo[:, 3]]
    else:
        allSets = []

    allLabels = dbinfo[:, 1]
    ulabels = np.unique(allLabels)
    nulabs = len(ulabels)
    # Create mapping for labels
    labmap = {}
    for ix, lab in enumerate(ulabels):
        labmap[int(lab)] = ix

    ntotal = len(allRecords)
    nval = int(perc * ntotal)
    nval_ps = int(nval/nulabs)

    # Find samples per subject
    subjs = {}
    idx_tr = []
    idx_val = []
    for i, lab in enumerate(ulabels):
        idx = list(np.where(allLabels == lab)[0])
        subjs[i] = idx

        idx_tr = idx_tr + idx[0:len(idx)-nval_ps]
        idx_val = idx_val + idx[len(idx)-nval_ps:len(idx)]

    #idx_tr = slice(ntotal - nval)
    allRecords_tr = [allRecords[ix] for ix in idx_tr ]
    #idx_val = slice(ntotal - nval, ntotal)
    allRecords_val = [allRecords[ix] for ix in idx_val ]

    sets_tr = [allSets[ix] for ix in idx_tr ] #allSets[idx_tr]
    sets_val = [allSets[ix] for ix in idx_val ] #allSets[idx_val]

    return allRecords_tr, allRecords_val, sets_tr, sets_val, labmap


def mj_splitTrainVal(datadir, perc=0.1):
    """ Very basic version """
    if datadir.split('/')[-1] == '':
        datadir = datadir[:len(datadir) - 1]
    npy_file = osp.join(datadir + ".npy")
    dbinfo = np.load(npy_file)

    allRecords = ["{:06d}".format(int(r)) for r in dbinfo[:, 0]]
    if dbinfo.shape[1] > 3:
        allSets = ["{:d}".format(int(r)) for r in dbinfo[:, 3]]
    else:
        allSets = []

    ntotal = len(allRecords)
    nval = int(perc * ntotal)

    idx_tr = slice(ntotal - nval)
    allRecords_tr = allRecords[idx_tr]
    idx_val = slice(ntotal - nval, ntotal)
    allRecords_val = allRecords[idx_val]

    sets_tr = allSets[idx_tr]
    sets_val = allSets[idx_val]

    allLabels = dbinfo[:, 1]
    ulabels = np.unique(allLabels)
    nulabs = len(ulabels)
    # Create mapping for labels
    labmap = {}
    for ix, lab in enumerate(ulabels):
        labmap[int(lab)] = ix

    return allRecords_tr, allRecords_val, sets_tr, sets_val, labmap


def mj_load_groups_file(filepath : str) -> dict:

    f = open(filepath, "rt")
    groups = {}
    for line in f:
        content = line.split(" ")
        s = content[slice(1,len(content)-1)] # Skip \n
        groups[int(content[0])] = [int(si) for si in s]

    return groups
