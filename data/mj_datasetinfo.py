__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

import os.path as osp
import numpy as np

class DatasetInfoTUM:
    """ Class to handle TUM-GAID partitions and extra info """

    def __init__(self, filepath, basedir="/home/mjmarin/databases/TUM_GAID/labels/"):
        """

        :param filepath: npy file
        :param basedir:
        """

        self.filepath = filepath
        self.basedir = basedir

        self.dbinfo = []

        self.labels = []
        self.videoids = []
        self.gaits = []
        self.sets = []

        self.labmap = []

        self.preprocess()

    def preprocess(self):

        npy_file = self.filepath
        dbinfo = np.load(npy_file)

        allRecords = ["{:06d}".format(int(r)) for r in dbinfo[:, 0]]
        if dbinfo.shape[1] > 3:
            allSets = ["{:d}".format(int(r)) for r in dbinfo[:, 3]]
        else:
            allSets = []

        # label, videoid, gait, set
        self.labels = dbinfo[:, 1]
        self.videoids = dbinfo[:,2]
        self.gaits = dbinfo[:,3]
        self.sets = dbinfo[:, 4]

        # Further preprocessing
        self.ulabels = np.unique(self.labels)

        # Create mapping for labels
        labmap = {}
        for ix, lab in enumerate(self.ulabels):
            labmap[int(lab)] = ix

        self.labmap = labmap

        self.dbinfo = dbinfo

        rec2lab = {int(dbinfo[ix, 0]): int(dbinfo[ix, 1]) for ix in range(dbinfo.shape[0])}

        self.rec2lab = rec2lab

    def getlabels(self):
        return self.labels

    def getgaits(self):
        return self.gaits

    def getvideoids(self):
        return self.videoids

    def getsets(self):
        return self.sets

    def gaitofsample(self, sample):
        idx = np.where(self.dbinfo[:,0] == sample)[0]

        return int(self.gaits[idx[0]])

# ============================== MAIN ==============================

if __name__ == "__main__":

    dddir = "/home/mjmarin/databases/tumgaid/dd_files"
    dbinfopath = osp.join(dddir, "matimdbtum_gaid_N150_of25_60x60.npy")

    db = DatasetInfoTUM(dbinfopath)

    gaits = db.getgaits()

    print("Done!")
