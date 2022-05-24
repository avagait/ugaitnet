__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

import os.path as osp


class TUMGAIDdb:
    """ Class to handle TUM-GAID partitions and extra info """

    def __init__(self, basedir="/home/mjmarin/databases/TUM_GAID/labels/"):

        self.basedir = basedir

        self.train = self.__loadlist(osp.join(basedir, "tumgaidtrainids.lst"))
        self.val = self.__loadlist(osp.join(basedir, "tumgaidvalids.lst"))
        self.test = self.__loadlist(osp.join(basedir, "tumgaidtestids.lst"))
        self.gender = None
        self.age = None
        self.shoe = None

    def __loadlist(self, filepath : str):
        f = open(filepath, "r")
        if not f:
            print("Error loading data from: "+filepath)
        l_train_ = f.read()
        l_train = l_train_.split()
        f.close()

        return l_train

    def get_train_val_samples_from_dbinfo(self, dbinfofilepath : str):
        """

        :param dbinfofilepath: path to numpy file with db info matrix
        :return: tuple (Idx_train, Idx_val)
        """
        import numpy as np
        D = np.load(dbinfofilepath)

        labels_train = self.train
        labels_val = self.val

        D_labels = D[:, 1]

        Idx_train = []
        for l in labels_train:
            idx = np.where(D_labels == np.float(l))[0]
            Idx_train = Idx_train + idx.tolist()

        Idx_val = []
        for l in labels_val:
            idx = np.where(D_labels == np.float(l))[0]
            Idx_val = Idx_val + idx.tolist()

        return Idx_train, Idx_val

    def __loadgender(self):
        filepath = osp.join(self.basedir, "allgender.txt")
        f = open(filepath, "r")
        if not f:
            print("Error loading data from: "+filepath)
        l_gender_ = f.read()
        l_gender = l_gender_.split()
        f.close()

        self.gender = l_gender

    def get_gender(self, label):
        if not self.gender:
            self.__loadgender()

        return self.gender[label]

    def __loadage(self):
        filepath = osp.join(self.basedir, "allage.txt")
        f = open(filepath, "r")
        if not f:
            print("Error loading data from: "+filepath)
        l_age_ = f.read()
        l_age = l_age_.split()
        f.close()

        self.age = l_age

    def get_age(self, label):
        if not self.age:
            self.__loadage()

        return self.age[label]

    def __loadshoe(self):
        filepath = osp.join(self.basedir, "allshoetype.txt")
        f = open(filepath, "r")
        if not f:
            print("Error loading data from: "+filepath)
        l_shoe_ = f.read()
        l_shoe = l_shoe_.split()
        f.close()

        self.shoe = l_shoe

    def get_shoe(self, label):
        if not self.shoe:
            self.__loadshoe()

        return self.shoe[label]

# ============================== MAIN ==============================

if __name__ == "__main__":

    db = TUMGAIDdb()
    dddir = "/home/mjmarin/databases/tumgaid/dd_files"
    dbinfopath = osp.join(dddir, "matimdbtum_gaid_N150_of25_60x60.npy")

    idx_train, idx_val = db.get_train_val_samples_from_dbinfo(dbinfopath)

    print("Done!")
