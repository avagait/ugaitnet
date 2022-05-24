'''
Based on the following example:
   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
(c) MJMJ/2020
'''

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

import numpy as np
import deepdish as dd
import tensorflow.keras as keras
import tensorflow as tf

np.random.seed(232323)
import random
import copy

from data.mj_tfdata import *
import os.path as osp

import data.mj_augmentation as DA
import gc


class DataGeneratorGaitMMUWYH(keras.utils.Sequence):
    """
    A class used to generate data for training/testing CNN gait models

    Attributes
    ----------
    dim : tuple
        dimensions of the input data
    n_classes : int
        number of classes
    ...
    """

    def __init__(self, allSamples, targets=[], batch_size=32, dim=[(50, 60, 60), (25, 60, 60)], n_classes=150,
                 shuffle=True, augmentation=True,
                 datadir=["/home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N150_of25_60x60/",
                          "/home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N150_gray25_60x60/"],
                 sess=None, labmap=[], gait=[], ntype=1,
                 isTest=False, augmentation_x=1, expand_level=2, balanced_classes=True,
                 isTriplet=False, use3D=False, isDebug=False, softlabel=False,
                 camera=[], nmods=2, use_weights=False,
                 meanSample=0.0, aux_losses=False, triplet_all_fc=False, nfcs=0, keep_data=False, gaitset=False, repetition=4):
        'Initialization'
        self.dim = dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.meanSample = meanSample
        self.shuffle = shuffle

        self.isTriplet = isTriplet
        self.isTest = isTest
        self.isDebug = isDebug
        self.use3D = use3D
        self.softlabel = softlabel
        self.aux_losses = aux_losses
        self.triplet_all_fc = triplet_all_fc
        self.nfcs = nfcs
        self.nmods = nmods  # Number of modalities
        self.cameras = None
        if nmods > 1:
            self.withof = dim[0][0] == 50
        else:
            self.withof = dim[0] == 50

        if n_classes > 0:
            self.multitask = True
        else:
            self.multitask = False
        self.use_weights = use_weights
        self.balanced = balanced_classes

        self.allSamples = allSamples
        self.targets = targets
        self.labmap = labmap
        self.gait = gait
        self.camera = camera

        self.datadir = datadir
        self.ntype = ntype

        self.__remove_empty_files()

        self.list_IDs = np.arange(len(allSamples))

        self.tfconfig = tf.compat.v1.ConfigProto()
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.compat.v1.get_default_session()  # tf.compat.v1.Session(config=self.tfconfig)
        self.graph = tf.compat.v1.get_default_graph()

        self.__prepare_labels()
        self.__prepare_gaits()
        self.on_epoch_end()

        self.expand_level = expand_level
        self.noise = 0.000000001

        self.augmentation = augmentation_x > 0
        self.img_gen = [DA.mj_transgenerator(isof=self.withof), DA.mj_transgenerator(isof=False)]

        self.keep_data = keep_data
        self.data = {}


        self.gaitset = gaitset
        self.repetition = repetition

    def __len__(self):
        'Number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(len(self.indexes) / self.batch_size))


    def __remove_empty_files(self):
        allSamples_ = []
        gait_ = []
        print(self.allSamples, flush=True)
        for i in range(len(self.allSamples)):
            if self.allSamples[i][0][0] != -1:
                if osp.exists(osp.join(self.datadir[0], self.allSamples[i][0][0])):
                    data_i = dd.io.load(osp.join(self.datadir[0], self.allSamples[i][0][0]))
                    if len(data_i["data"]) > 0:
                        if self.allSamples[i][0][1] != -1:
                            if osp.exists(osp.join(self.datadir[1], self.allSamples[i][0][1])):
                                data_i_2 = dd.io.load(osp.join(self.datadir[1], self.allSamples[i][0][1]))
                                if len(data_i_2["data"]) > 0:
                                    allSamples_.append(self.allSamples[i])
                                    gait_.append(self.gait[i])
                        else:
                            allSamples_.append(self.allSamples[i])
                            gait_.append(self.gait[i])
            elif self.allSamples[i][0][1] != -1:
                if osp.exists(osp.join(self.datadir[1], self.allSamples[i][0][1])):
                    data_i_2 = dd.io.load(osp.join(self.datadir[1], self.allSamples[i][0][1]))
                    if len(data_i_2["data"]) > 0:
                        allSamples_.append(self.allSamples[i])
                        gait_.append(self.gait[i]) 

        self.allSamples = allSamples_
        self.gait = gait_


    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        list_IDs_temp = []
        this_lab_used = 0
        this_lab_used_rep = 0
        while len(list_IDs_temp) < self.batch_size:
            for g in self.ugait:
                # Esto solucionaria el bug de las 33 muestras en un batch de 32
                if len(list_IDs_temp) == self.batch_size:
                    continue
                lab2ptr = self.gait2ptr[g]
                lab_ix = self.ulabs[self.nextlab_idx]
                ix = lab2ptr[lab_ix]
                lab2rec = self.gait2idx[g]
                if len(lab2rec[lab_ix]) > 0:
                    rec = lab2rec[lab_ix][ix]
                    list_IDs_temp.append(rec)

                this_lab_used += 1

                lab2ptr[lab_ix] += 1
                if lab2ptr[lab_ix] >= len(lab2rec[lab_ix]):
                    lab2ptr[lab_ix] = 0

                if this_lab_used >= 2:
                    this_lab_used = 0
                    this_lab_used_rep += 1
                    if this_lab_used_rep == self.repetition:
                        self.nextlab_idx += 1
                        this_lab_used_rep = 0
                        if self.nextlab_idx >= len(self.ulabs):
                            self.nextlab_idx = 0
        
        # Generate data
        if self.use_weights:
            X, y, w = self.__data_generation(list_IDs_temp)
            return X, y, w
        else:
            X, y = self.__data_generation(list_IDs_temp, expand=self.expand_level)
            if self.triplet_all_fc:
                y_ = y[0]
                for i_ll in range(self.nfcs-1):
                    y.insert(i_ll+1, y_)
            if self.aux_losses:
                y_ = y[-1]
                for i_ll in range(self.nmods):
                    y.append(y_)
            return X, y


    def __getitemwithinfo__(self, index, expand, withcam=False):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.use_weights:
            X, y, w = self.__data_generation(list_IDs_temp, expand=expand)
        else:
            X, y = self.__data_generation(list_IDs_temp, expand=expand)

        info = [self.allSamples[ID] for ID in list_IDs_temp]

        if withcam:
            cam = [self.camera[ID] for ID in list_IDs_temp]
            return X, y, info, cam
        else:
            return X, y, info

    def __prepare_gaits(self):
        self.ugait = np.unique(self.gait)

        self.gait2idx = {g: [] for g in self.ugait}

        for g in self.ugait:
            idx_g = np.where(self.gait == g)[0]

            # Separate into subjects
            g_labs = {ix: 0 for ix in self.ulabs}
            subset = [self.targets[j] for j in idx_g]
            for lab in self.ulabs:
                idx_g_and_lab = np.where(subset == lab)[0]
                g_labs[lab] = [idx_g[j] for j in idx_g_and_lab]

            self.gait2idx[g] = g_labs

    def __prepare_labels(self):
        self.targets = [int(ix[1]) for ix in self.allSamples]

        self.ulabs = np.unique(self.targets)

        self.lab2rec = {ix: 0 for ix in self.ulabs}

        targets = np.array(self.targets)
        for t in self.ulabs:
            idx = np.where(targets == t)[0]
            self.lab2rec[t] = idx.tolist()


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        self.gait2ptr = {g: {} for g in self.ugait}
        for g in self.ugait:
            ptrs = {l: 0 for l in self.ulabs}
            self.gait2ptr[g] = ptrs

        # self.lab2ptr = {l: 0 for l in self.ulabs }

        # self.nextlab_idx = {g: 0 for g in self.ugait}
        self.nextlab_idx = 0

        if (not self.isTest) and self.shuffle:
            np.random.shuffle(self.ulabs)
            for k in self.lab2rec.keys():
                np.random.shuffle(self.lab2rec[k])
                  
        tf.keras.backend.clear_session()
        gc.collect()

    def __load_dd(self, filepath: str, ntype=1, clip_max=0, clip_min=0):
        """
        Loads a dd file with gait samples
        :param filepath: full path to h5 file (deep dish compatible)
        :return: numpy array with data
        """
        if filepath is None:
            return None

        if not osp.exists(filepath):
            return None

        if self.keep_data:
            if filepath in self.data:
                sample = self.data[filepath]
            else:
                sample = dd.io.load(filepath)
                self.data[filepath] = copy.deepcopy(sample)
        else:
            sample = dd.io.load(filepath)

        if len(sample["data"]) == 0:
            return None

        if sample["compressFactor"] > 1:
            x = np.float32(sample["data"])
            #import pdb; pdb.set_trace()
            if clip_max > 0:
                x[np.abs(x) > clip_max] = 1e-8
            if clip_min > 0:
                x[np.abs(x) < clip_min] = 1e-8
            x = x / sample["compressFactor"]
            if ntype == 2:
                x = x * 0.1   # DEVELOP!
        else:
            if "silhouette" in filepath:
                x = (np.float32(sample["data"]) / 255.0)
            else:
                x = (np.float32(sample["data"]) / 255.0) - 0.5

        if x.shape[-1] == 1:
            x = np.squeeze(x)
            x = np.moveaxis(x, 0, -1)
        elif self.gaitset and (x.shape[0] == 25 or x.shape[0] == 50):
            x = np.moveaxis(x, 0, -1)

        if ntype == 2:
            try:
                x = np.moveaxis(x, 2, 0)
            except:
                print("Error with file: "+filepath)
                import pdb; pdb.set_trace()

        return x

    def __gen_batch(self, list_IDs_temp, expand=1):
        # Initialization
        # if self.use3D:
        dim0 = len(list_IDs_temp) * expand
        x = [np.empty((dim0, *self.dim[0])), np.empty((dim0, 1)),
             np.empty((dim0, *self.dim[1])), np.empty((dim0, 1))]
        # else:
        #     x = [np.empty((self.batch_size*expand, *self.dim[0])), np.empty((self.batch_size*expand,1)),
        #          np.empty((self.batch_size*expand, *self.dim[1])), np.empty((self.batch_size*expand,1))]

        if self.multitask:
            y = [np.empty((dim0, 1)),
                 np.empty((dim0, self.n_classes))]
        else:
            y = np.empty((dim0, 1))

        # Generate data
        if self.isDebug:
            # Fake
            x = []
            x.append(np.random.uniform(-1, 1, (self.batch_size, self.dim[0][0], 60, 60)))
            x.append(np.random.uniform(0, 2, (self.batch_size, 1)))
            x.append(np.random.uniform(-1, 1, (self.batch_size, 25, 60, 60)))
            x.append(np.random.uniform(0, 2, (self.batch_size, 1)))
            if self.multitask:
                y = [np.floor(np.random.uniform(0, 4, (self.batch_size, 1))),
                     np.floor(np.random.uniform(0, 2, (self.batch_size, self.n_classes)))]
                # if self.softlabel:
                #     y[1] = mj_softlabel(y[0], self.n_classes)
            else:
                y = np.floor(np.random.uniform(0, 32, (self.batch_size, 1)))
        else:
            for i, ID in enumerate(list_IDs_temp):

                pair = self.allSamples[ID]
                pair1 = pair[0]
                mod1 = pair1[0]
                mod2 = pair1[1]

                label = pair[1]

                if self.ntype == 1:
                    for set1 in range(3):
                        filepath1 = osp.join(self.datadir[0], "{:d}_sample{:06d}.h5".format(set1, mod1))
                        if osp.exists(filepath1):
                            break

                    for set2 in range(3):
                        filepath2 = osp.join(self.datadir[1], "{:d}_sample{:06d}.h5".format(set2, mod2))
                        if osp.exists(filepath2):
                            break
                else:
                    if mod1 != -1:
                        filepath1 = osp.join(self.datadir[0], mod1)
                    else:
                        filepath1 = None
                    if mod2 != -1:
                        filepath2 = osp.join(self.datadir[1], mod2)
                    else:
                        filepath2 = None

                # Data augmentation?
                if self.augmentation and np.random.randint(4) > 0:
                    trans = [self.img_gen[0].get_random_transform((self.dim[0][1], self.dim[0][2])),
                             self.img_gen[1].get_random_transform((self.dim[1][1], self.dim[1][2]))]
                    trans[1]["tx"] = trans[0]["tx"]
                    trans[1]["ty"] = trans[0]["ty"]
                    flip = np.random.randint(2) == 1
                else:
                    trans = None
                    flip = False

                if self.withof and self.augmentation and np.random.randint(2) == 1:
                    clip_max = 2300
                    clip_min = 50
                else:
                    clip_max = 0
                    clip_min = 0

                # Store sample
                x_tmp = self.__load_dd(filepath1, self.ntype, clip_max=clip_max, clip_min=clip_min)

                if x_tmp is None:
                    x[0][i * expand,] = self.noise
                    x[1][i * expand,] = 0.0
                else:
                    if trans is not None:
                        x_tmp = DA.mj_transformsequence(x_tmp, self.img_gen[0], trans[0])
                        #print("Mod-1: {}".format([x_tmp.min(), x_tmp.max()]))  # DEVELOP
                        if flip:
                            x_tmp = DA.mj_mirrorsequence(x_tmp, self.withof, True)

                    if self.dim[0][0] != 50 and self.use3D and x_tmp is not None:  # It is not OF
                        x_tmp = np.expand_dims(x_tmp, axis=3)

                    if self.gaitset:
                        if x_tmp.shape[0] == 50:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 2), dtype=x_tmp.dtype)
                            x_new[:, :, :, 0] = x_tmp[::2, :, :]
                            x_new[:, :, :, 1] = x_tmp[1::2, :, :]
                        else:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 1), dtype=x_tmp.dtype)
                            x_new[:, :, :, 0] = x_tmp
                        x_tmp = x_new

                    x[0][i * expand,] = x_tmp
                    x[1][i * expand,] = 1.0

                if mod2 == -1:
                    x[3][i * expand,] = 0.0
                    x[2][i * expand,] = self.noise
                else:
                    M_ = self.__load_dd(filepath2, self.ntype)
                    
                    if M_ is not None:
                        if trans is not None:
                            M_ = DA.mj_transformsequence(M_, self.img_gen[1], trans[1])
                            #print("Mod-2: {}".format([M_.min(), M_.max()]))  # DEVELOP
                            if flip:
                                M_ = DA.mj_mirrorsequence(M_, isof=False)

                        if self.use3D and M_ is not None:
                            M_ = np.expand_dims(M_, axis=3)

                        if self.gaitset:
                            if M_.shape[0] == 50:
                                x_new = np.zeros((25, M_.shape[1], M_.shape[2], 2), dtype=M_.dtype)
                                x_new[:, :, :, 0] = M_[::2, :, :]
                                x_new[:, :, :, 1] = M_[1::2, :, :]
                            else:
                                x_new = np.zeros((25, M_.shape[1], M_.shape[2], 1), dtype=M_.dtype)
                                x_new[:, :, :, 0] = M_
                            M_ = x_new


                        # import pdb; pdb.set_trace()
                        x[2][i * expand, :, :, :, ] = M_
                        x[3][i * expand,] = 1.0
                    else:
                        x[3][i * expand,] = 0.0
                        x[2][i * expand,] = self.noise

                lb = -1
                if self.labmap:
                    lb = self.labmap[int(label)]
                else:
                    lb = label

                if self.multitask:
                    y[0][i * expand] = lb
                else:
                    y[i * expand] = lb

                # Same pairs but disabling modalities
                if expand > 1:
                    # x[0][1 + i * expand,] = x[0][i * expand,]
                    # x[2][1 + i * expand,] = x[2][i * expand,]
                    if self.multitask:
                        y[0][1 + i * expand] = lb
                    else:
                        y[1 + i * expand] = lb

                    # Randomly choose which modality to disable
                    choice1 = random.randrange(0, 2, 1)
                    if choice1 == 0:
                        x[1][1 + i * expand,] = 0.0
                        x[0][1 + i * expand,] = self.noise

                        x[2][1 + i * expand,] = np.copy(x[2][i * expand,])
                        x[3][1 + i * expand,] = 1.0
                    else:
                        x[1][1 + i * expand,] = 1.0
                        x[0][1 + i * expand,] = np.copy(x[0][i * expand,])

                        x[3][1 + i * expand,] = 0.0
                        x[2][1 + i * expand,] = self.noise

                    # Check if more samples have to be added
                    if expand > 2:
                        if self.multitask:
                            y[0][2 + i * expand] = lb
                        else:
                            y[2 + i * expand] = lb

                        # Use the other choice
                        choice1 = 1 - choice1
                        if choice1 == 0:
                            x[1][2 + i * expand,] = 0.0
                            x[0][2 + i * expand,] = self.noise

                            x[2][2 + i * expand,] = np.copy(x[2][i * expand,])
                            x[3][2 + i * expand,] = 1.0
                        else:
                            x[1][2 + i * expand,] = 1.0
                            x[0][2 + i * expand,] = np.copy(x[0][i * expand,])

                            x[3][2 + i * expand,] = 0.0
                            x[2][2 + i * expand,] = self.noise

        if self.multitask:
            # if self.softlabel:
            #     y[1] = mj_softlabel(y[0], self.n_classes)
            # else:
            y[1] = keras.utils.to_categorical(y[0], num_classes=self.n_classes)

        if self.use_weights:
            if self.multitask:
                w = [np.ones(shape=(1,len(x[0]))), np.ones(shape=(1,len(x[0])))] # np.ones(shape=(1, len(x[0]))) #[None, None] #(np.ones(shape=(1, len(x[0]))), np.ones(shape=(1, len(x[0]))))
                #w = np.ones(shape=(len(x[0]),2)) #[ np.ones(shape=(len(x[0]), 1)), np.ones(shape=(len(x[0]), 1))]
                #w = [None, None]
            else:
                w = None
            return x, y, w
        else:
            return x, y


    def __gen_batchSingle(self, list_IDs_temp):
        # Initialization
        # if self.use3D:
        dim0 = len(list_IDs_temp)
        x = np.empty((dim0, *self.dim))

        if self.multitask:
            y = [np.empty((dim0, 1)),
                 np.empty((dim0, self.n_classes))]
        else:
            y = np.empty((dim0, 1))

        # Generate data
        if self.isDebug:
            # Fake
            x = np.random.uniform(-1, 1, (self.batch_size, self.dim[0], 60, 60))
            if self.multitask:
                y = [np.floor(np.random.uniform(0, 4, (self.batch_size, 1))),
                     np.floor(np.random.uniform(0, 2, (self.batch_size, self.n_classes)))]
                # if self.softlabel:
                #     y[1] = mj_softlabel(y[0], self.n_classes)
            else:
                y = np.floor(np.random.uniform(0, 32, (self.batch_size, 1)))
        else:
            for i, ID in enumerate(list_IDs_temp):

                pair = self.allSamples[ID]
                pair1 = pair[0]
                mod1 = pair1[0]
                mod2 = pair1[1]

                label = pair[1]

                if self.ntype == 1:
                    for set1 in range(3):
                        filepath1 = osp.join(self.datadir[0], "{:d}_sample{:06d}.h5".format(set1, mod1))
                        if osp.exists(filepath1):
                            break
                else:
                    if mod1 != -1:
                        filepath1 = osp.join(self.datadir[0], mod1)
                    else:
                        filepath1 = None

                # Data augmentation?
                if self.augmentation and np.random.randint(4) > 0:
                    trans = [self.img_gen[0].get_random_transform((self.dim[1], self.dim[2])),
                             self.img_gen[1].get_random_transform((self.dim[1], self.dim[2]))]
                    trans[1]["tx"] = trans[0]["tx"]
                    trans[1]["ty"] = trans[0]["ty"]
                    flip = np.random.randint(2) == 1
                else:
                    trans = None
                    flip = False

                if self.withof and self.augmentation and np.random.randint(2) == 1:
                    clip_max = 2300
                    clip_min = 50
                else:
                    clip_max = 0
                    clip_min = 0

                # Store sample
                # import pdb; pdb.set_trace()
                x_tmp = self.__load_dd(filepath1, self.ntype, clip_max=clip_max, clip_min=clip_min)

                if x_tmp is None:
                    print("WARN: this shouldn't happen!")
                    import pdb; pdb.set_trace()

                else:
                    if trans is not None:
                        x_tmp = DA.mj_transformsequence(x_tmp, self.img_gen[0], trans[0])
                        #print("Mod-1: {}".format([x_tmp.min(), x_tmp.max()]))  # DEVELOP
                        if flip:
                            x_tmp = DA.mj_mirrorsequence(x_tmp, self.withof, True)

                    if self.dim[0] != 50 and self.use3D and x_tmp is not None:  # It is not OF
                        x_tmp = np.expand_dims(x_tmp, axis=3)


                    if self.gaitset:
                        if x_tmp.shape[0] == 50:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 2), dtype=x_tmp.dtype)
                            x_new[:, :, :, 0] = x_tmp[::2, :, :]
                            x_new[:, :, :, 1] = x_tmp[1::2, :, :]
                        else:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 1), dtype=x_tmp.dtype)
                            x_new[:, :, :, 0] = x_tmp
                        x_tmp = x_new
                    x[i] = x_tmp

                # Label
                if self.labmap:
                    lb = self.labmap[int(label)]
                else:
                    lb = label

                if self.multitask:
                    y[0][i] = lb
                else:
                    y[i] = lb

        if self.multitask:
            y[1] = keras.utils.to_categorical(y[0], num_classes=self.n_classes)

        w = [None, np.ones(shape=(len(x), 1))]
        #return x, y, w
        return x, y

    def __gen_batchMM(self,list_IDs_temp, expand=1):
        """ The >2 modalities case """
        if expand < 1:
            expand = 1

        dim0 = len(list_IDs_temp) * expand
        x = []
        for mix in range(self.nmods):
            x_ = [np.empty((dim0, *self.dim[mix])), np.empty((dim0, 1))]
            x = x + x_

        if self.multitask:
            y = [np.empty((dim0, 1)),
                 np.empty((dim0, self.n_classes))]
        else:
            #y = np.empty((dim0 * expand, 1))
            y = np.empty((dim0, 1))

        # Generate data
        if self.isDebug:
            # Fake
            x = []
            x.append(np.random.uniform(-1, 1, (self.batch_size, self.dim[0][0], 60, 60)))
            x.append(np.random.uniform(0, 2, (self.batch_size, 1)))
            x.append(np.random.uniform(-1, 1, (self.batch_size, 25, 60, 60)))
            x.append(np.random.uniform(0, 2, (self.batch_size, 1)))
            x.append(np.random.uniform(-1, 1, (self.batch_size, 25, 60, 60)))
            x.append(np.random.uniform(0, 2, (self.batch_size, 1)))
            if self.multitask:
                y = [np.floor(np.random.uniform(0, 4, (self.batch_size, 1))),
                     np.floor(np.random.uniform(0, 2, (self.batch_size, self.n_classes)))]
                # if self.softlabel:
                #     y[1] = mj_softlabel(y[0], self.n_classes)
            else:
                y = np.floor(np.random.uniform(0, 32, (self.batch_size, 1)))
        else:
            for i, ID in enumerate(list_IDs_temp):

                pair = self.allSamples[ID]
                mods = pair[0]
                # mod1 = mods[0]
                # mod2 = mods[1]
                # mod3 = mods[2]

                label = pair[1]

                filepaths = []
                for j in range(self.nmods):
                    if mods[j] != -1:
                        filepaths.append(osp.join(self.datadir[j], mods[j]))
                    else:
                        filepaths.append(None)

                # Load data from file
                # Data augmentation?
                if self.augmentation and np.random.randint(4) > 0:
                    trans = [self.img_gen[0].get_random_transform((self.dim[0][1], self.dim[0][2])),
                             self.img_gen[1].get_random_transform((self.dim[1][1], self.dim[1][2]))]
                    trans[1]["tx"] = trans[0]["tx"]
                    trans[1]["ty"] = trans[0]["ty"]
                    flip = np.random.randint(2) == 1
                else:
                    trans = None
                    flip = False

                if self.withof and self.augmentation and np.random.randint(2) == 1:
                    clip_max = 2300
                    clip_min = 50
                else:
                    clip_max = 0
                    clip_min = 0

                # Store sample
                for mix in range(self.nmods):  # Loop over modalities
                    x_tmp = self.__load_dd(filepaths[mix], self.ntype,
                                                              clip_max=clip_max, clip_min=clip_min)

                    if x_tmp is None:
                        x[2*mix][i * expand,] = self.noise
                        x[2*mix+1][i * expand,] = 0.0
                    else:
                        if trans is not None:
                            x_tmp = DA.mj_transformsequence(x_tmp, self.img_gen[0], trans[0])
                            if flip:
                                x_tmp = DA.mj_mirrorsequence(x_tmp, self.withof, True)

#                        if self.dim[0][0] != 50 and self.use3D and x_tmp is not None:  # It is not OF
#                            x_tmp = np.expand_dims(x_tmp, axis=3)

#                        if (len(x[2*mix].shape)-1) != len(x_tmp.shape):
#                            x_tmp = np.expand_dims(x_tmp, axis=3)

                        if self.gaitset:
                            if x_tmp.shape[0] == 50:
                                x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 2), dtype=x_tmp.dtype)
                                x_new[:, :, :, 0] = x_tmp[::2, :, :]
                                x_new[:, :, :, 1] = x_tmp[1::2, :, :]
                            else:
                                x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 1), dtype=x_tmp.dtype)
                                x_new[:, :, :, 0] = x_tmp
                            x_tmp = x_new

                        x[2*mix][i * expand,] = x_tmp
                        x[2*mix+1][i * expand,] = 1.0

                #lb = -1
                if self.labmap:
                    lb = self.labmap[int(label)]
                else:
                    lb = label

                if self.multitask:
                    y[0][i * expand] = lb
                else:
                    y[i * expand] = lb

                # Same pairs but disabling modalities
                if expand > 1:
                    nmore = expand-1  # self.nmods - 1
                    # Set label
                    for ex in range(nmore):
                        if self.multitask:
                            y[0][(ex+1) + i * expand] = lb
                        else:
                            y[(ex+1) + i * expand] = lb

                    for ex in range(nmore):
                        # Randomly choose which modality to disable
                        if i % 2 == 0:
                            if expand > 2:
                                ndisable = min(ex+1, self.nmods-1)
                            else:
                                ndisable = random.randrange(1, self.nmods, 1)
                            l_dis = [1]*self.nmods
                            for ff in range(ndisable):
                                choice1 = random.randrange(0, self.nmods, 1)
                                l_dis[choice1] = 0
                        else:
                            # Just one single modality enabled
                            l_dis = [0] * self.nmods
                            l_dis[(i+ex) % 3] = 1

                        # Copy all but disabled
                        for j in range(self.nmods):
                            if l_dis[j] == 0:
                                x[2*j][(ex+1) + i * expand,] = self.noise
                                x[2*j+1][(ex+1) + i * expand,] = 0.0
                            else:
                                x[2*j][(ex+1) + i * expand,] = np.copy(x[2*j][i * expand,])
                                x[2*j+1][(ex+1) + i * expand,] = 1.0

                    #import pdb; pdb.set_trace()
                    #print("Stop")

        if self.multitask:
            # if self.softlabel:
            #     y[1] = mj_softlabel(y[0], self.n_classes)
            # else:
            y[1] = keras.utils.to_categorical(y[0], num_classes=self.n_classes)
        
        return x, y

    def __data_generation(self, list_IDs_temp, expand=2):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        w = None
        if self.nmods > 2:
            x, y = self.__gen_batchMM(list_IDs_temp, expand=expand)
        elif self.nmods == 1:
            x, y = self.__gen_batchSingle(list_IDs_temp)
        else:
            if self.use_weights:
                x, y, w = self.__gen_batch(list_IDs_temp, expand=expand)
            else:
                x, y = self.__gen_batch(list_IDs_temp, expand=expand)

        if self.use_weights:
            return x, y, w
        else:
            return x, y


def inflateListRecords(allRecords_single_tr):
    for i in range(len(allRecords_single_tr)):
        r = allRecords_single_tr[i]
        r_new = []
        for j in range(len(r)):
            if j == 0:
                t = r[j]
                r_new.append((t[0], t[1], t[1]))
            else:
                r_new.append(r[j])

        allRecords_single_tr[i] = r_new

    return allRecords_single_tr


def mj_splitTrainValGaitByInfo(Iof):
    allPairs = Iof["records"]

    rec2gait = Iof["rec2gait"]
    rec2vid = Iof["rec2vid"]

    # Split in training/val data, grouped per video
    rec2lab = {p[0][0]: p[1] for p in allPairs}
    all_keys = [p[0][0] for p in allPairs]  # rec2vid.keys()
    all_vids = [rec2vid[key] for key in all_keys]
    uvids = np.unique(all_vids)
    nvids = len(uvids)
    np.random.shuffle(uvids)
    perc = 0.09
    nval = int(perc * nvids)
    vids_tr = [uvids[i] for i in range(0, nvids - nval)]
    vids_val = [uvids[i] for i in range(nvids - nval, nvids)]
    allRecords_single_tr = []
    for vix in vids_tr:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_tr = allRecords_single_tr + this_vid_pairs

    if len(np.unique([l[1] for l in allRecords_single_tr])) < 150:
        print("More classes needed!")
        import pdb;
        pdb.set_trace()

    allRecords_single_val = []
    for vix in vids_val:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_val = allRecords_single_val + this_vid_pairs

    return allRecords_single_tr, allRecords_single_val, rec2gait
# ============= MAIN ================
if __name__ == "__main__":


    infofile = osp.join("/home/GAIT_local/SSD/mjmarin/experiments/tumgaid_info/", "tfimdb_tum_gaid_N150_train_of25_60x60_info_of+{}.h5".format("gray"))

    Iof = dd.io.load(infofile)

    allRecords_single_tr, allRecords_single_val, rec2gait = mj_splitTrainValGaitByInfo(Iof)

    allRecords_single_tr = inflateListRecords(allRecords_single_tr)
    allRecords_single_val = inflateListRecords(allRecords_single_val)

    allLabels = [r[1] for r in allRecords_single_tr] + [r[1] for r in allRecords_single_val]
    ulabels = np.unique(allLabels)
    # Create mapping for labels
    labmap = {}
    for ix, lab in enumerate(ulabels):
        labmap[int(lab)] = ix

    gait_tr = [rec2gait[r[0][0]] for r in allRecords_single_tr]
    gait_val = [rec2gait[r[0][0]] for r in allRecords_single_val]

    dbbasedir = "/home/GAIT_local/SSD/TUM_GAID_tf/"
    datadirs = [osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_of25_60x60"),
                osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_{}25_60x60".format("gray")),
                osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_depth25_60x60")]

    input_shape = [(50, 60, 60), (25, 60, 60), (25, 60, 60)]

    sess = tf.compat.v1.Session()
    with sess.as_default():
        dg = DataGeneratorGaitMMUWYH(allRecords_single_tr+allRecords_single_tr+allRecords_single_tr,
                                    datadir=datadirs, batch_size=32, isDebug=False, dim=input_shape, labmap=labmap,
                                     gait=gait_tr+gait_tr+gait_tr, expand_level=1, ntype=2, nmods=3, repetition=4)

    X = []
    Y = []
    for e in range(0, len(dg)):
        X_, Y_ = dg.__getitem__(e)
        X.append(X_)
        Y.append(Y_)
    print("Done!")
