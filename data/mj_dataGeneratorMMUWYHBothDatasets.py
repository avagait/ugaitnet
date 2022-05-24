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
import h5py

np.random.seed(232323)
import random

from data.mj_tfdata import *
import os.path as osp

import data.mj_augmentation as DA


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

    def __init__(self, allSamples, dataset_source, targets=[], batch_size=32, dim=(50, 60, 60), n_classes=224,
                 shuffle=True, augmentation=True,
                 datadir=["/home/GAIT_local/SSD/tfimdb_tum_gaid_N150_train_of25_60x60",
                          "/home/GAIT_local/CASIAB_tf/tfimdb_casia_b_N074_train_of25_60x60"],
                 sess=None, labmap=[], gait=[], buildGaits=[], ntype=1,
                 isTest=False, augmentation_x=1, expand_level=2, balanced_classes=True,
                 isTriplet=False, use3D=False, isDebug=False, softlabel=False,
                 camera=[], nmods=1, use_weights=False, gaitset=False,
                 meanSample=0.0, aux_losses=False, focal_loss=False, normalize_paths=None):
        'Initialization'
        self.dim = dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.meanSample = meanSample
        self.shuffle = shuffle
        self.gaitset = gaitset

        self.isTriplet = isTriplet
        self.isTest = isTest
        self.isDebug = isDebug
        self.use3D = use3D
        self.softlabel = softlabel
        self.aux_losses = aux_losses
        self.nmods = nmods  # Number of modalities
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
        self.dataset_source = dataset_source
        self.targets = targets
        self.labmap = labmap
        self.gait = gait
        self.focal_loss = focal_loss

        if buildGaits == []:
            self.buildGaits = np.unique(self.gait)
        else:
            self.buildGaits = buildGaits

        self.camera = camera

        self.datadir = datadir
        self.ntype = ntype

        self.normalize_paths = normalize_paths
        if normalize_paths is not None:
            hf_1 = h5py.File(normalize_paths[0], 'r')
            hf_2 = h5py.File(normalize_paths[1], 'r')

            self.normalize_means = [np.asarray(hf_1["mean"]), np.asarray(hf_2["mean"])]
            self.normalize_stds = [np.asarray(hf_1["std"]), np.asarray(hf_2["std"])]
        else:
            self.normalize_means = [None, None]
            self.normalize_stds = [None, None]

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

    def __len__(self):
        'Number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(len(self.indexes) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        list_IDs_temp = []
        this_lab_used = 0
        ind_g = 0
        count_build_gaits = np.zeros(len(self.ugait))
        global_count_builds_gaits = 0
        while len(list_IDs_temp) < self.batch_size:
            if ind_g == 0:
                global_count_builds_gaits = global_count_builds_gaits + 1

            indexPosList = [i for i in range(len(self.buildGaits)) if self.buildGaits[i] == self.buildGaits[ind_g]]

            if len(indexPosList) > 1:
                sum = np.sum(count_build_gaits[indexPosList])
                if sum < global_count_builds_gaits:
                    indexPosList.remove(ind_g)
                    jump_g = False
                    for ind_temp in indexPosList:
                        if count_build_gaits[ind_g] > count_build_gaits[ind_temp]:
                            jump_g = True
                    if jump_g:
                        ind_g = (ind_g + 1) % len(self.ugait)
                        continue
                else:
                    ind_g = (ind_g + 1) % len(self.ugait)
                    continue


            g = self.ugait[ind_g]


            lab2ptr = self.gait2ptr[g]
            lab_ix = self.ulabs[self.nextlab_idx]
            ix = lab2ptr[lab_ix]
            lab2rec = self.gait2idx[g]
            if len(lab2rec[lab_ix]) > 0:
                rec = lab2rec[lab_ix][ix]
                list_IDs_temp.append(rec)
                count_build_gaits[ind_g] = count_build_gaits[ind_g] + 1
                ind_g = (ind_g + 1) % len(self.ugait)

            this_lab_used += 1

            lab2ptr[lab_ix] += 1
            if lab2ptr[lab_ix] >= len(lab2rec[lab_ix]):
                lab2ptr[lab_ix] = 0

            if this_lab_used >= 2:
                this_lab_used = 0
                self.nextlab_idx += 1
                if self.nextlab_idx >= len(self.ulabs):
                    self.nextlab_idx = 0

        
        # Generate data
        if self.use_weights:
            X, y, w = self.__data_generation(list_IDs_temp)
            return X, y, w
        else:
            X, y = self.__data_generation(list_IDs_temp, expand=self.expand_level)
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

    def __load_dd(self, filepath: str, ntype=1, clip_max=0, clip_min=0, normalize_mean=None, normalize_std=None):
        """
        Loads a dd file with gait samples
        :param filepath: full path to h5 file (deep dish compatible)
        :return: numpy array with data
        """
        if filepath is None:
            return None

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

        if normalize_mean is not None:
            mean = np.repeat(normalize_mean[:, np.newaxis], 60, axis=1)
            mean = np.repeat(mean[:, np.newaxis], 60, axis=1)

            std = np.repeat(normalize_std[:, np.newaxis], 60, axis=1)
            std = np.repeat(std[:, np.newaxis], 60, axis=1)

            x = (x - mean)/std
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
                # mod1 = pair1[0]
                # mod2 = pair1[1]

                label = pair[1]

                if self.ntype == 1:
                    for set1 in range(3):
                        #filepath1 = osp.join(self.datadir[0], "{:d}_sample{:06d}.h5".format(set1, mod1))
                        filepath1 = osp.join(self.datadir[self.dataset_source[ID]*2], pair1)
                        if osp.exists(filepath1):
                            break

                    for set2 in range(3):
                        #filepath2 = osp.join(self.datadir[1], "{:d}_sample{:06d}.h5".format(set2, mod2))
                        filepath2 = osp.join(self.datadir[self.dataset_source[ID]*2+1], pair1)
                        if osp.exists(filepath2):
                            break
                else:

                    filepath1 = osp.join(self.datadir[self.dataset_source[ID] * 2], pair1)
                    filepath2 = osp.join(self.datadir[self.dataset_source[ID] * 2 + 1], pair1)
                    # if mod1 != -1:
                    #     filepath1 = osp.join(self.datadir[0], mod1)
                    # else:
                    #     filepath1 = None
                    # if mod2 != -1:
                    #     filepath2 = osp.join(self.datadir[1], mod2)
                    # else:
                    #     filepath2 = None

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
                x_tmp = self.__load_dd(filepath1, self.ntype, clip_max=clip_max, clip_min=clip_min,
                                                          normalize_mean=self.normalize_means[self.dataset_source[ID]],
                                                          normalize_std=self.normalize_stds[self.dataset_source[ID]])

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
                            x_new[:, :, :, 0] = x_tmp[::2, :, :, 0]
                            x_new[:, :, :, 1] = x_tmp[1::2, :, :, 0]
                        else:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 1), dtype=x_tmp.dtype)
                            x_new = x_tmp
                        x_tmp = x_new

                    x[0][i * expand,] = x_tmp
                    x[1][i * expand,] = 1.0

                M_ = self.__load_dd(filepath2, self.ntype, normalize_mean=self.normalize_means[self.dataset_source[ID]],
                                                          normalize_std=self.normalize_stds[self.dataset_source[ID]])

                # if M_ is None:
                #     import pdb; pdb.set_trace()

                if trans is not None:
                    M_ = DA.mj_transformsequence(M_, self.img_gen[1], trans[1])
                    #print("Mod-2: {}".format([M_.min(), M_.max()]))  # DEVELOP
                    if flip:
                        M_ = DA.mj_mirrorsequence(M_, isof=False)

                if self.use3D and M_ is not None:
                    M_ = np.expand_dims(M_, axis=3)
                # import pdb; pdb.set_trace()

                if self.gaitset:
                    if M_.shape[0] == 50:
                        x_new = np.zeros((25, M_.shape[1], M_.shape[2], 2), dtype=M_.dtype)
                        x_new[:, :, :, 0] = M_[::2, :, :]
                        x_new[:, :, :, 1] = M_[1::2, :, :]
                    else:
                        x_new = np.zeros((25, M_.shape[1], M_.shape[2], 1), dtype=M_.dtype)
                        x_new[:, :, :, 0] = M_[:,:,:,0]
                    M_ = x_new


                x[2][i * expand, :, :, :, ] = M_
                x[3][i * expand,] = 1.0

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
            # w = [np.ones(shape=(len(x[0]))), np.ones(shape=(len(x[0])))] # np.ones(shape=(1, len(x[0]))) #[None, None] #(np.ones(shape=(1, len(x[0]))), np.ones(shape=(1, len(x[0]))))
            # w = [None, np.ones(shape=(len(x[0]),1))]
            if self.multitask:
                w = [None, None]
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
                if self.focal_loss:
                    y = [np.floor(np.random.uniform(0, 4, (self.batch_size, 1))),
                         np.floor(np.random.uniform(0, 2, (self.batch_size, 1)))]
                else:
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
                        filepath1 = osp.join(self.datadir[self.dataset_source[ID]], pair1)
                        if osp.exists(filepath1):
                            break
                else:
                    # if mod1 != -1:
                    #     filepath1 = osp.join(self.datadir[0], mod1)
                    # else:
                    #     filepath1 = None
                    filepath1 = osp.join(self.datadir[self.dataset_source[ID]], pair1)

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
                x_tmp = self.__load_dd(filepath1, self.ntype, clip_max=clip_max, clip_min=clip_min,
                                                          normalize_mean=self.normalize_means[self.dataset_source[ID]],
                                                          normalize_std=self.normalize_stds[self.dataset_source[ID]])

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
                            x_new[:, :, :, 0] = x_tmp[::2, :, :, 0]
                            x_new[:, :, :, 1] = x_tmp[1::2, :, :, 0]
                        else:
                            x_new = np.zeros((25, x_tmp.shape[1], x_tmp.shape[2], 1), dtype=x_tmp.dtype)
                            x_new = x_tmp
                        x_tmp = x_new
                    x[i] = x_new

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
            y[1] = keras.utils.to_categorical(y[0], num_classes=self.n_classes).astype(np.float32)

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
            y = np.empty((self.batch_size * expand, 1))

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
                                                              clip_max=clip_max, clip_min=clip_min,
                                                              normalize_mean=self.normalize_means[self.dataset_source[ID]],
                                                          normalize_std=self.normalize_stds[self.dataset_source[ID]])

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

                        if (len(x[2*mix].shape)-1) != len(x_tmp.shape):
                            x_tmp = np.expand_dims(x_tmp, axis=3)
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
        if self.nmods == 1:
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



# ============= MAIN ================
if __name__ == "__main__":
    import glob

    datadir_1 = "/home/GAIT_local/SSD/tfimdb_tum_gaid_N150_train_of25_60x60"
    datadir_2 = "/home/GAIT_local/CASIAB_tf/tfimdb_casia_b_N074_train_of25_60x60"
    datadirs = ["/home/GAIT_local/SSD/tfimdb_tum_gaid_N150_train_of25_60x60",
                "/home/GAIT_local/SSD/tfimdb_tum_gaid_N150_train_gray25_60x60",
                "/home/GAIT_local/CASIAB_tf/tfimdb_casia_b_N074_train_of25_60x60",
                "/home/GAIT_local/CASIAB_tf/tfimdb_casia_b_N074_train_gray25_60x60"]

    info_file_1 = dd.io.load("/home/GAIT_local/SSD/mjmarin/experiments/tumgaid_info/tfimdb_tum_gaid_N150_train_of25_60x60_info_of+gray.h5")
    info_file_2 = dd.io.load("/home/GAIT_local/CASIAB_tf/tfimdb_casia_b_N074_train_of25_60x60.h5")
    allRecords = []
    dataset_source = []
    allGaits = []   # n:0, b:1, s:2, nm:3, bg:4, cl:5
    alllabels = []
    for i in info_file_1['records']:
        allRecords.append((i[0][0], i[1]))
        alllabels.append(i[1])
        dataset_source.append(0)
        if i[0][0][5] == 'n':
            allGaits.append(0)
        elif i[0][0][5] == 'b':
            allGaits.append(1)
        else:
            allGaits.append(2)


    for i in range(len(info_file_2['file'])):
        allRecords.append((info_file_2['file'][i], info_file_2['label'][i]+305))
        alllabels.append(info_file_2['label'][i]+305)
        dataset_source.append(1)
        if info_file_2['file'][i][4:6] == 'nm':
            allGaits.append(3)
        elif info_file_2['file'][i][4:6] == 'bg':
            allGaits.append(4)
        else:
            allGaits.append(5)

    # allLabels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
    #allGaits = [0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 0, 1]
    # Find label mapping for training

    ulabels = np.unique(alllabels)
    # Create mapping for labels
    labmap = {}
    for ix, lab in enumerate(ulabels):
        labmap[int(lab)] = ix

    buildGaits = [0, 1, 2, 0, 4, 5]

    input_shape = [(50, 60, 60), (25, 60, 60)]

    sess = tf.compat.v1.Session()
    with sess.as_default():
        dg = DataGeneratorGaitMMUWYH(allRecords, dataset_source, datadir=datadirs, batch_size=24, isDebug=False,
                                     gait=allGaits, buildGaits=buildGaits, ntype=2, labmap=labmap, nmods=2, dim=input_shape)

    for e in range(0, len(dg)):
        X, Y = dg.__getitem__(e)
        print("debug")

    print("Done!")
