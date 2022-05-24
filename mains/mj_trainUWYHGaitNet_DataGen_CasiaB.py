# Trains a gait recognizer CNN
# This version uses a custom DataGenerator

import sys, os

import math
import numpy as np
import gc

import os.path as osp
from os.path import expanduser

import pathlib
maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")

# --------------------------------
import tensorflow as tf
theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90 #gpu_rate # TODO

tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import optimizers

import deepdish as dd
from nets.mj_uwyhNets_ba import UWYHSemiNet
from nets.mj_metrics import mj_eerVerifDist

from data.mj_dataGeneratorMMUWYHsingle_repetitions import DataGeneratorGaitMMUWYH
from utils.mj_netUtils import mj_findLatestFileModel
from utils.rd_JSONInfo import rd_JSONInfo
import tensorflow_addons as tfa
from utils.mj_utils import mj_isDebugging
IS_DEBUG = False

# =============== PROJECTOR FUNCTIONS ===========================
from nets.mj_utils import mj_register_embedding, mj_save_labels_tsv, mj_save_sprite, mj_save_filters, mj_save_filters3d
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'

# ===============================================================

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
    perc = 0.08
    nval = int(perc * nvids)
    vids_tr = [uvids[i] for i in range(0, nvids - nval)]
    vids_val = [uvids[i] for i in range(nvids - nval, nvids)]
    allRecords_single_tr = []
    for vix in vids_tr:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_tr = allRecords_single_tr + this_vid_pairs

    allRecords_single_val = []
    for vix in vids_val:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_val = allRecords_single_val + this_vid_pairs

    return allRecords_single_tr, allRecords_single_val, rec2gait


def mj_computeDistMetrics(model, val_generator, multitask, lab4color, gaitset=False):
    gt_labels = []
    distances = []
    all_labs0 = []
    all_codes0 = []
    l_sprites = []

    for bix in range(len(val_generator)):
        # import pdb; pdb.set_trace()
        tuples, labels = val_generator.__getitem__(bix)

        if multitask:
            labels = labels[0]

        of0, uof0, gray0, ug0 = tuples
        # labs = np.squeeze(labs)

        codes0 = UWYHSemiNet.encode(model, [of0, gray0], [uof0, ug0], gaitset)
        if multitask:
            code0_labs = [int(x) for x in labels]  # Transformation not needed in this case
        else:
            code0_labs = [lab4color[int(x)] for x in labels]  # Use correlative labels for color
        all_labs0 = all_labs0 + code0_labs
        if bix == 0:
            all_codes0 = codes0
        else:
            all_codes0 = np.append(all_codes0, codes0, axis=0)

        # Create positive and negative pairs
        _ulabs = np.unique(code0_labs)
        difs = []
        dif_labs = []
        NnegsPerLab = 3
        for u in _ulabs:
            idx_u_pos = np.where(code0_labs == u)[0]
            np.random.shuffle(idx_u_pos)
            idx_u_neg = np.where(code0_labs != u)[0]
            np.random.shuffle(idx_u_neg)

            # One positive for this one
            if len(idx_u_pos) > 1:
                i = idx_u_pos[0]
                j = idx_u_pos[1]
                if len(difs) > 0:
                    difs = np.vstack((difs, codes0[i] - codes0[j]))
                else:
                    difs = codes0[i] - codes0[j]
                dif_labs.append(1)

            # N negatives for this one
            if len(idx_u_neg) >= NnegsPerLab:
                i = idx_u_pos[0]
                for t in range(0, NnegsPerLab):
                    j = idx_u_neg[t]
                    if len(difs) > 0:
                        difs = np.vstack((difs, codes0[i] - codes0[j]))
                    else:
                        difs = codes0[i] - codes0[j]
                    dif_labs.append(0)


        # Compute distance
        dist = np.linalg.norm(difs, axis=1)
        if bix == 0:
            gt_labels = dif_labs
            distances = dist
        else:
            gt_labels = np.append(gt_labels, dif_labs)
            distances = np.append(distances, dist)

    # Compute EER
    eer_val, thr_eer_val = mj_eerVerifDist(gt_labels, distances)

    chance = np.sum(gt_labels > 0) / len(gt_labels)  # Compute chance

    return distances, eer_val, chance, [all_codes0, all_labs0]


def sign_max(**kwargs):
    def compute(x):
        dims = tf.shape(x[0])
        cat_data = tf.reshape(tf.stack(x, axis=0), [len(x), -1])
        max_pos = tf.math.argmax(tf.math.abs(cat_data), axis=0)
        max_pos = tf.stack([max_pos, tf.range(dims[0]*dims[1]*dims[2], dtype=max_pos.dtype)], axis=0)
        data = tf.gather_nd(cat_data, tf.transpose(max_pos))
        data = tf.reshape(data, dims)
        return data
    return tf.keras.layers.Lambda(compute, **kwargs)


def trainUWYHGaitNet(datadir="tfimdb_casia_b_N074_train_of25_60x60", dbbasedir='/home/GAIT_local/SSD', experfix="demo",
                     nclasses=74, lr=0.001, dropout=0.4, dropout0=-1,
                     experdirbase=".", epochs=5, batchsize=32, optimizer="SGD",
                     ndense_units=512, margin=0.2, savemodelfreq=2, casenet='B',
                     dynmargin=False, hardnegs=4, loss_weights=[1.0, 1.0],
                     modality="gray", initnet="", use3D=False, softlabel=False,
                     datatype = 1, infodir="", freeze_all=False, nofreeze=False,
                     postriplet=1, with_missing=True, mergefun="Maximum",
                     logdir="", extra_epochs=0, verbose=0, fActivation='relu', gaitset=False, repetitions=16, multigpu=0):
    """
    Trains a CNN for gait recognition
    :param datadir: root dir containing dd files
    :param experfix: string to customize experiment name
    :param nclasses: number of classes
    :param lr: starting learning rate
    :param dropout: dropout value
    :param tdim: extra dimension of samples. Usually 50 for OF, and 25 for gray and depth
    :param epochs: max number of epochs
    :param batchsize: integer
    :param optimizer: options like {SGD, Adam,...}
    :param logdir: path to save Tensorboard info
    :param ndense_units: number of dense units for last FC layer
    :param verbose: integer
    :return: model, experdir, accuracy
    """
    
    if use3D:
        input_shape = [(50, 60, 60), (25, 60, 60, 1)]
    else:
        input_shape = [(50, 60, 60), (25, 60, 60)]

    if gaitset:
        input_shape = [(25, 60, 60, 2), (25, 60, 60, 1)]

    multitask = nclasses > 0
    number_convolutional_layers = 4
    if not casenet == 'A':
        filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
        filters_numbers = [96, 192, 512, 512]
    else:
        filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
        filters_numbers = [64, 128, 512, 512]

    weight_decay = 0.00005
    momentum = 0.9

    optimfun = optimizers.Adam(lr=lr)
    infix = "_opAdam"
    if optimizer != "Adam":
        infix = "_op" + optimizer
        if optimizer == "SGD":
            optimfun = optimizers.SGD(lr=lr, momentum=momentum, decay=1e-05)
        elif optimizer == "AMSGrad":
            optimfun = optimizers.Adam(lr=lr, amsgrad=True)
        elif optimizer == "AdamW":
            optimfun = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        else:
            optimfun = eval("optimizers." + optimizer + "(lr=initialLR)")


    if not with_missing:
        infix = "_full"

    if use3D:
        infix = "_of+" + modality + "3D" + infix
    else:
        infix = "_of+" + modality + infix

    if datatype == 2:
        infix = infix + "_dt2"

    if softlabel:
        infix = infix + "_sl"
        smoothlabval = 0.1
    else:
        smoothlabval = 0

    if casenet != 'B':
        infix = infix + "_v" + casenet
    if postriplet != 1:
        infix = infix + "pt{}".format(postriplet)

    if dropout0 != -1:
        infix = infix + "_dz{:.1f}".format(dropout0)
    else:
        dropout0 = dropout

    if nofreeze:
        freeze_convs = False
    else:
        freeze_convs = True
    if initnet != "" and freeze_all:
        infix = infix + "_frall"

    if nclasses > 0:
        infix = infix + "_mTask"
        if loss_weights[0] != 1.0:
            infix = infix + "V{:.2f}".format(loss_weights[0])
        if loss_weights[1] != 1.0:
            infix = infix + "I{:.2f}".format(loss_weights[1])

    if ndense_units != 2048:
        infix = infix + "_nd{:04d}".format(ndense_units)

    if margin != 0.5:
        infix = infix + "_mg{:03d}".format(int(margin*100))
    if dynmargin:
        infix = infix + "_dyn"
    if hardnegs != 0:
        infix = infix + "_hn{:02d}".format(hardnegs)

    if mergefun != "Maximum":
        infix = infix + "_mg" + mergefun[0:3]

    if gaitset:
        infix = infix + 'gaitset'

    # Create a TensorBoard instance with the path to the logs directory
    subdir = experfix + '_datagen{}_bs{:03d}_lr{:0.6f}_dr{:0.2f}'.format(infix, batchsize, lr,
                                                                         dropout)  # To be customized

    experdir = osp.join(experdirbase, subdir)
    if verbose > 0:
        print(experdir)
    if not osp.exists(experdir):
        import os
        os.makedirs(experdir)

    # Add extra dense layer with dropout?
    if casenet == 'C':
        ndense_units = [ndense_units, 256]

    if casenet == 'D':
        ndense_units = [ndense_units]

    # Custom merge function?
    from tensorflow.keras.layers import Maximum, Average
    if mergefun == "Maximum":
        fMerge = Maximum
    elif mergefun == "Average":
        fMerge = Average
    else:
        fMerge = eval(mergefun)

    # Prepare model
    pattern_file = "model-state-{:04d}.hdf5"
    previous_model = mj_findLatestFileModel(experdir, pattern_file, epoch_max=epochs)
    print(previous_model)
    initepoch = 0
 
    if previous_model != "":
        pms = previous_model.split("-")
        initepoch = int(pms[len(pms) - 1].split(".")[0])
        print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))

        model = UWYHSemiNet.loadnet(previous_model)
    else:
        if initnet != "":
            print("* Model will be init from: "+initnet)


        if multigpu > 0:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = UWYHSemiNet.build_or_load(input_shape, number_convolutional_layers,
                                   filters_size, filters_numbers, ndense_units, weight_decay, [dropout0, dropout],
                                   optimizer=optimfun, margin=margin, nclasses=nclasses, loss_weights=loss_weights,
                                   initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, smoothlabels=smoothlabval,
                                    freeze_all=freeze_all, postriplet=postriplet, fMerge=fMerge, fActivation=fActivation, gaitset=gaitset)
        else:
            model = UWYHSemiNet.build_or_load(input_shape, number_convolutional_layers,
                                   filters_size, filters_numbers, ndense_units, weight_decay, [dropout0, dropout],
                                   optimizer=optimfun, margin=margin, nclasses=nclasses, loss_weights=loss_weights,
                                   initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, smoothlabels=smoothlabval,
                                    freeze_all=freeze_all, postriplet=postriplet, fMerge=fMerge, fActivation=fActivation, gaitset=gaitset)
    model.summary()


    # Tensorboard
    if logdir == "":
        logdir = experdir
        # Save checkpoint
        chkptname = osp.join(logdir, "weights.{epoch:02d}.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(chkptname, save_best_only=True, save_weights_only=True)

        from tensorflow.keras.callbacks import TensorBoard

        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=2, write_graph=True, write_images=False,
                                  profile_batch = 5)
        callbacks = [tensorboard, checkpoint]
    else: # This case is for parameter tuning
        # Save checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(logdir, save_best_only=True)

        from tensorflow.keras.callbacks import TensorBoard
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
                                  profile_batch = 3)

        # hpcallback = hp.KerasCallback("/tmp/mjmarin/logs/hparam_tuning", hparams)
        hpcall = hp.KerasCallback(logdir, hparams)
        callbacks = [tensorboard, checkpoint, hpcall]

    # ---------------------------------------
    # Prepare data
    # ---------------------------------------

    if verbose > 0:
        print("Preparing training/val splits...")

    rec2gait = []
    rec2vid = []

    if infodir == "":
        h5dir = "/home/mjmarin/databases/casiab/h5"
    else:
        h5dir = infodir

    if nclasses == 74:
        infofile = osp.join(h5dir, "tfimdb_casia_b_N074_train_of25_60x60_info_of+{}.h5".format(modality))
    else:
        infofile = osp.join(h5dir, "tfimdb_casia_b_N050_ft_of25_60x60_info_of+{}.h5".format(modality))
        #"tfimdb_casia_b_N050_ft_of25_60x60_info_of+gray.h5"
    Iof = dd.io.load(infofile)

    allRecords_single_tr, allRecords_single_val, rec2gait = mj_splitTrainValGaitByInfo(Iof)

    # Find label mapping for training
    if nclasses > 0:
        allLabels = [r[1] for r in allRecords_single_tr] + [r[1] for r in allRecords_single_val]
        ulabels = np.unique(allLabels)
        # Create mapping for labels
        labmap = {}
        for ix, lab in enumerate(ulabels):
            labmap[int(lab)] = ix
    else:
        labmap = None


    if IS_DEBUG:
        print("********** DEBUG MODE *************")
        savemodelfreq = 1
        import random
        random.shuffle(allRecords_single_tr)
        random.shuffle(allRecords_single_val)

        allRecords_single_tr = [allRecords_single_tr[ix] for ix in range(0, 500*9, 9)] #allRecords_tr[0:500]
        if nclasses != 155:
            allRecords_single_val = [allRecords_single_val[ix] for ix in range(0,100*9, 9)] #allRecords_val[0:300]

    if dynmargin or hardnegs > 0:
        savemodelfreq = 2            # FIXME

    steps_per_epoch = int(len(allRecords_single_tr) / batchsize) * 2  # * x means data augmentation
    validation_steps = int(len(allRecords_single_val) / batchsize)

    # Info about samples
    # Gait type for each sample
    gait_tr = [rec2gait[r[0][0]] for r in allRecords_single_tr]
    gait_val = [rec2gait[r[0][0]] for r in allRecords_single_val]

    # Define directories
    if nclasses == 74:
        datadirs = [osp.join(dbbasedir, "tfimdb_casia_b_N074_train_of25_60x60"),
                    osp.join(dbbasedir, "tfimdb_casia_b_N074_train_{}25_60x60".format(modality))]
    else:
        datadirs = [osp.join(dbbasedir, "tfimdb_casia_b_N050_ft_of25_60x60"),
                    osp.join(dbbasedir, "tfimdb_casia_b_N050_ft_{}25_60x60".format(modality))]

    expand_level_tr = 3
    expand_level_val = 2
    if not with_missing:
        exp_lev_tr = 0
        exp_lev_val = 0


    print("datadirs", datadirs, flush=True)

    if multigpu > 0:
        batchsize_ = batchsize / multigpu
    else:
        batchsize_ = batchsize

    # MJ: augment training data
    train_generator = DataGeneratorGaitMMUWYH(allRecords_single_tr, datadir=datadirs, batch_size=batchsize_,
                                          dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                              ntype=datatype,
                                           gait=gait_tr, expand_level=expand_level_tr, use3D=use3D, repetition=repetitions, gaitset=gaitset)

    val_generator = DataGeneratorGaitMMUWYH(allRecords_single_val, datadir=datadirs, batch_size=batchsize_,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                            ntype=datatype,
                                           gait=gait_val, expand_level=expand_level_val, use3D=use3D, augmentation_x=0, repetition=repetitions, gaitset=gaitset)

    # Save useful info for recovering the model with different Python versions
    modelpars = {'filters_size': filters_size,
                 'filters_numbers': filters_numbers,
                 'input_shape': input_shape,
                 'ndense_units': ndense_units,
                 'weight_decay': weight_decay,
                 'dropout': dropout,
                 'optimizer': optimizer,
                 'margin': margin,
                 'custom': 'TripletSemiHardLoss',
                 'nclasses': nclasses,
                 'softlabel': smoothlabval,
                 'use3D': use3D,
                 'loss_weights': loss_weights,
                 'fMerge': mergefun}
    dd.io.save(osp.join(experdir, "model-config.hdf5"), modelpars)

    rd_JSONInfo(experdir, input_shape, nclasses, number_convolutional_layers, casenet, weight_decay, momentum, optimizer,
                use3D, datatype, softlabel, [dropout0, dropout], nofreeze, loss_weights, ndense_units,
                margin, dynmargin, hardnegs, mergefun, batchsize, lr, filters_size, filters_numbers)

    # Validation labels mapping for projector
    all_val_labs = [t[1] for t in allRecords_single_val]
    ulabs_val = np.unique(all_val_labs)
    lab4color = {l: ix for ix, l in enumerate(ulabs_val)}

    if multitask:
        lab4color = labmap

    # ---------------------------------------
    # Train model
    # --------------------------------------
    print("* Starting training...", flush=True)
    last_lr = lr

    ep_steps = int(epochs / savemodelfreq)
    if ep_steps > 1: # Save partial models in case the process dies
        ecum = initepoch

        for eix in range(ep_steps):
            if ecum >= epochs:
                print("End of main training.")
                break

            epochs_ = savemodelfreq
            if verbose > 1:
                print(experdir)
            model, hist = UWYHSemiNet.fit_generator(model, ecum+epochs_, callbacks, train_generator, None,
                                                    ecum, train_generator.__len__(), None) #ecum, steps_per_epoch, validation_steps)
            ecum += len(hist.epoch)
            model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(ecum)))

            # Save in such a way that can be recovered from different Python versions
            model.save_weights(osp.join(experdir, "model-state-{:04d}_weights.hdf5".format(ecum)))
            tf.keras.backend.clear_session() # Experimental use, see if it solves memory leak
            gc.collect()

        epochs = ecum # Final number of processed epochs 
    else:
        model, hist = UWYHSemiNet.fit_generator(model, epochs, callbacks, train_dataset.get(), val_dataset.get(),
                                                 initepoch, steps_per_epoch=train_generator.__len__(), validation_steps=val_generator.__len__())

    print("Last epoch: {}".format(epochs))
    model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(epochs)))
    model.save_weights(osp.join(experdir, "model-state-{:04d}_weights.hdf5".format(epochs)))

    # Fine-tune on remaining validation samples
    if extra_epochs > 0:
        if verbose > 0:
            print("Adding validation samples to training and run for few epochs...")
        del train_generator

        train_generator = DataGeneratorGaitMMUWYH(allRecords_single_val+allRecords_single_tr, datadir=datadirs, batch_size=batchsize_,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                        gait=gait_val+gait_tr, expand_level=3, use3D=use3D,
                                                  ntype=datatype, repetition=repetitions)

        ft_epochs = epochs + extra_epochs    # DEVELOP!
        if nclasses == 150:
            new_lr = (10**math.ceil(math.log10(last_lr))) * 0.1            # DEVELOP!
        else:
            new_lr = min(10**math.ceil(math.log10(last_lr)), last_lr)  # last_lr
        #new_lr = last_lr
        model, hist = UWYHSemiNet.fit_generator(model, ft_epochs, callbacks, train_dataset.get(), val_dataset.get(),
                                                epochs, steps_per_epoch=train_generator.__len__(), validation_steps=val_generator.__len__(), new_lr=new_lr)
        # Update num epochs
        epochs = ft_epochs

    model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(epochs)))

    return model, experdir, 0.5


################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Trains a CNN for gait')
    parser.add_argument('--tuning', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use3d', default=False, action='store_true')
    parser.add_argument('--gaitset', default=False, action='store_true')
    parser.add_argument('--softlabel', default=False, action='store_true')
    parser.add_argument('--freezeall', default=False, action='store_true', help="Freeze all weights?")
    parser.add_argument('--nofreeze', default=False, action='store_true', help="Avoid freezing any weight?")
    parser.add_argument('--nomissing', default=False, action='store_true', help="Disable missing modalities?")
    parser.add_argument('--dropout', type=float, required=False,
                        default=0.0,
                        help='Dropout value')
    parser.add_argument('--dropout0', type=float, required=False,
                        default=-1, help='Dropout value for 2D branches')
    parser.add_argument('--lr', type=float, required=False,
                        default=0.001,
                        help='Starting learning rate')
    parser.add_argument('--datadir', type=str, required=False,
                        default=osp.join('PATH/datatum', 'matimdbtum_gaid_N150_of25_60x60_lite/'),
                        help="Full path to data directory")
    parser.add_argument('--dbbasedir', type=str, required=False,
                        default='/home/GAIT_local/SSD_grande/CASIAB_tf',
                        help="Relative path to data directory")
    parser.add_argument('--infodir', type=str, required=False,
                        default="",
                        help="Full path to info dir (h5)")
    parser.add_argument('--experdir', type=str, required=True,
                        default=osp.join('PATH/experiments', 'tumgaid_uwyh'),
                        help="Base path to save results of training")
    parser.add_argument('--prefix', type=str, required=True,
                        default="demo",
                        help="String to prefix experiment directory name.")
    parser.add_argument('--bs', type=int, required=False,
                        default=32,
                        help='Batch size')
    parser.add_argument('--repetitions', type=int, required=False,
                        default=32,
                        help='Repetitions')
    parser.add_argument('--epochs', type=int, required=False,
                        default=5,
                        help='Maximum number of epochs')
    parser.add_argument('--extraepochs', type=int, required=False,
                        default=10,
                        help='Extra number of epochs to add validation data')
    parser.add_argument('--nclasses', type=int, required=True,
                        default=0,
                        help='Maximum number of epochs')
    parser.add_argument('--ndense', type=int, required=False,
                        default=2048,
                        help='Number of dense units')
    parser.add_argument('--casenet', type=str, required=False,
                        default="B",
                        help="Type of net: A, B")
    parser.add_argument('--margin', type=float, required=False,
                        default=0.5,
                        help='Margin used in loss')
    parser.add_argument('--dynmargin', default=False, action='store_true',
                        help="Use dynamic margin?")
    parser.add_argument('--hn', type=int, required=False,
                        default=0,
                        help='Hard negatives per batch')
    parser.add_argument('--tdim', type=int, required=False,
                        default=50,
                        help='Number of dimensions in 3rd axis time. E.g. OF=50')
    parser.add_argument('--optimizer', type=str, required=False,
                        default="SGD",
                        help="Optimizer: SGD, Adam, AMSGrad")
    parser.add_argument('--mod', type=str, required=False,
                        default="gray",
                        help="Extra modality: gray, depth")
    parser.add_argument('--datatype', type=int, required=False,
                        default=2,
                        help='Code 1 (matlab-h5) or 2 (new h5)')
    parser.add_argument('--postriplet', type=int, required=False,
                        default=1,
                        help='1=after_fusion, 2=dense_small')
    parser.add_argument('--wid', type=float, required=False,
                        default=1.0,
                        help='Weight for identification task')
    parser.add_argument('--wver', type=float, required=False,
                        default=1.0,
                        help='Weight for verification task')
    parser.add_argument('--initnet', type=str, required=False,
                        default="",
                        help="Path to net to initialize")
    parser.add_argument('--mergefun', type=str, required=False,
                        default="Maximum",
                        help="Choose: Maximum, Average")
    parser.add_argument("--verbose", type=int,
                        nargs='?', const=False, default=1,
                        help="Whether to enable verbosity of output")
    parser.add_argument('--factivation', type=str, required=False,
                        default="relu",
                        help="Choose: relu, leaky")
    parser.add_argument('--multigpu', type=int, required=False,
                        default=0,
                        help='number of gpus. 0=1')

    args = parser.parse_args()
    verbose = args.verbose
    dropout = args.dropout
    dropout0 = args.dropout0
    datadir = args.datadir
    dbbasedir = args.dbbasedir
    infodir = args.infodir
    prefix = args.prefix
    epochs = args.epochs
    extraepochs = args.extraepochs
    batchsize = args.bs
    nclasses = args.nclasses
    ndense = args.ndense
    casenet = args.casenet
    lr = args.lr
    tdim = args.tdim
    margin = args.margin
    dynmargin = args.dynmargin
    optimizer = args.optimizer
    tuning = args.tuning
    experdirbase = args.experdir
    modality = args.mod
    use3D = args.use3d
    softlabel= args.softlabel
    IS_DEBUG = args.debug
    hardnegs = args.hn
    postriplet = args.postriplet
    freeze_all = args.freezeall
    nofreeze = args.nofreeze
    with_missing = not args.nomissing
    datatype= args.datatype
    initnet = args.initnet #"/home/mjmarin/experiments/tumgaid_uwyhtri/demo__datagen_of+depth_opSGD_vA_mTaskI0.10_nd0062_mg031_bs032_lr0.001000_dr0.00/model-state-0001.hdf5"
    mergefun = args.mergefun
    wid = args.wid
    wver = args.wver
    fActivation = args.factivation
    gaitset = args.gaitset
    repetitions = args.repetitions
    multigpu = args.multigpu

    if freeze_all and nofreeze:
        print("Error: you cannot ask for both freezing and not freezing at the same time!")
        exit(-1)

    # Start the processing
    if tuning:
        run_dir_base = osp.join(experdirbase, "logs/"+prefix+"_hparam_tuning/")

        # Experimental part ====================================
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 1024, 2048]))
        HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.4))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['Adam', 'SGD']))

        METRIC_ACCURACY = 'accuracy'

        with tf.summary.create_file_writer(run_dir_base).as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
            )

        trial = 0
        for optimizer in HP_OPTIMIZER.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer
                    }
                    run_dir = run_dir_base + "run-{:d}".format(trial)
                    with tf.summary.create_file_writer(run_dir).as_default():
                        hp.hparams(hparams)  # record the values used in this trial
                        print("* Starting trial {:d}".format(trial))
                        final_model, experdir, val_acc = trainUWYHGaitNet(datadir=datadir, dbbasedir=dbbasedir,
                                                                          experfix=prefix, lr=lr, dropout=dropout_rate,
                                                                           experdirbase=experdirbase,
                                                                           nclasses=nclasses,
                                                                           optimizer=optimizer, logdir=run_dir,
                                                                           epochs=epochs, batchsize=batchsize,
                                                                           margin=margin,
                                                                           ndense_units=num_units, fActivation=fActivation)
                        tf.summary.scalar(METRIC_ACCURACY, val_acc, step=epochs)
                        trial += 1
    else:
        final_model, experdir, val_acc = trainUWYHGaitNet(datadir=datadir, dbbasedir=dbbasedir, experfix=prefix, lr=lr,
                                                           dropout=dropout, dropout0=dropout0,
                                                          experdirbase=experdirbase,
                                                           nclasses=nclasses, optimizer=optimizer,
                                                           epochs=epochs, batchsize=batchsize, ndense_units=ndense,
                                                           margin=margin, casenet=casenet, dynmargin=dynmargin,
                                                           hardnegs=hardnegs, logdir="", loss_weights=[wver, wid],
                                                           modality=modality, initnet=initnet, use3D=use3D,
                                                          freeze_all=freeze_all, nofreeze=nofreeze,
                                                          softlabel=softlabel, datatype=datatype, infodir=infodir,
                                                          postriplet=postriplet, with_missing=with_missing, mergefun=mergefun,
                                                           extra_epochs=extraepochs, verbose=verbose, fActivation=fActivation, gaitset=gaitset, repetitions=repetitions, multigpu=multigpu)

    final_model.save(osp.join(experdir, "model-final-{:04d}.hdf5".format(epochs)))
    final_model.save_weights(osp.join(experdir, "model-final-{:04d}_weights.hdf5".format(epochs)))

    print("* End of training: {}".format(experdir))
