# Trains a gait recognizer CNN
# This version uses a custom DataGenerator

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

import sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np

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
config.gpu_options.per_process_gpu_memory_fraction = 0.43  # gpu_rate # TODO

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
from tensorflow.keras.callbacks import ReduceLROnPlateau

from data.mj_dataGeneratorMMUWYHsingle import DataGeneratorGaitMMUWYH
from utils.mj_netUtils import mj_findLatestFileModel
from data.mj_datasetinfo import DatasetInfoTUM

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
    perc = 0.09
    nval = int(perc * nvids)
    vids_tr = [uvids[i] for i in range(0, nvids - nval)]
    vids_val = [uvids[i] for i in range(nvids - nval, nvids)]
    allRecords_single_tr = []
    for vix in vids_tr:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_tr = allRecords_single_tr + this_vid_pairs

    if len(np.unique([l[1] for l in allRecords_single_tr])) < nclasses:
        print("More classes needed!")
        import pdb;
        pdb.set_trace()

    allRecords_single_val = []
    for vix in vids_val:
        idx = np.where(all_vids == vix)[0]
        this_vid_pairs = [allPairs[ix] for ix in idx]  # [all_keys[ix] for ix in idx]
        allRecords_single_val = allRecords_single_val + this_vid_pairs

    return allRecords_single_tr, allRecords_single_val, rec2gait


def mj_computeDistMetrics(model, val_generator, multitask, lab4color):
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

        of0 = tuples
        # labs = np.squeeze(labs)

        #codes0 = UWYHSemiNet.encode(model, of0)
        codes0 = model.predict(of0)

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

        # these_sprites = [gray0[ix, 13] for ix in range(gray0.shape[0])]
        # l_sprites = l_sprites + these_sprites

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


def trainUWYHGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", dbbasedir='/home/GAIT_local/SSD', experfix="demo",
                     nclasses=0, lr=0.001, dropout=0.4, dropout0=-1,
                     experdirbase=".", epochs=5, batchsize=32, optimizer="SGD",
                     ndense_units=2048, margin=0.5, savemodelfreq=2, casenet='B',
                     dynmargin=False, hardnegs=4, loss_weights=None,
                     modality="of",
                     initnet="", use3D=False, softlabel=False,
                     datatype = 1, infodir="", freeze_all=False, nofreeze=False,
                     logdir="", extra_epochs=0, verbose=0):
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

    if modality == 'of':
        input_shape1 = (50, 60, 60)
    else:
        if use3D:
            input_shape1 = (25, 60, 60, 1)
        else:
            input_shape1 = (25, 60, 60)

    input_shape = input_shape1

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
        else:
            optimfun = eval("optimizers." + optimizer + "(lr=initialLR)")

    # with graph.as_default():
    #     with session.as_default():

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.00001)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

    if use3D:
        infix = "_" + modality + "3D" + infix
    else:
        infix = "_" + modality + infix

    if datatype == 2:
        infix = infix + "_dt2"

    if softlabel:
        infix = infix + "_sl"
        smoothlabval = 0.1
    else:
        smoothlabval = 0

    if casenet != 'B':
        infix = infix + "_v"+casenet

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

    # Prepare model
    pattern_file = "model-state-{:04d}.hdf5"
    previous_model = mj_findLatestFileModel(experdir, pattern_file, epoch_max=epochs)
    print(previous_model)
    initepoch = 0
    if previous_model != "":
        pms = previous_model.split("-")
        initepoch = int(pms[len(pms) - 1].split(".")[0])
        print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
        from tensorflow.keras.models import load_model
        #model = load_model(previous_model, custom_objects={"TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss()})
        model = UWYHSemiNet.loadnet(previous_model)
    else:
        if initnet != "":
            print("* Model will be init from: "+initnet)

        model = UWYHSemiNet.build_or_load(input_shape, number_convolutional_layers,
                                   filters_size, filters_numbers, ndense_units, weight_decay, [dropout0, dropout],
                                   optimizer=optimfun, margin=margin, nclasses=nclasses, loss_weights=loss_weights,
                                   initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, smoothlabels=smoothlabval,
                                          freeze_all=freeze_all)

    model.summary()

#    plot_model(model, to_file=osp.join(experdir, 'model.png'))
    #model.save(osp.join(experdir, "model-init-{:04d}.hdf5".format(00)))

    # Tensorboard
    if logdir == "":
        logdir = experdir
        # Save checkpoint
        chkptname = osp.join(logdir, "model-weights-{epoch:02d}x{val_loss:.2f}_weights.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(chkptname, save_best_only=True, save_weights_only=True)

        from tensorflow.keras.callbacks import TensorBoard

        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=2, write_graph=True, write_images=False,
                                  profile_batch = 5)
        callbacks = [reduce_lr, tensorboard, checkpoint, es_callback]
    else: # This case is for parameter tuning
        # Save checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(logdir, save_best_only=True)

        from tensorflow.keras.callbacks import TensorBoard
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
                                  profile_batch = 3)

        # hpcallback = hp.KerasCallback("/tmp/mjmarin/logs/hparam_tuning", hparams)
        hpcall = hp.KerasCallback(logdir, hparams)
        callbacks = [reduce_lr, tensorboard, checkpoint, hpcall]

    # ---------------------------------------
    # Prepare data
    # ---------------------------------------

    if verbose > 0:
        print("Preparing training/val splits...")

    rec2gait = []
    rec2vid = []
    if datatype == 2:
        if infodir == "":
            h5dir = "/home/mjmarin/databases/tumgaid/h5"
        else:
            h5dir = infodir
        if nclasses == 155:
            infofile = osp.join(h5dir, "tfimdb_tum_gaid_N155_ft_of25_60x60_info_of+{}.h5".format(modality))
        else:
            #infofile = osp.join(h5dir, "tfimdb_tum_gaid_N150_train_of25_60x60_info_of+{}.h5".format(modality))
            if modality == "depth":
                infofile = osp.join(h5dir, "tfimdb_tum_gaid_N150_train_of25_60x60_info_of+depth.h5")
            else:
                infofile = osp.join(h5dir, "tfimdb_tum_gaid_N150_train_of25_60x60_info_of+{}.h5".format(modality))
        Iof = dd.io.load(infofile)
#        Igray = dd.io.load(osp.join("tfimdb_tum_gaid_N{}_train_{}25_60x60.h5".format(nclasses, modality)))

        allRecords_single_tr, allRecords_single_val, rec2gait = mj_splitTrainValGaitByInfo(Iof)
    else:
        if nclasses == 155:
            pairspath = osp.join("/home/mjmarin/databases/tumgaid/dd_files",
                                 "matimdbtum_gaid_N155_of_{}_triplets_val.h5".format(modality))

            allPairs = dd.io.load(pairspath)
            np.random.shuffle(allPairs)   # Randomize

            ntotal = len(allPairs)
            perc = 0.06
            nval = int(perc * ntotal)

            idx_tr = slice(ntotal - nval)
            allRecords_tr = allPairs[idx_tr]
            idx_val = slice(ntotal - nval, ntotal)
            allRecords_val = allPairs[idx_val]
        else:
            pairspath_tr = osp.join("/home/mjmarin/databases/tumgaid/dd_files",
                                 "matimdbtum_gaid_N150_of_{}_triplets_train.h5".format(modality))
            allRecords_tr = dd.io.load(pairspath_tr)

            pairspath_val = osp.join("/home/mjmarin/databases/tumgaid/dd_files",
                                 "matimdbtum_gaid_N150_of_{}_triplets_val.h5".format(modality))
            allRecords_val = dd.io.load(pairspath_val)

        # Prepare data in expected format (record, label)
        allRecords_single_tr = [(r[0], r[3])  for r in allRecords_tr]
        allRecords_single_val = [(r[0], r[3]) for r in allRecords_val]

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
    if infodir == "":
        dd_dir = "/home/mjmarin/databases/tumgaid/dd_files"
    else:
        dd_dir = infodir
    if nclasses == 155:
        dbfilepath = osp.join(dd_dir, "matimdbtum_gaid_N155_of25_60x60.npy")
    else:
        dbfilepath = osp.join(dd_dir, "matimdbtum_gaid_N150_of25_60x60.npy")
    # D = np.load(dbfilepath)
    # rec2lab = {int(D[ix,0]) : int(D[ix,1]) for ix in range(D.shape[0])}


    # Gait type for each sample
    if datatype == 1:
        db = DatasetInfoTUM(dbfilepath)
        rec2lab = db.rec2lab
        gait_tr = [db.gaitofsample(t[0][0]) for t in allRecords_single_tr]
        gait_val = [db.gaitofsample(t[0][0]) for t in allRecords_single_val]
    else:
        gait_tr = [rec2gait[r[0][0]] for r in allRecords_single_tr]
        gait_val = [rec2gait[r[0][0]] for r in allRecords_single_val]

    # Define directories
    if datatype == 1:
        if nclasses == 155:
            datadirs = [osp.join(dbbasedir, "matimdbtum_gaid_N155_of25_60x60"),
                        osp.join(dbbasedir, "matimdbtum_gaid_N155_{}25_60x60".format(modality))]
        else:
            datadirs = [osp.join(dbbasedir, "matimdbtum_gaid_N150_of25_60x60"),
                        osp.join(dbbasedir, "matimdbtum_gaid_N150_{}25_60x60".format(modality))]
    else:
        if nclasses == 155:
            datadirs = [osp.join(dbbasedir, "tfimdb_tum_gaid_N155_ft_of25_60x60"),
                        osp.join(dbbasedir, "tfimdb_tum_gaid_N155_ft_{}25_60x60".format(modality))]
        else:
            datadirs = [osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_of25_60x60"),
                        osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_{}25_60x60".format(modality))]

            datadirs[0] = osp.join(dbbasedir, "tfimdb_tum_gaid_N150_train_{}25_60x60".format(modality))

    # MJ: augment training data
    train_generator = DataGeneratorGaitMMUWYH(allRecords_single_tr+allRecords_single_tr+allRecords_single_tr,
                                              datadir=datadirs, batch_size=batchsize,
                                          dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                              ntype=datatype,
                                           gait=gait_tr+gait_tr+gait_tr,
                                              expand_level=3, use3D=use3D, nmods=1)

    val_generator = DataGeneratorGaitMMUWYH(allRecords_single_val, datadir=datadirs, batch_size=batchsize,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                            ntype=datatype,
                                           gait=gait_val, expand_level=2, use3D=use3D, augmentation_x=0, nmods=1)

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
                 'loss_weights': loss_weights}
    dd.io.save(osp.join(experdir, "model-config.hdf5"), modelpars)

    # Validation labels mapping for projector
    all_val_labs = [t[1] for t in allRecords_single_val]
    ulabs_val = np.unique(all_val_labs)
    lab4color = {l: ix for ix, l in enumerate(ulabs_val)}

    if multitask:
        lab4color = labmap

    # ---------------------------------------
    # Train model
    # --------------------------------------
    print("* Starting training...")
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
            model, hist = UWYHSemiNet.fit_generator(model, ecum+epochs_, callbacks, train_generator, val_generator,
                                                    ecum, None, None) #ecum, steps_per_epoch, validation_steps)
            #ecum += epochs_
            ecum += len(hist.epoch)
            model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(ecum)))

            # Save in such a way that can be recovered from different Python versions
            model.save_weights(osp.join(experdir, "model-state-{:04d}_weights.hdf5".format(ecum)))
            # checkpoint = tf.train.Checkpoint(model=model)
            # checkpoint.save(file_prefix=osp.join(experdir, "ckpt"))
            last_lr = hist.history["lr"][-1]

            # Export some filters to tensorboard
            if modality == "of":
                mj_save_filters(model, osp.join(experdir, "metrics"), step=ecum + epochs_, branchname="ofBranch", title="filters-of")
            elif use3D:
                mj_save_filters3d(model, osp.join(experdir, "metrics"), step=ecum + epochs_, branchname="ofBranch",
                                  title="filters-" + modality + "-3d")
            else:
                mj_save_filters(model, osp.join(experdir, "metrics"), step=ecum + epochs_, branchname="ofBranch",
                                title="filters-"+modality)

            # if not use3D:
            #     mj_save_filters(model, osp.join(experdir, "metrics"), step=ecum + epochs_, branchname="grayBranch",
            #                     title="filters-"+modality)
            # else:
            #     mj_save_filters3d(model, osp.join(experdir, "metrics"), step=ecum + epochs_, branchname="grayBranch",
            #                     title="filters-"+modality+"-3d")

            # Compute EER metric on VALIDATION data
            # -------------------------------------
            if verbose > 0:
                print("+ Computing metrics...", flush=True)

            distances, eer_val, chance, data_metric = mj_computeDistMetrics(model, val_generator, multitask, lab4color)
            all_codes0, all_labs0 = data_metric  # Unpack

            # Save to tensorboard scalars
            with tf.summary.create_file_writer(osp.join(experdir, "metrics")).as_default():
                tf.summary.scalar('eer_val', data=eer_val, step=ecum + epochs_)
                tf.summary.histogram('dist_hist_val', data=distances, step=ecum + epochs_)

            #chance = np.sum(gt_labels > 0) / len(gt_labels) # Compute chance
            print("EER-val: {:.6f} [{:.2f}]".format(eer_val, 1.0-chance), flush=True)
            del distances

            # Save codes to Projector
            print("Exporting to projector", flush=True)
            prjdir = osp.join(experdir, "metrics")
            EMBEDDINGS_FPATH = osp.join(prjdir, EMBEDDINGS_TENSOR_NAME + '.ckpt')
            STEP = ecum+epochs_

            step_val = 3  # FIXME
            sprite_path = None
            imsize = (0, 0)
            # sprite_path, imsize = mj_save_sprite(l_sprites, "sprite.jpg", prjdir)
            # del l_sprites

            mj_register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, prjdir, sprite_path, imsize)

            # Subsample to save space and time
            all_labs0 = [all_labs0[i] for i in range(0, len(all_labs0), step_val)]
            all_codes0 = all_codes0[range(0, len(all_codes0), step_val)]

            mj_save_labels_tsv(all_labs0, META_DATA_FNAME, prjdir)
            tensor_embeddings = tf.Variable(all_codes0, name=EMBEDDINGS_TENSOR_NAME)
            saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
            saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)
            del all_codes0

            tf.keras.backend.clear_session() # Experimental use, see if it solves memory leak

            # Stop if accurary is almost perfect
            if "classprob_acc" in hist.history.keys():
                tr_accuracy = hist.history["classprob_acc"][-1]
                if ecum >= epochs/2 and tr_accuracy > 0.990: # DEVELOP!
                    print("* Training accuracy is very high: {}. Stop!".format(tr_accuracy*100))
                    epochs = ecum
                    break

        epochs = ecum # Final number of processed epochs 
    else:
        model, hist = UWYHSemiNet.fit_generator(model, epochs, callbacks, train_generator, val_generator,
                                                 initepoch, steps_per_epoch, validation_steps)

    print("Last epoch: {}".format(epochs))
    model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(epochs)))
    model.save_weights(osp.join(experdir, "model-state-{:04d}_weights.hdf5".format(epochs)))

    # Get val accuracy from history
    if 'hist' in locals():
        if "classprob_acc" in hist.history.keys():
            try:
                #accuracy = hist.history["val_accuracy"][-1]
                accuracy = hist.history["val_classprob_acc"][-1]
            except:
                accuracy = hist.history["classprob_acc"][-1]
        else:
            accuracy = 0.5
    else:
        accuracy = 0.5

    # Fine-tune on remaining validation samples
    if extra_epochs > 0:
        if verbose > 0:
            print("Adding validation samples to training and run for few epochs...")
        del train_generator

        train_generator = DataGeneratorGaitMMUWYH(allRecords_single_val+allRecords_single_tr, datadir=datadirs, batch_size=batchsize,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging(),
                                        gait=gait_val+gait_tr, expand_level=3, use3D=use3D, nmods=1,
                                                  ntype=datatype)

        ft_epochs = epochs + extra_epochs    # DEVELOP!
        if nclasses == 150:
            new_lr = (10**math.ceil(math.log10(last_lr))) * 0.1            # DEVELOP!
        else:
            new_lr = min(10**math.ceil(math.log10(last_lr)), last_lr)  # last_lr

        model, hist = UWYHSemiNet.fit_generator(model, ft_epochs, callbacks, train_generator, val_generator,
                                                epochs, None, None, new_lr=new_lr)
        # Update num epochs
        epochs = ft_epochs

    model.save(osp.join(experdir, "model-state-{:04d}.hdf5".format(epochs)))

    return model, experdir, accuracy


################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Trains a CNN for gait')
    parser.add_argument('--tuning', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")
    parser.add_argument('--softlabel', default=False, action='store_true', help="Use soft labels?")
    parser.add_argument('--freezeall', default=False, action='store_true', help="Freeze all weights?")
    parser.add_argument('--nofreeze', default=False, action='store_true', help="Avoid freezing any weight?")
    parser.add_argument('--dropout', type=float, required=False,
                        default=0.5, help='Dropout value for after-fusion layers')
    parser.add_argument('--dropout0', type=float, required=False,
                        default=-1, help='Dropout value for 2D branches')
    parser.add_argument('--lr', type=float, required=False,
                        default=0.001,
                        help='Starting learning rate')
    parser.add_argument('--datadir', type=str, required=False,
                        default=osp.join('PATH/datatum', 'matimdbtum_gaid_N150_of25_60x60_lite/'),
                        help="Full path to data directory")
    parser.add_argument('--dbbasedir', type=str, required=False,
                        default='PATH',
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
                        help="Input modality: of, gray, depth")

    parser.add_argument('--datatype', type=int, required=False,
                        default=2,
                        help='Code 1 (matlab-h5) or 2 (new h5)')
    parser.add_argument('--wid', type=float, required=False,
                        default=0,
                        help='Weight for identification task')
    parser.add_argument('--wver', type=float, required=False,
                        default=1.0,
                        help='Weight for verification task')
    parser.add_argument('--initnet', type=str, required=False,
                        default="",
                        help="Path to net to initialize")
    parser.add_argument("--verbose", type=int,
                        nargs='?', const=False, default=1,
                        help="Whether to enable verbosity of output")
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
    freeze_all = args.freezeall
    nofreeze = args.nofreeze
    datatype= args.datatype
    initnet = args.initnet #"/home/mjmarin/experiments/tumgaid_uwyhtri/demo__datagen_of+depth_opSGD_vA_mTaskI0.10_nd0062_mg031_bs032_lr0.001000_dr0.00/model-state-0001.hdf5"

    wid = args.wid
    wver = args.wver

    if wid > 0:
        l_wei = [wver, wid]
    else:
        l_wei = None

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
                                                                           ndense_units=num_units)
                        tf.summary.scalar(METRIC_ACCURACY, val_acc, step=epochs)
                        trial += 1
    else:
        final_model, experdir, val_acc = trainUWYHGaitNet(datadir=datadir, dbbasedir=dbbasedir, experfix=prefix, lr=lr,
                                                           dropout=dropout, dropout0=dropout0,
                                                          experdirbase=experdirbase,
                                                           nclasses=nclasses, optimizer=optimizer,
                                                           epochs=epochs, batchsize=batchsize, ndense_units=ndense,
                                                           margin=margin, casenet=casenet, dynmargin=dynmargin,
                                                           hardnegs=hardnegs, logdir="", loss_weights=l_wei,
                                                           modality=modality,
                                                          initnet=initnet, use3D=use3D, freeze_all=freeze_all,
                                                          nofreeze=nofreeze,
                                                          softlabel=softlabel, datatype=datatype, infodir=infodir,
                                                           extra_epochs=extraepochs, verbose=verbose)

    final_model.save(osp.join(experdir, "model-final-{:04d}.hdf5".format(epochs)))
    final_model.save_weights(osp.join(experdir, "model-final-{:04d}_weights.hdf5".format(epochs)))

    print("* End of training: {}".format(experdir))
