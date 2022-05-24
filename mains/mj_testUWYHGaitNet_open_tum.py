# Tests a gait recognizer CNN
# This version uses a custom DataGenerator

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'April 2020'

import os
import sys
import numpy as np
import time
import os.path as osp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from os.path import expanduser

import pathlib
maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")
import deepdish as dd
from utils.mj_utils import mj_isDebugging
from sklearn.metrics import confusion_matrix
import statistics
from data.mj_augmentation import mj_mirrorsequence
from data.mj_dataGeneratorMMUWYHsingle import DataGeneratorGaitMMUWYH

# --------------------------------
import tensorflow as tf

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.25  # gpu_rate # TODO
# tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------


def evalUWYHNet_set(model, datadir="matimdbtum_gaid_N150_of25_60x60_lite",
                nclasses=150, tdim=50, modality='gray', use_avg=True,
                ntype=1, use_mods = [1,1,1], typecode=1, mirror=False,
                    input_shape=[(50, 60, 60), (25, 60, 60)], modality0="of", gaitset=False,
                    batchsize=64, experdir=".", singlemod=False, verbose=0):


    #input_shape = [(50, 60, 60), (25, 60, 60)]

    # Data from directory
    allRecords_single_tr = []
    import glob, re

    # datadir = "/home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N155-n-05_06-of25_60x60"
    if verbose > 0:
        print("Preparing test data...")
    all_vid_ids = []
    rec2vid = {}
    lfiles = glob.glob(osp.join(datadir, "*.h5"))
    for ix, file in enumerate(lfiles):
        bfile = osp.basename(file)
        n0 = bfile.split(".")[0]
        if ntype == 1:
            parted = n0.split("_")[1]
            record = re.search("\d+", parted).group(0)
            pair = (int(record), -1)
        else:
            parted = n0.split("-")
            per = parted[0]
            seq = parted[1]
            subseq = parted[2]

            pair = []
            if singlemod:
                pair.append(bfile)
                pair.append(-1)
                pair.append(-1)
            else:
                for m in use_mods:
                    if m == 1:
                        pair.append(bfile)
                    else:
                        pair.append(-1)

        D_ = dd.io.load(file)
        if 'labels' in D_.keys():
            label = D_["labels"]
        else:
            label = D_["label"]
        item = (pair, label)
        allRecords_single_tr.append(item)
        all_vid_ids.append(D_["videoId"])
        if ntype == 1:
            rec2vid[int(record)] = D_["videoId"]
        else:
            rec2vid[bfile] = D_["videoId"]

    if ntype == 1:
        datadirs = [datadir, ""]
    else:
        datadirs = [datadir, datadir.replace("of25", modality+"25"),
                    datadir.replace("of25", "depth25")]

    # Find label mapping for training
    if nclasses > 0:
        allLabels = [r[1] for r in allRecords_single_tr]
        ulabels = np.unique(allLabels)
        # Create mapping for labels
        labmap = {}
        for ix, lab in enumerate(ulabels):
            labmap[int(lab)] = ix
    else:
        labmap = None

    if np.sum(use_mods) == 1 and singlemod:
        input_shape = input_shape[0]
        nmods = np.sum(use_mods)
#    elif gaitset:
#        nmods = 2
    else:
        nmods = 3

    # Create data generator
    val_generator = DataGeneratorGaitMMUWYH(allRecords_single_tr, datadir=datadirs, batch_size=batchsize,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=mj_isDebugging() and ntype==1,
                                            ntype=ntype, nmods=nmods, gaitset=gaitset,
                                            expand_level=1, isTest=True, augmentation_x=0, shuffle=False)

    # Prepare output for model
    if typecode == 1:
        codename = "signature"
    elif typecode == 3:
        codename = "flatten"
    else:
        codename = "code"

    output = model.get_layer(codename).output
    from tensorflow.keras.models import Model
    model_code = Model(inputs=model.input, outputs=output)

    multitask = True
    dist = []
    all_vids = []
    all_gt_labs = []
    all_pred_labs = []
    all_codes = []
    #nbatches = len(val_generator)
    nbatches = int(np.ceil(len(val_generator.list_IDs) / np.float(batchsize)))
    if verbose > 0:
        print("Encoding...")
    for bix in range(nbatches):  # We want the remaining samples as well
        sys.stdout.write("{}/{}\r".format(bix, nbatches))
        sys.stdout.flush()

        tuples, labels, info = val_generator.__getitemwithinfo__(bix, expand=1)

        if multitask:
            labels = labels[0]

        if not singlemod:
            if gaitset:
                of0, uof0, gray0, ug0, depth0, ud0 = tuples
            else:
                of0, uof0, gray0, ug0, depth0, ud0 = tuples

        if mirror:
            of1 = np.empty(shape=of0.shape)
            gray1 = np.empty(shape=gray0.shape)
            depth1 = np.empty(shape=depth0.shape)
            for tix in range(len(of0)):
                of1[tix] = mj_mirrorsequence(of0[tix], True, True)
                gray1[tix] = mj_mirrorsequence(gray0[tix], False, True)
                depth1[tix] = mj_mirrorsequence(depth0[tix], False, True)

            of0 = np.vstack((of0, of1))
            uof0 = np.vstack((uof0, uof0))
            gray0 = np.vstack((gray0, gray1))
            ug0 = np.vstack((ug0, ug0))
            depth0 = np.vstack((depth0, depth1))
            ud0 = np.vstack((ud0, ud0))
            labels = np.vstack((labels, labels))

        if not singlemod:
            if gaitset:
                signatures = model_code.predict([of0, uof0, gray0, ug0, depth0, ud0])
            else:
                signatures = model_code.predict([of0, uof0, gray0, ug0, depth0, ud0])
        else:
            signatures = model_code.predict(tuples)
        gt_labs = labels.astype(np.int)
        l_vids = len(gt_labs) * [0]

        # Find corresponding video-id
        for i, rg in enumerate(info):
            if use_mods[0]:
                rec = rg[0][0]
            elif use_mods[1]:
                rec = rg[0][1]
            else:
                rec = rg[0][2]
            #idx = np.where(Drecs == rec)[0]
            vid = rec2vid[rec] #Dvids[idx[0]]
            l_vids[i] = vid

        all_vids = all_vids + l_vids
        if bix == 0:
            all_gt_labs = gt_labs
            #all_pred_labs = pred_labs
            all_codes = signatures
        else:
            all_gt_labs = np.append(all_gt_labs, gt_labs)
            #all_pred_labs = np.append(all_pred_labs, pred_labs)
            all_codes = np.vstack((all_codes, signatures))

    return all_codes, all_gt_labs, all_vids


def evalUWYHNet(model, datadir_train="matimdbtum_gaid_N150_of25_60x60_lite",
                datadir_test="tfimdb_tum_gaid_N155_test_s01-02_of25_60x60",
                experfix="0",
                nclasses=150, tdim=50, modality='gray', use_avg=True,
                ntype=1, use_mods=[1,1,1], typecode=1,
                knn=7, usemirror=False, modality0="of",
                allcombosgallery=False, gaitset=False,
                batchsize=64, experdir=".", singlemod=False,
                prefix="", verbose=0):

    if gaitset:
        input_shape_ = [(25, 60, 60, 2), (25, 60, 60, 1), (25, 60, 60, 1)]
    else:
        input_shape_ = [(50, 60, 60), (25, 60, 60), (25, 60, 60)]

    if singlemod:
        input_shape = [input_shape_[mod_ix] for mod_ix in range(len(use_mods)) if use_mods[mod_ix]]
    else:
        input_shape = input_shape_

    # Prepare gallery
    experfix = experfix+prefix
    if allcombosgallery:
        outfile_gal = osp.join(experdir, "codes_gallery_{}_Mall_t{}_mir{}.h5".format(experfix, typecode, int(usemirror)))
    else:
        outfile_gal = osp.join(experdir, "codes_gallery_{}_M{}{}{}_t{}_mir{}.h5".format(experfix, int(use_mods[0]),
                                                                                        int(use_mods[1]),int(use_mods[2]),
                                                                                   typecode, int(usemirror)))
    save_gal = True
    if osp.exists(outfile_gal):
        exper = dd.io.load(outfile_gal)
        print("Gallery loaded from file: "+outfile_gal)
        all_codes_gallery = exper["codes"]
        all_gt_labs_gallery = exper["gtlabs"]
        all_vids_gallery = exper["vids"]

        save_gal = False
    else:
        if usemirror:
            print(" - Mirror samples will be added to gallery.")
        if not allcombosgallery:
            all_codes_gallery, all_gt_labs_gallery, all_vids_gallery = evalUWYHNet_set(model, datadir_train,
                                                                                   nclasses, tdim, modality,
                                                                                   use_avg,
                                                                                   ntype, use_mods,
                                                                                   typecode, usemirror,
                                                                                   input_shape, modality0, gaitset,
                                                                                   batchsize, experdir, singlemod, verbose)
        else:
            l_combos = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]  # TODO more combos
            all_codes_gallery = []
            all_gt_labs_gallery = []
            all_vids_gallery = []
            for cmb in l_combos:
                print("Processing combo: {}".format(cmb))
                all_codes_gallery_c, all_gt_labs_gallery_c, all_vids_gallery_c = evalUWYHNet_set(model, datadir_train,
                                                                                   nclasses, tdim, modality,
                                                                                   use_avg,
                                                                                   ntype, cmb,
                                                                                   typecode, usemirror,
                                                                                   input_shape, modality0, gaitset,
                                                                                   batchsize, experdir, singlemod, verbose)
                if all_codes_gallery == []:
                    all_codes_gallery = all_codes_gallery_c
                    all_gt_labs_gallery = all_gt_labs_gallery_c
                    all_vids_gallery = all_vids_gallery_c
                else:
                    all_gt_labs_gallery = np.append(all_gt_labs_gallery, all_gt_labs_gallery_c)
                    all_codes_gallery = np.vstack((all_codes_gallery, all_codes_gallery_c))
                    all_vids_gallery = all_vids_gallery + all_vids_gallery_c

    # Save data, just in case
    if save_gal:
        exper = {}
        exper["codes"] = all_codes_gallery
        exper["gtlabs"] = all_gt_labs_gallery
        exper["vids"] = all_vids_gallery
        dd.io.save(outfile_gal, exper)
        print("Data saved to: "+outfile_gal)

    # ---------------------
    # Prepare test samples
    # ---------------------
    all_codes_test, all_gt_labs_test, all_vids_test = evalUWYHNet_set(model, datadir_test,
                                                                      nclasses, tdim, modality, use_avg,
                                                                      ntype, use_mods, typecode, False,
                                                                      input_shape, modality0, gaitset,
                                                                      batchsize, experdir, singlemod, verbose)

    # Save data, just in case
    if not osp.exists(experdir):
        os.makedirs(experdir)

    outfile = osp.join(experdir, "codes_probe.h5")
    exper = {}
    exper["codes"] = all_codes_test
    exper["gtlabs"] = all_gt_labs_test
    exper["vids"] = all_vids_test
    dd.io.save(outfile, exper)
    if verbose > 0:
        print("Data saved to: "+outfile)

    if verbose > 0:
        print("# Preparing NN classifier.")
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=knn)

    clf.fit(all_codes_gallery, all_gt_labs_gallery)
    if verbose > 0:
        print(all_codes_gallery.shape)

    print("# Predicting with {}-NN classifier.".format(knn))
    start = time.time()
    all_pred_labs = clf.predict(all_codes_test)
    end = time.time()
    print("- Elapsed time: {}".format(end - start))
    if verbose > 0:
        print(all_codes_test.shape)

    # score = clf.score(all_codes_test, all_gt_labs_test)
    # print("*** Score at subseq level: {}".format(100*score))

    M = confusion_matrix(all_gt_labs_test, all_pred_labs)
    acc = M.diagonal().sum()/len(all_gt_labs_test)
    print("*** Accuracy [subseq]: {:.2f}".format(acc*100))

    # # Summarize per video: gallery
    uvids = np.unique(all_vids_gallery)
    # Majority voting per video
    all_codes_per_vid_gallery = []
    all_gt_labs_per_vid_gallery = []
    for vix in uvids:
        idx = np.where(all_vids_gallery ==vix)[0]

        vid_logits = all_codes_gallery[idx,]

        if use_avg:
            lg = vid_logits.mean(axis=0)
        else:
            # Alternative
            lg = vid_logits.max(axis=0)

        if all_codes_per_vid_gallery == []:
            all_codes_per_vid_gallery = lg
        else:
            all_codes_per_vid_gallery = np.vstack((all_codes_per_vid_gallery, lg))

        try:
            gt_lab_vid = statistics.mode(all_gt_labs_gallery[idx])
        except:
            gt_lab_vid = all_gt_labs_gallery[idx][0]

        all_gt_labs_per_vid_gallery.append(gt_lab_vid)

    # # Summarize per video
    uvids = np.unique(all_vids_test)

    # Majority voting per video
    all_codes_per_vid_test = []
    all_gt_labs_per_vid = []
    all_pred_labs_per_vid = []
    all_pred_labs_per_vid2 = []
    for vix in uvids:
        idx = np.where(all_vids_test ==vix)[0]

        vid_logits = all_codes_test[idx,]

        if use_avg:
            lg = vid_logits.mean(axis=0)
        else:
            # Alternative
            lg = vid_logits.max(axis=0)

        pred_lab_vid2 = 0

        if all_codes_per_vid_test == []:
            all_codes_per_vid_test = lg
        else:
            all_codes_per_vid_test = np.vstack((all_codes_per_vid_test, lg))

        try:
            gt_lab_vid = statistics.mode(all_gt_labs_test[idx])
        except:
            gt_lab_vid = all_gt_labs_test[idx][0]

        try:
            pred_lab_vid = statistics.mode(all_pred_labs[idx])
        except:
            pred_lab_vid = all_pred_labs[idx][0]

        all_gt_labs_per_vid.append(gt_lab_vid)
        all_pred_labs_per_vid.append(pred_lab_vid)
        all_pred_labs_per_vid2.append(pred_lab_vid2)
    #
    #     # At subsequence level
    # M = confusion_matrix(all_gt_labs, all_pred_labs)
    # acc = M.diagonal().sum()/len(all_gt_labs)
    # print("*** Accuracy [subseq]: {:.2f}".format(acc*100))
    #
    # # import pdb; pdb.set_trace()

    # Classifier at video level
    if verbose > 0:
        print("# Preparing NN classifier at video-level (merged).")
    clf = KNeighborsClassifier(n_neighbors=knn)
    clf.fit(all_codes_per_vid_gallery, all_gt_labs_per_vid_gallery)
    if verbose > 0:
        print(all_codes_per_vid_gallery.shape)

    print("# Predicting with {}-NN classifier (merged codes).".format(knn))
    start = time.time()
    all_pred_labs_vid = clf.predict(all_codes_per_vid_test)
    end = time.time()
    print("- Elapsed time: {}".format(end - start))
    if verbose > 0:
        print(all_codes_per_vid_test.shape)

    score = clf.score(all_codes_per_vid_test, all_gt_labs_per_vid)
    print("*** Score at video level (merged): {:.2f}".format(100*score))

    # At video level
    Mvid = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid)
    acc_vid = Mvid.diagonal().sum() / len(all_gt_labs_per_vid)
    print("*** Accuracy [video]: {:.2f}".format(acc_vid*100))

    # # At video level
    # Mvid2 = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid2)
    # acc_vid2 = Mvid2.diagonal().sum() / len(all_gt_labs_per_vid)
    # print("*** Accuracy-avg [video]: {:.2f}".format(acc_vid2*100))
    #
    # print("Tested on: "+datadir)
    summary = (acc, acc_vid, score)

    return M, Mvid, summary


################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

    parser.add_argument('--datadir', type=str, required=False,
                        default='PATH/TUM_GAID_tf/tfimdb_tum_gaid_N155_test_s01-02_of25_60x60',
                        help="Full path to data directory")

    parser.add_argument('--datadirtrain', type=str, required=False,
                        default='PATH/TUM_GAID_tf/tfimdb_tum_gaid_N155_ft_of25_60x60',
                        help="Full path to data gallery directory")

    parser.add_argument('--model', type=str, required=True,
                        default='PATH/experiments/tumgaid_mj_tf/model-state-0002.hdf5',
                        help="Full path to model file (DD: .hdf5)")

    parser.add_argument('--prefix', type=str, required=False,
                        default="",
                        help="String to prefix experiment directory name.")
    parser.add_argument('--bs', type=int, required=False,
                        default=64,
                        help='Batch size')
    parser.add_argument('--knn', type=int, required=False,
                        default=7,
                        help='k in kNN')
    parser.add_argument('--nclasses', type=int, required=True,
                        default=150,
                        help='Maximum number of epochs')
    parser.add_argument('--tdim', type=int, required=False,
                        default=50,
                        help='Number of dimensions in 3rd axis time. E.g. OF=50')
    parser.add_argument('--modality', type=str, required=False,
                        default="gray",
                        help="gray|depth")
    parser.add_argument('--modality0', type=str, required=False,
                        default="of",
                        help="of|gray|depth")
    parser.add_argument('--nametype', type=int, required=False,
                        default=2,
                        help='Type of filenames: {1,2}')
    parser.add_argument("--verbose", type=int,
                        nargs='?', const=False, default=1,
                        help="Whether to enable verbosity of output")

    parser.add_argument("--useavg", type=int,
                        nargs='?', const=False, default=1,
                        help="Use average logits?")

    parser.add_argument("--usemirror", type=int,
                        nargs='?', const=False, default=0,
                        help="Use mirror samples in gallery?")

    parser.add_argument("--typecode", type=int,
                        nargs='?', const=False, default=1,
                        help="1=signature-layer; 2=code-layer (shorter)")

    parser.add_argument("--usemod1", type=int,
                        nargs='?', const=False, default=1,
                        help="Use first modality?")

    parser.add_argument("--usemod2", type=int,
                        nargs='?', const=False, default=1,
                        help="Use second modality?")
    parser.add_argument("--usemod3", type=int,
                        nargs='?', const=False, default=1,
                        help="Use third modality?")
    parser.add_argument('--allcombos', default=False, action='store_true', help="Create gallery with all combos?")

    parser.add_argument('--allcombostest', default=False, action='store_true', help="Evaluate all possible combos?")

    parser.add_argument('--gaitset', default=False, action='store_true', help="Gaitset model?")
    parser.add_argument('--singlemod', default=False, action='store_true', help="Single model?")

    args = parser.parse_args()
    verbose = args.verbose
    datadir = args.datadir
    datadirtrain = args.datadirtrain
    prefix = args.prefix
    batchsize = args.bs
    nclasses = args.nclasses
    tdim = args.tdim
    modelpath = args.model
    use_avg = args.useavg > 0
    nametype = args.nametype
    usemod1 = args.usemod1 > 0
    usemod2 = args.usemod2 > 0
    usemod3 = args.usemod3 > 0
    usemirror = args.usemirror > 0
    modality = args.modality
    modality0 = args.modality0
    typecode = args.typecode
    knn = args.knn
    allcombosgallery = args.allcombos
    allcombostest = args.allcombostest
    gaitset = args.gaitset
    singlemod = args.singlemod

    # Load the model
    from nets.mj_uwyhNets_ba import UWYHSemiNet

    if mj_isDebugging():
        model = None
        filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
        filters_numbers = [96, 192, 512, 512]
        model = UWYHSemiNet.build([(50,60,60), (25,60,60)], 4, filters_size, filters_numbers, 64, nclasses=155)
    else:
        model = UWYHSemiNet.loadnet(modelpath)

    if model is not None and verbose > 0:
        model.summary()

    experdir = osp.dirname(modelpath)
    modelcase = os.path.basename(modelpath).split("-")[2].split(".")[0]
    experfix = "st"+modelcase

    if not allcombostest:
        # Call the evaluator
        CM = evalUWYHNet(model, datadir_train=datadirtrain, datadir_test=datadir, experfix=experfix,
                         nclasses=nclasses, tdim=tdim, batchsize=batchsize, use_avg=use_avg,
                         modality=modality,
                         use_mods=[usemod1, usemod2, usemod3], typecode=typecode,
                         knn= knn, usemirror=usemirror, modality0=modality0,
                         allcombosgallery=allcombosgallery,
                         experdir=experdir, ntype=nametype, gaitset=gaitset,
                         prefix=prefix, singlemod=singlemod, verbose=verbose)

        # Save CM
        outfile = osp.join(experdir, "evaluation_s{}_M{}{}{}_K{}_t{}.h5".format(modelcase, int(usemod1), int(usemod2),
                                                                                int(usemod3), knn,
                                                                              typecode))
        dd.io.save(outfile, CM)
    else:
        l_combos = [[0, 0, 1], [0, 1, 0], [1, 0, 0],
                    [0, 1, 1], [1, 0, 1], [1, 1, 0],
                    [1, 1, 1]]
        lresults = []
        for c in l_combos:
            print("+*+ Combo: {}".format(c))
            usemod1 = c[0]
            usemod2 = c[1]
            usemod3 = c[2]
            # Call the evaluator
            l_res = evalUWYHNet(model, datadir_train=datadirtrain, datadir_test=datadir, experfix=experfix,
                             nclasses=nclasses, tdim=tdim, batchsize=batchsize, use_avg=use_avg,
                             modality=modality,
                             use_mods=[usemod1, usemod2, usemod3], typecode=typecode,
                             knn=knn, usemirror=usemirror, modality0=modality0,
                             allcombosgallery=allcombosgallery,
                             experdir=experdir, ntype=nametype,
                                prefix=prefix, verbose=verbose)

            lresults.append(l_res[2])
            # Save CM
            outfile = osp.join(experdir,
                               "evaluation_s{}_M{}{}{}_K{}_t{}.h5".format(modelcase, int(usemod1), int(usemod2),
                                                                          int(usemod3), knn,
                                                                          typecode))
            dd.io.save(outfile, l_res)
        scenario = os.path.basename(datadir).split("_")[5]
        outfilecombos = osp.join(experdir, "evaluation_s{}_{}_Mcombos_K{}_t{}.h5".format(modelcase, scenario, knn, typecode))
        dd.io.save(outfilecombos, lresults)
        print("File saved: "+outfilecombos)

    print("Done!")
