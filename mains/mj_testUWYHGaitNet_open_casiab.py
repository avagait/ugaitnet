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
import pathlib
maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")

import glob, re
import deepdish as dd
from sklearn.metrics import confusion_matrix
from data.mj_augmentation import mj_mirrorsequence
from data.mj_dataGeneratorMMUWYHsingle_repetitions import DataGeneratorGaitMMUWYH

# --------------------------------
import tensorflow as tf
from scipy import stats

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.45  # gpu_rate
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------

def clear_files(lfiles):
    lfiles_ = []
    for file in lfiles:
        data_i = dd.io.load(file)
        if len(data_i["data"]) > 0:
            lfiles_.append(file)

    return lfiles_

def evalUWYHNet_set(model, datadir="matimdbtum_gaid_N150_of25_60x60_lite",
                nclasses=150, tdim=50, modality='gray', use_avg=True,
                ntype=1, use_mod1=1, use_mod2=1, typecode=1, mirror=False,
                    input_shape=[(50, 60, 60), (25, 60, 60)], modality0="of",
                    batchsize=64, experdir=".", multimodal=True, verbose=0, gaitset=False):


    #input_shape = [(50, 60, 60), (25, 60, 60)]
    if not multimodal:
        nmods = 1
    else:
        nmods = len(input_shape)

    # Data from directory
    allRecords_single_tr = []

    # datadir = "/home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N155-n-05_06-of25_60x60"
    if verbose > 0:
        print("Preparing test data [{},{}]...".format(use_mod1, use_mod2))
    all_vid_ids = []
    all_cam_ids = []
    all_gait_ids = []
    rec2vid = {}
    rec2cam = {}
    lfiles = glob.glob(osp.join(datadir.replace("{}25".format("of"), "{}25".format(modality)), "*.h5"))

    lfiles = clear_files(lfiles)
    #lfiles = [lfiles[i] for i in range(0, 1000, 5)]   # TODO
    print("nmods: ", nmods, flush=True)
    print("len lfiles: ", len(lfiles), flush=True)
    if verbose > 1:
        print("Found {} files".format(len(lfiles)))
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
            if use_mod1 and use_mod2:
                pair = (bfile, bfile)
            if use_mod1 and not use_mod2:
                pair = (bfile, -1)
            if not use_mod1 and use_mod2:
                pair = (-1, bfile)

        D_ = dd.io.load(file)
        if 'labels' in D_.keys():
            label = D_["labels"]
        else:
            label = D_["label"]
        item = (pair, label)
        allRecords_single_tr.append(item)
        all_vid_ids.append(D_["videoId"])
        if 'cam' in D_.keys():
            all_cam_ids.append(D_["cam"])
        else:
            parts_ = file.split("-")
            all_cam_ids.append(int(parts_[3]))
        if ntype == 1:
            rec2vid[int(record)] = D_["videoId"]
        else:
            rec2vid[bfile] = D_["videoId"]
            all_gait_ids.append(D_["gait"])
            if 'cam' in D_.keys():
                rec2cam[bfile] = D_["cam"]
            else:
                parts_ = file.split("-")   
                rec2vid[bfile] = int(parts_[3])

    if ntype == 1:
        datadirs = [datadir, ""]
    elif multimodal:
        datadirs = [datadir, datadir.replace("{}25".format("of"), "{}25".format(modality))]
    else:
        datadirs = [datadir.replace("{}25".format("of"), "{}25".format(modality)), ""]


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

    # Create data generator
    val_generator = DataGeneratorGaitMMUWYH(allRecords_single_tr, datadir=datadirs, batch_size=batchsize,
                                        dim=input_shape, n_classes=nclasses, labmap=labmap, isDebug=False and ntype==1,
                                            ntype=ntype, camera=all_cam_ids, nmods=nmods, gait=all_gait_ids,
                                            expand_level=1, isTest=True, augmentation_x=0, shuffle=False, gaitset=gaitset)

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
    all_cams = []
    #nbatches = len(val_generator)
    nbatches = int(np.ceil(len(val_generator.list_IDs) / np.float(batchsize)))
    if verbose > 0:
        print("Evaluating...")
    for bix in range(nbatches):  # We want the remaining samples as well
        sys.stdout.write("{}/{}\r".format(bix, nbatches))
        sys.stdout.flush()

        #tuples, labels = val_generator.__getitem__(bix)
        tuples, labels, info, cams = val_generator.__getitemwithinfo__(bix, expand=1, withcam=True)

        if multitask:
            labels = labels[0]
        
        if multimodal:
            of0, uof0, gray0, ug0 = tuples
        else:
            of0 = tuples        

        if mirror:
            of1 = np.empty(shape=of0.shape)
            gray1 = np.empty(shape=gray0.shape)
            for tix in range(len(of0)):
                of1[tix] = mj_mirrorsequence(of0[tix], True, True)
                gray1[tix] = mj_mirrorsequence(gray0[tix], False, True)

            of0 = np.vstack((of0, of1))
            uof0 = np.vstack((uof0, uof0))
            gray0 = np.vstack((gray0, gray1))
            ug0 = np.vstack((ug0, ug0))
            labels = np.vstack((labels, labels))
            cams = np.vstack((cams, cams))

        # signatures = model_code.predict([of0, uof0, gray0, ug0])
        if multimodal:
            signatures = model_code.predict([of0, uof0, gray0, ug0])
        else:
            signatures = model_code.predict(of0)
        # if not mj_isDebugging():
        #     estim = model.predict(tuples)
        #     logits = estim[1]
        #     pred_labs = logits.argmax(axis=1)
        # else:
        #     logits = np.array([])
        #     pred_labs = np.array([])


        gt_labs = labels.astype(np.int)
        l_vids = len(gt_labs) * [0]

        # Find corresponding video-id
        for i, rg in enumerate(info):
            if use_mod1:
                rec = rg[0][0]
            else:
                rec = rg[0][1]
            #idx = np.where(Drecs == rec)[0]
            vid = rec2vid[rec] #Dvids[idx[0]]
            l_vids[i] = vid

        all_vids = all_vids + l_vids
        all_cams = all_cams + cams

        if bix == 0:
            all_gt_labs = gt_labs
            #all_pred_labs = pred_labs
            all_codes = signatures
        else:
            all_gt_labs = np.append(all_gt_labs, gt_labs)
            #all_pred_labs = np.append(all_pred_labs, pred_labs)
            all_codes = np.vstack((all_codes, signatures))

        #import pdb; pdb.set_trace()

    return all_codes, all_gt_labs, all_vids, all_cams


def evalUWYHNet(model, datadir_train="tfimdb_casia_b_N050_ft_of25_60x60",
                datadir_test="tfimdb_casia_b_N050_test_nm05-06_072_of25_60x60",
                experfix="demo",
                nclasses=50, tdim=50, modality='gray', use_avg=True,
                ntype=1, use_mod1=1, use_mod2=1, typecode=1,
                knn=7, usemirror=False, modality0="of",
                batchsize=64, experdir=".", verbose=0, gaitset=False):

    if gaitset:
        if modality0 == 'of':
            input_shape0 = (25, 60, 60, 2)
        else:
            input_shape0 = (25, 60, 60, 1)

        if modality == 'of':
            input_shape1 = (25, 60, 60, 2)
        else:
            input_shape1 = (25, 60, 60, 1)
    else:
        if modality0 == 'of':
            input_shape0 = (50, 60, 60)
        else:
            input_shape0 = (25, 60, 60)

        if modality == 'of':
            input_shape1 = (50, 60, 60)
        else:
            input_shape1 = (25, 60, 60)

    if modality0.lower() == 'none':
        input_shape = input_shape1
        multimodal = False
    else:
        input_shape = [input_shape0, input_shape1]
        multimodal = True

    if usemirror:
        print(" - Mirror samples will be added to gallery.")

    outfile_gal = osp.join(experdir, "codes_gallery_{}_M{}{}_t{}_mir{}.h5".format(experfix, int(use_mod1), int(use_mod2),
                                                                               typecode, int(usemirror)))
    # import pdb; pdb.set_trace()

    if osp.exists(outfile_gal):
        exper = dd.io.load(outfile_gal)
        print("Gallery loaded from file: "+outfile_gal)
        all_codes_gallery = exper["codes"]
        all_gt_labs_gallery = exper["gtlabs"]
        all_vids_gallery = exper["vids"]
        all_cams_gallery = exper["cams"]
    else:
        start = time.time()
        all_codes_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery = evalUWYHNet_set(model, datadir_train,
                                                                                   nclasses, tdim, modality,
                                                                                   use_avg,
                                                                                   ntype, use_mod1, use_mod2,
                                                                                   typecode, usemirror,
                                                                                   input_shape, modality0,
                                                                                   batchsize, experdir, 
                                                                                   multimodal, verbose, gaitset=gaitset)
        print("all_gt_labs_gallery", all_gt_labs_gallery)
        print("all_cams_gallery", all_cams_gallery)
        end_time = time.time()
        print("- Elapsed time for GALLERY: {}".format(end_time - start))

    # Save data, just in case
    exper = {}
    exper["codes"] = all_codes_gallery
    exper["gtlabs"] = all_gt_labs_gallery
    exper["vids"] = all_vids_gallery
    exper["cams"] = all_cams_gallery
    dd.io.save(outfile_gal, exper)
    print("Data saved to: "+outfile_gal)

    if verbose > 0:
        print("# Preparing NN classifier.")
    from sklearn.neighbors import KNeighborsClassifier

    if verbose > 0:
        print(len(all_codes_gallery))

    # Find test directories per camera
    if multimodal:
        ltestcases = glob.glob("/localscratch/users/fcastro/data/CASIAB_tf/tfimdb_casia_b_N050_test_*of25*.h5")
    else:
        ltestcases = glob.glob("/localscratch/users/fcastro/data/CASIAB_tf/tfimdb_casia_b_N050_test_*{}25*.h5".format(modality))

    lresults = {}
    laccs = []
    ncases = len(ltestcases)
    for cix in range(ncases):
        datadir_test2 = ltestcases[cix].split(".")[0]
        bdir = os.path.basename(datadir_test2)
        camera = int(bdir.split("_")[6])
        case_f = bdir.split("_")[5][0:2]
        print("+ Target camera: {} [{}]".format(camera, case_f))

        all_codes_test, all_gt_labs_test, all_vids_test, all_cams_test = evalUWYHNet_set(model, datadir_test2,
                                                                          nclasses, tdim, modality, use_avg,
                                                                          ntype, use_mod1, use_mod2, typecode, False,
                                                                          input_shape, modality0,
                                                                          batchsize, experdir, 
                                                                          multimodal, verbose, gaitset=gaitset)

        # Save data, just in case
        if not osp.exists(experdir):
            os.makedirs(experdir)

        outfile = osp.join(experdir, "codes_probe_{}_{}.h5".format(case_f, camera))
        exper = {}
        exper["codes"] = all_codes_test
        exper["gtlabs"] = all_gt_labs_test
        exper["vids"] = all_vids_test
        exper["cams"] = all_cams_test
        dd.io.save(outfile, exper)
        if verbose > 0:
            print("Data saved to: "+outfile)

        print("# Predicting with {}-NN classifier.".format(knn))
        acc = 0.0
        acc_vid = 0.0

        #import pdb; pdb.set_trace()
        #for cameras_ids in np.unique(all_cams_gallery):
            # Remove same camera samples from gallery during test
        #    if cameras_ids == camera:
        #        continue

        cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        avg_acc_subseq = 0.0;
        avg_acc_video = 0.0;
        for cam_gallery in cameras:
            if cam_gallery != camera:
#                print("gallery: ", cam_gallery, ", test: ", camera)
                print(np.array(all_cams_gallery))
                idx_cam = np.where(np.array(all_cams_gallery) == cam_gallery)[0]
                print(idx_cam)
                clf = KNeighborsClassifier(n_neighbors=knn)
                clf.fit(all_codes_gallery[idx_cam], all_gt_labs_gallery[idx_cam])

                start = time.time()
                all_pred_labs = clf.predict(all_codes_test)
                end = time.time()

                # Summarize per video
                uvids = np.unique(all_vids_test)

                # Majority voting per video
                all_gt_labs_per_vid = []
                all_pred_labs_per_vid = []
                for vix in uvids:
                    idx = np.where(all_vids_test == vix)[0]

                    gt_lab_vid = stats.mode(list(np.asarray(all_gt_labs_test)[idx]))[0][0]
                    pred_lab_vid = stats.mode(list(np.asarray(all_pred_labs)[idx]))[0][0]

                    all_gt_labs_per_vid.append(gt_lab_vid)
                    all_pred_labs_per_vid.append(pred_lab_vid)

                all_gt_labs_per_vid = np.asarray(all_gt_labs_per_vid)
                all_pred_labs_per_vid = np.asarray(all_pred_labs_per_vid)

                # At subsequence level
                M = confusion_matrix(all_gt_labs_test, all_pred_labs)
                acc = M.diagonal().sum() / len(all_gt_labs_test)
                print("*** Accuracy [subseq]: {:.2f}".format(acc * 100))

                # At video level
                Mvid = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid)
                acc_vid = Mvid.diagonal().sum() / len(all_gt_labs_per_vid)
                print("*** Accuracy [video]: {:.2f}".format(acc_vid * 100))
                avg_acc_subseq = avg_acc_subseq + (acc * 100)
                avg_acc_video = avg_acc_video + (acc_vid * 100)

        avg_acc_subseq = avg_acc_subseq / (len(cameras) - 1)
        avg_acc_video = avg_acc_video / (len(cameras) - 1)
        print("@@@ ACC SUBSEQ CAMERA ", camera, ": ", avg_acc_subseq)
        print("@@@ ACC VIDEO CAMERA ", camera, ": ", avg_acc_video)

    # Save to disk
    outfile = osp.join(experdir, "all_test_results_{}_M{}{}_K{}_t{}.h5".format(experfix, int(use_mod1), int(use_mod2),
                                                                               knn, typecode))
    dd.io.save(outfile, lresults)
    print("*** All results saved to: " + outfile)

    print(lresults)
    #print(np.array(laccs).mean())

    #return (M, Mvid)
    return lresults


################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

    # /home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N155-n-05_06-of25_60x60
    parser.add_argument('--datadir', type=str, required=False,
                        default='PATH/CASIAB_tf/tfimdb_casia_b_N050_test_nm05-06_054_of25_60x60',
                        help="Full path to data directory")

    parser.add_argument('--datadirtrain', type=str, required=False,
                        default='PATH/CASIAB_tf/tfimdb_casia_b_N050_ft_of25_60x60',
                        help="Full path to data gallery directory")

    parser.add_argument('--model', type=str, required=True,
                        default='PATH/experiments/casiab/model-state-0002.hdf5',
                        help="Full path to model file (DD: .hdf5)")

    parser.add_argument('--prefix', type=str, required=False,
                        default="demo",
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
    parser.add_argument('--gaitset', default=False, action='store_true')

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
    usemirror = args.usemirror > 0
    modality = args.modality
    modality0 = args.modality0
    typecode = args.typecode
    knn = args.knn
    gaitset=args.gaitset

    # Load the model
    #from nets.mj_uwyhNets import UWYHSemiNet
    from nets.mj_uwyhNets_ba import UWYHSemiNet
    model = UWYHSemiNet.loadnet(modelpath)


    if model is not None and verbose > 0:
        model.summary()

    experdir = osp.dirname(modelpath)
    #modelcase = os.path.basename(modelpath).split("-")[2].split(".")[0]
    experfix = "st"#+modelcase

    # Call the evaluator
    lresults = evalUWYHNet(model, datadir_train=datadirtrain, datadir_test=datadir, experfix=experfix,
                     nclasses=nclasses, tdim=tdim, batchsize=batchsize, use_avg=use_avg,
                     modality=modality,
                     use_mod1=usemod1, use_mod2=usemod2, typecode=typecode,
                     knn= knn, usemirror=usemirror, modality0=modality0,
                     experdir=experdir, ntype=nametype, verbose=verbose, gaitset=gaitset)

    # Save CM
    modelcase = os.path.basename(modelpath).split("-")[2].split(".")[0]
    outfile = osp.join(experdir, "evaluation_s{}_M{}{}_K{}_t{}.h5".format(modelcase, int(usemod1), int(usemod2), knn,
                                                                          typecode))
    dd.io.save(outfile, lresults)

    print("Done!")
