"""
Utils for networks

(c) MJMJ/2020
"""
__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'March 2020'

from tensorboard.plugins import projector
import os.path as osp
import numpy as np

def mj_freezeModel(model, onlyconv=False):
    """
    Freeze the weights of the layers of the input model
    :param model: tf.keras model
    :return: updated model
    """
    for layer in model.layers:
        if onlyconv:
            if 'conv' in layer.name:
                layer.trainable = False
        else:
            layer.trainable = False

    return model

def mj_load_model(modelpath : str, custom_objects=None, verbose=1):
    """
    Load a saved model. This function tries to use the net configuration file and weights if direct loading doesn't work
    :param modelpath: full path to file
    :param custom_objects: for example, {'VerifLossLayer': VerifLossLayer}
    :param verbose: integer >= 0
    :return: the model
    """
    from tensorflow.keras.models import load_model
    import deepdish as dd

    if osp.exists(modelpath):
        try:
            model = load_model(modelpath, custom_objects=custom_objects)
            if verbose > 0:
                print("Model loaded.")
        except:
            from nets.mj_verifNets import VerifNet

            bdir = osp.dirname(modelpath)
            fconfig = osp.join(bdir, "model-config.hdf5")
            netconfig = dd.io.load(fconfig)

            filters_size = netconfig["filters_size"]  #[(7, 7), (5, 5), (3, 3), (2, 2)]
            filters_numbers =  netconfig["filters_numbers"] #[64, 128, 512, 512]
            input_shape = netconfig["input_shape"] #[(50, 60, 60), (25, 60, 60)]
            ndense_units = netconfig["ndense_units"] #256
            weight_decay = netconfig["weight_decay"] #0.00005
            dropout = netconfig["dropout"] #0

            model = VerifNet.build(input_shape, len(filters_numbers),
                                   filters_size, filters_numbers, ndense_units, weight_decay, dropout)

            bname = osp.basename(modelpath)
            fparts = osp.splitext(bname)
            filewes = osp.join(bdir, fparts[0]+"_weights.hdf5" )
            model.load_weights(filewes)
        if verbose > 1:
            model.summary()
    else:
        if verbose> 0:
            print("WARNING: model doesn't exist.")
        model = None

    return model


def mj_register_embedding(embedding_tensor_name, meta_data_fname, log_dir, sprite_path=None, image_size=(1,1)):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname

    if sprite_path:
        embedding.sprite.image_path = sprite_path
        embedding.sprite.single_image_dim.extend(image_size)

    projector.visualize_embeddings(log_dir, config)


def mj_save_labels_tsv(labels, filepath, log_dir):
    with open(osp.join(log_dir, filepath), 'w') as f:
        #f.write('Class\n') # Not allowed as we have just one column
        for label in labels:
            f.write('{}\n'.format(label))


def mj_save_sprite(l_images, filepath, log_dir):
    from PIL import Image
    from sklearn.preprocessing import MinMaxScaler
    import math
    import numpy as np

    scaler = MinMaxScaler()

    grid = int(math.sqrt(len(l_images))) + 1
    image_height = int(8192 / grid)  # tensorboard supports sprite images up to 8192 x 8192
    image_width = int(8192 / grid)

    big_image = Image.new(
        mode='RGB',
        size=(image_width * grid, image_height * grid),
        color=(0, 0, 0))  # RGB

    for i in range(len(l_images)):
        row = int(i / grid)
        col = i % grid

        # Prepare to be compatible with PIL
        img = l_images[i]
        scaler.fit(img)
        img = np.uint8(scaler.transform(img) * 255)
        img = Image.fromarray(img)

        img = img.resize((image_height, image_width), Image.ANTIALIAS)
        row_loc = row * image_height
        col_loc = col * image_width
        big_image.paste(img, (col_loc, row_loc))  # NOTE: the order is reverse due to PIL saving
        #print(row_loc, col_loc)

    sprite_name = osp.join(log_dir, filepath)
    big_image.save(sprite_name)

    return sprite_name, (image_height, image_width)


def mj_save_filters(model, logdir: str, step: int, branchname="ofBranch", title="filters-of", Nfi=12):
    """
    Saves conv filters to tensorboard images
    :param model: trained model
    :param logdir: path
    :param step: epoch
    :param branchname: currently "ofBranch", "grayBranch"
    :param title: string to name images
    :param Nfi: number of filters to export
    """
    import tensorflow as tf
    import numpy as np

    if branchname == "ofBranch":
        isOF = True
    else:
        isOF = False

    ofb = model.get_layer(branchname)

    #layer = ofb.get_layer("conv2d")
    for layer in ofb.layers:
        if 'conv' in layer.name:
            break

    w = layer.get_weights()
    filters = w[0]

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    Nf = 16

    if isOF:
        nrows = 2
        mul = 2
    else:
        nrows = 1
        mul = 1

    img = np.zeros((Nfi, nrows * 7, Nf * 7, 1))

    for ix in range(0, Nfi):
        for j in range(0, Nf):
            fx = filters[:, :, mul * j, ix]
            img[ix, 0:7, j * 7:(j + 1) * 7, 0] = fx

            if nrows > 1:
                fy = filters[:, :, 2 * j + 1, ix]
                img[ix, 7:14, j * 7:(j + 1) * 7, 0] = fy

    fw = tf.summary.create_file_writer(logdir)
    with fw.as_default():
        tf.summary.image(title, img, step=step, max_outputs=Nfi)


def mj_save_filters3d(model, logdir: str, step: int, branchname="grayBranch", title="filters-gray-3d", Nfi=12):
    """
    Saves conv filters to tensorboard images
    :param model: trained model
    :param logdir: path
    :param step: epoch
    :param branchname: currently "ofBranch", "grayBranch"
    :param title: string to name images
    :param Nfi: number of filters to export
    """
    import tensorflow as tf
    import numpy as np

    if branchname == "ofBranch":
        isOF = True
    else:
        isOF = False

    ofb = model.get_layer(branchname)

    #layer = ofb.get_layer("conv2d")
    for layer in ofb.layers:
        if 'conv' in layer.name:
            break

    w = layer.get_weights()
    filters = w[0]
    shape = w[0].shape

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    Nf = shape[0]
    nrows = 1
    mul = 1

    img = np.zeros((Nfi, nrows * shape[1], Nf * shape[2], 1))

    for ix in range(0, Nfi):
        for j in range(0, Nf):
            fx = filters[j, :, :, 0, ix]
            img[ix, :, j * shape[2]:(j + 1) * shape[2], 0] = fx

    fw = tf.summary.create_file_writer(logdir)
    with fw.as_default():
        tf.summary.image(title, img, step=step, max_outputs=Nfi)


def mj_softlabel(l_labels, nclasses, epsilon=0.1):
    the_class = 1.0 - epsilon*(nclasses-1) / nclasses
    others = epsilon / nclasses

    ns = len(l_labels)

    output = np.full((ns, nclasses), others, dtype='float32')
    #import pdb; pdb.set_trace()
    output[np.arange(ns), np.array(l_labels, dtype='int')] = the_class

    return output
