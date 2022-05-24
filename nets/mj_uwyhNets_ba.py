__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

# import tensorflow as tf

import os.path as osp

import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Maximum, Average
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
from nets.mj_loss import *
from nets.triplet_loss_all import triplet_loss
from nets.aux_loss import ContrastiveLoss
import numpy as np

# tf.compat.v1.disable_eager_execution()

class MatMul(Layer):
    def __init__(self, bin_num=31, hidden_dim=256, **kwargs):
        super(MatMul, self).__init__(**kwargs)

        self.bin_num = bin_num
        self.hidden_dim = hidden_dim

        # Create a trainable weight variable for this layer.
        w_init = tf.keras.initializers.GlorotUniform()
        self.kernel = tf.Variable(name="MatMul_kernel"+str(np.random.randint(100, size=1)), initial_value=w_init(shape=(bin_num*2, 128, hidden_dim), dtype="float32"),
                                  trainable=True)

    def call(self, x):
        # Implicit broadcasting occurs here.
        # Shape x: (BATCH_SIZE, N, M)
        # Shape kernel: (N, M)
        # Shape output: (BATCH_SIZE, N, M)
        return tf.matmul(x, self.kernel)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bin_num': self.bin_num,
            'hidden_dim': self.hidden_dim,
        })
        return config


def mj_tensor_times_scalar(d):
    tensor = d[0]
    scalar = d[1]
    return tensor * scalar


def fc_loadBranch(init_branch):
    model = tf.keras.models.load_model(init_branch,
                                       custom_objects={"triplet_loss": triplet_loss(), "MatMul": MatMul(), "ContrastiveLoss": ContrastiveLoss()}, compile=False)
    branch = model.get_layer("convBranch")
    return branch


class UWYHNet:

    @staticmethod
    def buildBranch(name, input_shape=(50, 60, 60), number_convolutional_layers=4, filters_size=None,
                    filters_numbers=None, ndense_units=512, weight_decay=1e-4,
                    dropout=0.4, init_branch=None):
        if init_branch != "":
            # Load already trained branch.
            print("Loading branch from ", init_branch, "...")
            ofBranch = fc_loadBranch(init_branch)
            ofBranch._init_set_name(name)
            print("Done!")
        else:
            if filters_numbers is None:
                filters_numbers = [64, 128, 512, 512]
            L2_norm = regularizers.l2(weight_decay)
            ofBranch = Sequential(name=name)

            ofBranch.add(Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), kernel_regularizer=L2_norm,
                                activation='relu', input_shape=input_shape, data_format='channels_first'))

            ofBranch.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

            for i in range(1, number_convolutional_layers):
                ofBranch.add(Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), kernel_regularizer=L2_norm,
                                    activation='relu', data_format='channels_first'))

                if i != number_convolutional_layers - 1:
                    ofBranch.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

            ofBranch.add(Flatten(name="ofFlat"))

            # Insert a dense+dropout layers here
            if dropout > 0.001:
                ofBranch.add(Dense(ndense_units * 2, name="dense"))
                ofBranch.add(Dropout(dropout, name="drop"))
            else:
                ofBranch.add(Dense(ndense_units * 2, name="dense"))

            # Dense without activation function
            ofBranch.add(Dense(ndense_units, activation=None, kernel_regularizer=regularizers.l2(1e-3),
                               kernel_initializer='he_uniform', name="ofCode"))

        return ofBranch

    @staticmethod
    def buildBranchLReLU(name, input_shape=(50, 60, 60), number_convolutional_layers=4, filters_size=None,
                         filters_numbers=None, ndense_units=512, weight_decay=1e-4,
                         dropout=0.4, init_branch=None, alpha=0.3):
        if init_branch != "":
            # Load already trained branch.
            print("Loading branch from ", init_branch, "...")
            ofBranch = fc_loadBranch(init_branch)
            ofBranch._init_set_name(name)
            print("Done!")
        else:
            if filters_numbers is None:
                filters_numbers = [64, 128, 512, 512]
            L2_norm = regularizers.l2(weight_decay)
            ofBranch = Sequential(name=name)

            ofBranch.add(Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), kernel_regularizer=L2_norm,
                                activation=None, input_shape=input_shape, data_format='channels_first'))
            ofBranch.add(LeakyReLU(alpha=alpha))

            ofBranch.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

            for i in range(1, number_convolutional_layers):
                ofBranch.add(Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), kernel_regularizer=L2_norm,
                                    activation=None, data_format='channels_first'))
                ofBranch.add(LeakyReLU(alpha=alpha))

                if i != number_convolutional_layers - 1:
                    ofBranch.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

            ofBranch.add(Flatten(name="ofFlat"))

            # Insert a dense+dropout layers here
            if dropout > 0.001:
                ofBranch.add(Dense(ndense_units * 2, name="dense"))
                ofBranch.add(Dropout(dropout, name="drop"))
            else:
                ofBranch.add(Dense(ndense_units * 2, name="dense"))

            # Dense without activation function
            ofBranch.add(Dense(ndense_units, activation=None, kernel_regularizer=regularizers.l2(1e-3),
                               kernel_initializer='he_uniform', name="ofCode"))

        return ofBranch

    @staticmethod
    def build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
              ndense_units=512, weight_decay=1e-4, dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2,
              init_branches=None, freeze_branches=False, activation_fn='relu', alpha=0.3):
        """

		:param input_shapes: tuple ((50,60,60), (25,60,60))
		:param number_convolutional_layers:
		:param filters_size:
		:param filters_numbers:
		:param ndense_units:
		:param weight_decay:
		:param dropout:
		:param optimizer:
		:param margin:
		:return:
		"""

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        if init_branches is None:
            init_branches = {'of': '', 'gray': '', 'depth': ''}

        if activation_fn == 'relu':
            ofBranch = UWYHNet.buildBranch("ofBranch", (50, 60, 60), number_convolutional_layers, filters_size,
                                           filters_numbers, ndense_units, weight_decay, dropout, init_branch=init_branches['of'])

            grayBranch = UWYHNet.buildBranch("grayBranch", (25, 60, 60), number_convolutional_layers, filters_size,
                                             filters_numbers, ndense_units, weight_decay, dropout, init_branch=init_branches['gray'])
        else:
            ofBranch = UWYHNet.buildBranchLReLU("ofBranch", (50, 60, 60), number_convolutional_layers, filters_size,
                                                filters_numbers, ndense_units, weight_decay, dropout,
                                                init_branch=init_branches['of'], alpha=alpha)

            grayBranch = UWYHNet.buildBranchLReLU("grayBranch", (25, 60, 60), number_convolutional_layers, filters_size,
                                                  filters_numbers, ndense_units, weight_decay, dropout,
                                                  init_branch=init_branches['gray'], alpha=alpha)

        if freeze_branches:
            for layer in ofBranch.layers:
                layer.trainable = False

            for layer in grayBranch.layers:
                layer.trainable = False

        # Process data: branch 1
        # ----------------------
        ofinput1 = Input(shape=input_shapes[0], name='ofinput1')
        of_use1 = Input(shape=(1), name='ofuse1')
        grayinput1 = Input(shape=input_shapes[1], name='grayinput1')
        gray_use1 = Input(shape=(1), name='grayuse1')

        ofout1 = ofBranch(ofinput1)
        ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofout1, of_use1])

        grayout1 = grayBranch(grayinput1)
        grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayout1, gray_use1])

        # Combine layers
        agg1 = Maximum(name="agg1")([ofout1, grayout1])
        # L2 normalize embeddings
        agg1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedL2_1")(agg1)

        # Process data: branch 2
        # ----------------------
        ofinput2 = Input(shape=(50, 60, 60), name='ofinput2')
        of_use2 = Input(shape=(1), name='ofuse2')
        grayinput2 = Input(shape=(25, 60, 60), name='grayinput2')
        gray_use2 = Input(shape=(1), name='grayuse2')

        ofout2 = ofBranch(ofinput2)
        ofout2 = layers.Lambda(mj_tensor_times_scalar, name="gate_of2")([ofout2, of_use2])

        grayout2 = grayBranch(grayinput2)
        grayout2 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray2")([grayout2, gray_use2])

        # Combine layers
        agg2 = Maximum(name="agg2")([ofout2, grayout2])
        # L2 normalize embeddings
        agg2 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedL2_2")(agg2)

        label = Input(shape=(1), name='label')
        loss_layer = VerifLossLayer(margin, name="loss_pair")([agg1, agg2, label])

        # Build composed model
        the_inputs = [ofinput1, of_use1, grayinput1, gray_use1, ofinput2, of_use2, grayinput2, gray_use2, label]

        model = Model(inputs=the_inputs, outputs=loss_layer)
        model.compile(optimizer=optimizer)

        return model

    @staticmethod
    def fit_generator(model, epochs, callbacks, training_generator, validation_generator, current_step, steps_per_epoch,
                      validation_steps, nworkers=0, new_lr=None):
        """
        Trains a model by using data generators
        :param epochs:
        :param callbacks: list of Keras callbacks
        :param training_generator:
        :param validation_generator:
        :param current_step:
        :param steps_per_epoch:
        :param validation_steps:
        :param nworkers: for parallel processing
        :param new_lr: force this learning rate
        :return: model, history
        """

        if new_lr is not None:
            import tensorflow.keras.backend as K
            K.set_value(model.optimizer.lr, new_lr)
            print("INFO: learning rate has been changed to {}".format(new_lr))
        hist = model.fit(training_generator, validation_data=validation_generator,
                         epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                         validation_steps=validation_steps, initial_epoch=current_step, verbose=2) #, use_multiprocessing=False,
                         #workers=nworkers)
        return model, hist

    @staticmethod
    def encode(model, batch_data, use_data):
        seq_of = model.get_layer('ofBranch')  # model.layers[4]
        seq_gray = model.get_layer('grayBranch')  # model.layers[5]

        of_codes = seq_of(batch_data[0])
        of_codes = layers.Lambda(mj_tensor_times_scalar, name="gate_of")([of_codes, use_data[0]])

        gray_codes = seq_gray(batch_data[1])
        gray_codes = layers.Lambda(mj_tensor_times_scalar, name="gate_gray")([gray_codes, use_data[1]])

        import tensorflow.keras.layers as KL
        max_codes = KL.Maximum()([of_codes, gray_codes])

        # L2 normalize embeddings
        codes_norm_tf = KL.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(max_codes)

        # Get the numpy matrix
        codes_norm = codes_norm_tf.numpy()

        return codes_norm


# ----------------------------------- function -----------------

def mj_buildnet_by_config(netconfig, buildfun):
    filters_size = netconfig["filters_size"]
    filters_numbers = netconfig["filters_numbers"]
    input_shape = netconfig["input_shape"]
    ndense_units = netconfig["ndense_units"]
    weight_decay = netconfig["weight_decay"]
    dropout = netconfig["dropout"]
    if "nclasses" in netconfig.keys():
        nclasses = netconfig["nclasses"]
    else:
        nclasses = 150
    optimizer = netconfig["optimizer"]
    margin = netconfig["margin"]
    # import numpy as np
    # margin = np.float32(margin) # TODO
    if "loss_weights" in netconfig.keys():
        loss_weights = netconfig["loss_weights"]
    else:
        loss_weights = [1.0, 0.1]

    if "use3D" in netconfig.keys():
        use3D = netconfig["use3D"]
    else:
        use3D = False

    model = buildfun(input_shape, len(filters_numbers),
                     filters_size, filters_numbers, ndense_units, weight_decay, dropout,
                     nclasses=nclasses, loss_weights=loss_weights,
                     optimizer=optimizer, margin=margin, use3D=use3D)

    return model


# --------------- With semi-hard triplet loss ---------------
class UWYHSemiNet:

    @staticmethod
    def build_3Dbranch(name, input_shape=(25, 60, 60, 1), ndense_units=512, init_branch=""):

        if init_branch != "":
            # Load already trained branch.
            print("Loading branch from ", init_branch, "...")
            model = fc_loadBranch(init_branch)
            model._init_set_name(name)
            print("Done!")
        else:
            model = Sequential(name=name)

            model.add(layers.Conv3D(64, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            model.add(layers.Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            model.add(layers.Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            model.add(layers.Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            model.add(layers.Conv3D(512, (3, 2, 2), strides=(1, 1, 1), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            model.add(layers.Conv3D(512, (2, 1, 1), strides=(1, 1, 1), padding='valid', activation='relu',
                                    input_shape=input_shape, data_format='channels_last'))

            # Dense without activation function
            model.add(layers.Conv3D(ndense_units, (1, 1, 1), strides=(1, 1, 1), activation=None,
                                    kernel_regularizer=regularizers.l2(1e-3),
                                    kernel_initializer='he_uniform', name="grayCode"))

            model.add(layers.Flatten())

        return model

    @staticmethod
    def build_3DbranchLReLU(name, input_shape=(25, 60, 60, 1), ndense_units=512, init_branch="", alpha=0.3):

        if init_branch != "":
            # Load already trained branch.
            print("Loading branch from ", init_branch, "...")
            model = fc_loadBranch(init_branch)
            model._init_set_name(name)
            print("Done!")
        else:
            model = Sequential(name=name)

            model.add(layers.Conv3D(64, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            model.add(layers.Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            model.add(layers.Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            model.add(layers.Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            model.add(layers.Conv3D(512, (3, 2, 2), strides=(1, 1, 1), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            model.add(layers.Conv3D(512, (2, 1, 1), strides=(1, 1, 1), padding='valid', activation=None,
                                    input_shape=input_shape, data_format='channels_last'))
            model.add(LeakyReLU(alpha=alpha))

            # Dense without activation function
            model.add(layers.Conv3D(ndense_units, (1, 1, 1), strides=(1, 1, 1), activation=None,
                                    kernel_regularizer=regularizers.l2(1e-3),
                                    kernel_initializer='he_uniform', name="grayCode"))

            model.add(layers.Flatten())

        return model

    @staticmethod
    def build_gaitset_branch(name, input_layer, input_shape=(25, 60, 60, 1), ndense_units=512, init_branch="", norm=True):
        if init_branch != "":
            # Load already trained branch.
            print("Loading branch from ", init_branch, "...")
            model = fc_loadBranch(init_branch)
            model._init_set_name(name)
            print("Done!")
        else:
            branch_a = layers.TimeDistributed(layers.ZeroPadding2D(padding=2))(input_layer)
            branch_a = layers.TimeDistributed(layers.Conv2D(32, kernel_size=5, activation=None, padding='same', use_bias=False, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)
            branch_a = layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, activation=None, padding='same', use_bias=False, input_shape=input_shape, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)
            branch_a = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))(branch_a)

            branch_b = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_a)
            branch_b = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False, data_format='channels_last')(branch_b)
            branch_b = layers.LeakyReLU()(branch_b)

            branch_b = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False, data_format='channels_last')(branch_b)
            branch_b = layers.LeakyReLU()(branch_b)
            branch_b = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(branch_b)

            branch_a = layers.TimeDistributed(layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False, input_shape=input_shape, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)

            branch_a = layers.TimeDistributed(layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False,
														   input_shape=input_shape, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)
            branch_a = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))(branch_a)

            branch_b_ = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_a)
            branch_b = layers.Add()([branch_b, branch_b_])
            branch_b = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False, data_format='channels_last')(branch_b)
            branch_b = layers.LeakyReLU()(branch_b)
            branch_b = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False, data_format='channels_last')(branch_b)
            branch_b = layers.LeakyReLU()(branch_b)

            branch_a = layers.TimeDistributed(layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False, input_shape=input_shape, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)

            branch_a = layers.TimeDistributed(layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False, input_shape=input_shape, data_format='channels_last'))(branch_a)
            branch_a = layers.TimeDistributed(layers.LeakyReLU())(branch_a)
            branch_a = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_a)

            branch_b = layers.Add()([branch_b, branch_a])

            # HPP
            feature = list()
            bin_num = [1, 2, 4, 8, 16]
            n, h, w, c = branch_b.shape
            print(branch_b.shape)
            for num_bin in bin_num:
                branch_a_ = layers.Reshape((num_bin, -1, c))(branch_a)
                branch_a_ = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2) + tf.math.reduce_max(x, axis=2))(branch_a_)
                feature.append(branch_a_)
                branch_b_ = layers.Reshape((num_bin, -1, c))(branch_b)
                branch_b_ = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2) + tf.math.reduce_max(x, axis=2))(branch_b_)
                feature.append(branch_b_)

            model = layers.Concatenate(axis=1)(feature)
            model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
            model = MatMul()(model)

        return model

    @staticmethod
    def build_by_config(netconfig):

        filters_size = netconfig["filters_size"]
        filters_numbers = netconfig["filters_numbers"]
        input_shape = netconfig["input_shape"]
        ndense_units = netconfig["ndense_units"]
        weight_decay = netconfig["weight_decay"]
        dropout = netconfig["dropout"]
        if "nclasses" in netconfig.keys():
            nclasses = netconfig["nclasses"]
        else:
            nclasses = 150
        optimizer = netconfig["optimizer"]
        margin = netconfig["margin"]
        # import numpy as np
        # margin = np.float32(margin) # TODO
        if "loss_weights" in netconfig.keys():
            loss_weights = netconfig["loss_weights"]
        else:
            loss_weights = [1.0, 0.1]

        if "use3D" in netconfig.keys():
            use3D = netconfig["use3D"]
        else:
            use3D = False

        if "postriplet" in netconfig.keys():
            postriplet = netconfig["postriplet"]
        else:
            postriplet = 1

        if "fMerge" in netconfig.keys():
            fMerge = netconfig['fMerge']
        else:
            fMerge = Maximum

        if "fActivation" in netconfig.keys():
            fActivation = netconfig['fActivation']
        else:
            fActivation = 'relu'


        model = UWYHSemiNet.build(input_shape, len(filters_numbers),
                                  filters_size, filters_numbers, ndense_units, weight_decay, dropout,
                                  nclasses=nclasses, loss_weights=loss_weights, optimizer=optimizer, margin=margin,
                                  use3D=use3D, postriplet=postriplet, fMerge=fMerge, fActivation=fActivation)

        return model

    @staticmethod
    def get_weights_filename(modelpath):

        bdir = osp.dirname(modelpath)
        bname = osp.basename(modelpath)
        fparts = osp.splitext(bname)
        filewes = osp.join(bdir, fparts[0] + "_weights.hdf5")

        return filewes

    @staticmethod
    def get_netconfig_filename(modelpath):
        bdir = osp.dirname(modelpath)
        fconfig = osp.join(bdir, "model-config.hdf5")

        return fconfig

    @staticmethod
    def loadnet(netpath: str):

        model = None

        from tensorflow.keras.models import load_model
        import deepdish as dd

#        try:
        print(netpath)
        model = load_model(netpath, custom_objects={"triplet_loss": triplet_loss(), "MatMul": MatMul(), "ContrastiveLoss": ContrastiveLoss()}, compile=False)
        # print("++++++++++++++++++++++++++++++++++++++++++++")
        # print(netpath)
        # model = load_model(netpath)
#        except:
#            print("ಠ_ಠ")
#            bdir = osp.dirname(netpath)
#            fconfig = osp.join(bdir, "model-config.hdf5")
#            netconfig = dd.io.load(fconfig)
#
#            model = UWYHSemiNet.build_by_config(netconfig)
#
#            filewes = UWYHSemiNet.get_weights_filename(netpath)
#
#            model.load_weights(filewes, by_name=True)

        return model

    @staticmethod
    def build_or_load(input_shapes, number_convolutional_layers, filters_size, filters_numbers, ndense_units=512,
                      weight_decay=1e-4, dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2, nclasses=0,
                      loss_weights=[1.0, 1.0], initnet="", freeze_convs=False, use3D=False, smoothlabels=0,
                      freeze_all=False, postriplet=1, init_branches=None, freeze_branches=False, aux_losses=False,
                      fMerge=Maximum, fActivation='relu', gaitset=False):

        if gaitset:
            fActivation = 'leaky'

        if initnet == "":
            model = UWYHSemiNet.build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
                                      ndense_units, weight_decay, dropout, optimizer, margin, nclasses, loss_weights,
                                      use3D=use3D, smoothlabels=smoothlabels, postriplet=postriplet,
                                      init_branches=init_branches, freeze_branches=freeze_branches,
                                      aux_losses=aux_losses, fMerge=fMerge, fActivation=fActivation, gaitset=gaitset)
        else:
            model = None
            model_base = UWYHSemiNet.loadnet(initnet)
            # Check top layer and change if needed
            # try:
            if True:
                try:
                    l = model_base.get_layer("classprob")
                    withclassification = True
                except:
                    print("This model doesn't contain classification layer")
                    withclassification = False

                if withclassification and l.units != nclasses:
                    print("Surgery needed: {} vs {}".format(l.units, nclasses))
                    import deepdish as dd

                    fconfig = UWYHSemiNet.get_netconfig_filename(initnet)
                    netconfig = dd.io.load(fconfig)
                    netconfig["nclasses"] = nclasses
                    netconfig["loss_weights"] = loss_weights
                    netconfig["dropout"] = dropout
                    netconfig["margin"] = margin
                    netconfig["optimizer"] = optimizer
                    netconfig["use3D"] = use3D
                    netconfig["postriplet"] = postriplet
                    netconfig["fMerge"] = fMerge
                    netconfig["fActivation"] = fActivation

                    model = UWYHSemiNet.build_by_config(netconfig)

                    filewes = UWYHSemiNet.get_weights_filename(initnet)

                    model.load_weights(filewes, by_name=True, skip_mismatch=True)  # Load compatible weights
                else:
                    model = model_base

            # Check if freeze some weights
            if freeze_convs or freeze_all:
                if gaitset:
                    model = UWYHSemiNet.build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
                                      ndense_units, weight_decay, dropout, optimizer, margin, nclasses, loss_weights,
                                      use3D=use3D, smoothlabels=smoothlabels, postriplet=postriplet,
                                      init_branches=init_branches, freeze_branches=freeze_branches,
                                      aux_losses=aux_losses, fMerge=fMerge, fActivation=fActivation, gaitset=gaitset)
                    filewes = UWYHSemiNet.get_weights_filename(initnet)

                    model.load_weights(filewes, by_name=True, skip_mismatch=True)  # Load compatible weights
                    for layer_ix in range(len(model.layers)-1):
                        if freeze_all:
                            if hasattr(model.layers[layer_ix], 'layer'):
                                model.layers[layer_ix].layer.trainable = False
                            model.layers[layer_ix].trainable = False
                else:
                    seq1 = model.get_layer("ofBranch")
                    seq2 = model.get_layer("grayBranch")

                    for layer in seq1.layers:
                        if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                            layer.trainable = False

                    for layer in seq2.layers:
                        if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                            layer.trainable = False

        # except:
        # This is not a classification network, just verification
        print("Alright")

        return model

    @staticmethod
    def build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
              ndense_units=512, weight_decay=1e-4, dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2,
              nclasses=0, loss_weights=[1.0, 1.0], use3D=False, smoothlabels=0, postriplet=1, init_branches=None,
              freeze_branches=False, aux_losses=False, fMerge=Maximum, fActivation='relu', alpha=0.3, gaitset=False):
        """

        :param input_shapes: tuple ((50,60,60), (25,60,60))
        :param number_convolutional_layers:
        :param filters_size:
        :param filters_numbers:
        :param ndense_units:
        :param weight_decay:
        :param dropout:
        :param optimizer:
        :param margin:
        :return:
        """

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        if type(input_shapes) is list:
            multimodal = True
        else:
            multimodal = False

        if type(dropout) is list:
            dropout2 = dropout[1]
            dropout = dropout[0]
        else:
            dropout2 = dropout

        if type(ndense_units) is list:
            ndense_units0 = ndense_units[0]
            add_extra_dense = len(ndense_units) > 1
        else:
            ndense_units0 = ndense_units
            add_extra_dense = False

        if init_branches == None:
            init_branches = {'of': "", 'gray': ""}

        if multimodal:
            ofinput1 = Input(shape=input_shapes[0], name='ofinput1')
            of_use1 = Input(shape=(1), name='ofuse1')
            grayinput1 = Input(shape=input_shapes[1], name='grayinput1')
            gray_use1 = Input(shape=(1), name='grayuse1')
        else:
            ofinput1 = Input(shape=input_shapes, name='ofinput1')

        if fActivation == 'relu':
            if multimodal:
                if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                    ofBranch = UWYHNet.buildBranch("ofBranch", input_shapes[0], number_convolutional_layers, filters_size,
                                                   filters_numbers,
                                                   ndense_units0, weight_decay, dropout, init_branch=init_branches['of'])
                else:
                    ofBranch = UWYHSemiNet.build_3Dbranch("ofBranch", ndense_units=ndense_units0,
                                                          init_branch=init_branches['of'])

                if not use3D:
                    grayBranch = UWYHNet.buildBranch("grayBranch", input_shapes[1], number_convolutional_layers,
                                                     filters_size, filters_numbers,
                                                     ndense_units0, weight_decay, dropout,
                                                     init_branch=init_branches['gray'])
                else:
                    grayBranch = UWYHSemiNet.build_3Dbranch("grayBranch", ndense_units=ndense_units0,
                                                            init_branch=init_branches['gray'])
            else:
                if input_shapes[0] == 50 or not use3D:  # This should be OF
                    ofBranch = UWYHNet.buildBranch("ofBranch", input_shapes, number_convolutional_layers, filters_size,
                                                   filters_numbers, ndense_units0, weight_decay, dropout,
                                                   init_branch=init_branches['of'])
                else:
                    ofBranch = UWYHSemiNet.build_3Dbranch("ofBranch", ndense_units=ndense_units0,
                                                          init_branch=init_branches['of'])
        else:
            if multimodal:
                if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                    if gaitset:
                        ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1, input_shape=input_shapes[0], ndense_units=ndense_units0, init_branch=init_branches['of'])
                    else:
                        ofBranch = UWYHNet.buildBranchLReLU("ofBranch", input_shapes[0], number_convolutional_layers,
                                                   filters_size,
                                                   filters_numbers,
                                                   ndense_units0, weight_decay, dropout,
                                                   init_branch=init_branches['of'], alpha=alpha)
                else:
                    if gaitset:
                        ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1, input_shape=input_shapes[0], ndense_units=ndense_units0, init_branch=init_branches['of'])
                    else:
                        ofBranch = UWYHSemiNet.build_3DbranchLReLU("ofBranch", ndense_units=ndense_units0,
                                                          init_branch=init_branches['of'], alpha=alpha)

                if not use3D:
                    if gaitset:
                        grayBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=grayinput1, input_shape=input_shapes[1], ndense_units=ndense_units0, init_branch=init_branches['of'])
                    else:
                        grayBranch = UWYHNet.buildBranchLReLU("grayBranch", input_shapes[1], number_convolutional_layers,
                                                     filters_size, filters_numbers,
                                                     ndense_units0, weight_decay, dropout,
                                                     init_branch=init_branches['gray'], alpha=alpha)
                else:
                    grayBranch = UWYHSemiNet.build_3DbranchLReLU("grayBranch", ndense_units=ndense_units0,
                                                            init_branch=init_branches['gray'], alpha=alpha)
            else:
                if input_shapes[0] == 50 or not use3D:  # This should be OF
                    if gaitset:
                        ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1, input_shape=input_shapes, ndense_units=ndense_units0, init_branch=init_branches['of'])
                    else:
                        ofBranch = UWYHNet.buildBranchLReLU("ofBranch", input_shapes, number_convolutional_layers, filters_size,
                                                   filters_numbers, ndense_units0, weight_decay, dropout,
                                                   init_branch=init_branches['of'], alpha=alpha)
                else:
                    if gaitset:
                        ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1, input_shape=input_shapes, ndense_units=ndense_units0, init_branch=init_branches['of'])
                    else:
                        ofBranch = UWYHSemiNet.build_3DbranchLReLU("ofBranch", ndense_units=ndense_units0,
                                                          init_branch=init_branches['of'])

        if freeze_branches:
            for layer in ofBranch.layers:
                layer.trainable = False

            if multimodal:
                for layer in grayBranch.layers:
                    layer.trainable = False

        if multimodal:
            # Process data: branch 1
            # ----------------------
            if gaitset:
                ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofBranch, of_use1])
            else:
                ofout1 = ofBranch(ofinput1)
                ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofout1, of_use1])

            if gaitset:
                grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayBranch, gray_use1])
            else:
                grayout1 = grayBranch(grayinput1)
                grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayout1, gray_use1])

            # Combine layers
            #agg1 = Maximum(name="fusion")([ofout1, grayout1])
            agg1 = fMerge(name="fusion")([ofout1, grayout1])  # Custom merge can be used
            if postriplet == 1 or not add_extra_dense:
                # L2 normalize embeddings
                agg1 = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(agg1)
                outsignature = agg1

            if add_extra_dense:
                if postriplet == 2:
                    if fActivation == 'relu':
                        x = layers.Dense(ndense_units[1], activation='relu', activity_regularizer=regularizers.l2(1e-3),
                                     name="signature")(agg1)
                    else:
                        x = layers.Dense(ndense_units[1], activation=None, activity_regularizer=regularizers.l2(1e-3),
                                         name="signature")(agg1)
                        x = LeakyReLU(alpha=alpha)(x)

                    x = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="code")(x)
                    outsignature = x
                    agg1 = layers.Dropout(dropout2, name="dropcode")(x)
                else:
                    if fActivation == 'relu':
                        x = layers.Dense(ndense_units[1], activation='relu', activity_regularizer=regularizers.l2(1e-3),
                                     name="code")(agg1)
                    else:
                        x = layers.Dense(ndense_units[1], activation=None, activity_regularizer=regularizers.l2(1e-3),
                                         name="code")(agg1)
                        x = LeakyReLU(alpha=alpha)(x)

                    agg1 = layers.Dropout(dropout2, name="dropcode")(x)

            # Build composed model
            the_inputs = [ofinput1, of_use1, grayinput1, gray_use1]

            if nclasses > 0:
                agg1 = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(agg1)
                agg1 = layers.Flatten()(agg1)
                classprob = layers.Dense(nclasses, activation='softmax', name="classprob")(agg1)
                outputs = [outsignature, classprob]

                # Add auxiliar classifiers.
                if aux_losses:
                    classprob_of = layers.Dense(nclasses, activation='softmax', name="classprob_of")(ofout1)
                    classprob_gray = layers.Dense(nclasses, activation='softmax', name="classprob_gray")(grayout1)

                    outputs.append(classprob_of)
                    outputs.append(classprob_gray)

                loss_weights = loss_weights
                metrics = [[], 'acc']

                if smoothlabels == 0:
                    losses = [triplet_loss(margin=margin), 'categorical_crossentropy']
                    # Add auxiliar losses.
                    if aux_losses:
                        losses.append('categorical_crossentropy')
                        losses.append('categorical_crossentropy')
                        metrics.append('acc')
                        metrics.append('acc')
                else:
                    losses = [triplet_loss(margin=margin),
                              tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels)]
                    # Add auxiliar losses.
                    if aux_losses:
                        losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))
                        losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))

                if len(loss_weights) < len(losses):
                    import numpy as np
                    loss_weights_ = np.zeros(len(losses)) + loss_weights[-1]
                    loss_weights_[0:len(loss_weights)] = np.asarray(loss_weights)
                    loss_weights = list(loss_weights_)
            else:
                outputs = outsignature
                losses = triplet_loss(margin=margin)
                loss_weights = 1.0
                metrics = []
        else:
            the_inputs = ofinput1
            if gaitset:
                ofout1 = ofBranch
            else:
                ofout1 = ofBranch(ofinput1)
            # L2 normalize embeddings
            outsignature = ofout1
            if nclasses > 0:
                agg1 = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(ofout1)
                agg1 = layers.Flatten()(agg1)
                classprob = layers.Dense(nclasses, activation='softmax', name="classprob")(agg1)
                losses = [triplet_loss(margin=margin), 'categorical_crossentropy']
                outputs = [outsignature, classprob]
                metrics = [[], 'acc']
            else:
                outputs = outsignature
                losses = triplet_loss(margin=margin)
                metrics = []

            if False and add_extra_dense:  # TODO see if we want this
                x = layers.Dense(ndense_units[1], activation='relu', activity_regularizer=regularizers.l2(1e-3),
                                 name="code")(agg1)
                agg1 = layers.Dropout(dropout2, name="dropcode")(x)


        model = Model(inputs=the_inputs, outputs=outputs)

        # Check compatible types
        model_type = model.dtype
        same_type = model_type in str(type(margin))
        #if not same_type:
        #    import numpy as np
        #    if model_type == "float32":
        #        margin = np.float32(margin)
        #    else:
        #        margin = np.float64(margin)

        #    if nclasses > 0:
        #        losses[0].margin = margin
        #    else:
        #        losses.margin = margin

        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

        return model

    @staticmethod
    def fit_generator(model, epochs, callbacks, training_generator, validation_generator, current_step, steps_per_epoch,
                      validation_steps, nworkers=0, new_lr=None):
        """
        Trains a model by using data generators
        :param epochs:
        :param callbacks: list of Keras callbacks
        :param training_generator:
        :param validation_generator:
        :param current_step:
        :param steps_per_epoch:
        :param validation_steps:
        :param nworkers: for parallel processing
        :param new_lr: force this learning rate
        :return: model, history
        """

        if new_lr is not None:
            import tensorflow.keras.backend as K
            K.set_value(model.optimizer.lr, new_lr)
            print("INFO: learning rate has been changed to {}".format(new_lr))

        if nworkers > 0:
            mp = True
        else:
            mp = False
        hist = model.fit(training_generator, validation_data=validation_generator,
                         epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                         validation_steps=validation_steps, initial_epoch=current_step, verbose=2) #, use_multiprocessing=mp,
                         #workers=nworkers)

        return model, hist

    @staticmethod
    def encode(model, batch_data, use_data, gaitset=False):
        if gaitset:
            seq_of = model.get_layer('flatten')
            seq_gray = model.get_layer('flatten_1')
            batch_data.insert(1, 1)
            batch_data.insert(3, 1)
            mod = tf.keras.Model(inputs=model.input, outputs=[seq_of.output, seq_gray.output])
            of_codes, gray_codes = mod(batch_data)
        else:
            seq_of = model.get_layer('ofBranch')  # model.layers[4]
            seq_gray = model.get_layer('grayBranch')  # model.layers[5]

            of_codes = seq_of(batch_data[0])
            gray_codes = seq_gray(batch_data[1])


        of_codes = layers.Lambda(mj_tensor_times_scalar, name="gate_of")([of_codes, use_data[0]])
        gray_codes = layers.Lambda(mj_tensor_times_scalar, name="gate_gray")([gray_codes, use_data[1]])

        import tensorflow.keras.layers as KL
        max_codes = KL.Maximum()([of_codes, gray_codes])

        # L2 normalize embeddings
        codes_norm_tf = KL.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(max_codes)

        # Get the numpy matrix
        codes_norm = codes_norm_tf.numpy()

        return codes_norm


# -----------------------------------------------------------------------------
class UWYHSemiNet3Mods(UWYHSemiNet):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loadnet(netpath: str):

        model = None

        from tensorflow.keras.models import load_model
        import deepdish as dd

        try:
            model = load_model(netpath, custom_objects={"triplet_loss": triplet_loss(), "MatMul": MatMul()}, compile=False)
        except:
            print("థ_థ")
            bdir = osp.dirname(netpath)
            fconfig = osp.join(bdir, "model-config.hdf5")
            netconfig = dd.io.load(fconfig)

            model = mj_buildnet_by_config(netconfig, UWYHSemiNet3Mods.build)

            filewes = UWYHSemiNet.get_weights_filename(netpath)

            model.load_weights(filewes, by_name=True)

        return model

    @staticmethod
    def build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
              ndense_units=512, weight_decay=1e-4,
              dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2,
              nclasses=0, loss_weights=[1.0, 1.0], use3D=False, smoothlabels=0,
              postriplet=1, init_branches=None, freeze_branches=False, aux_losses=False, fMerge=Maximum,
              normbfmerge=False, fActivation='relu', alpha=0.3, gaitset=False):
        """

        :param input_shapes: tuple ((50,60,60), (25,60,60))
        :param number_convolutional_layers:
        :param filters_size:
        :param filters_numbers:
        :param ndense_units:
        :param weight_decay:
        :param dropout:
        :param optimizer:
        :param margin:
        :param nclasses: if 0, cross-entropy is not used
        :param loss_weights: if any position is 0, that one is not used
        :return: the model
        """

        # TODO implement use of 'postriplet'

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        if type(ndense_units) is list:
            ndense_units0 = ndense_units[0]
            add_extra_dense = len(ndense_units) > 1
        else:
            ndense_units0 = ndense_units
            add_extra_dense = False

        if init_branches is None:
            init_branches = {'of': '', 'gray': '', 'depth': ''}

        ofinput1 = Input(shape=input_shapes[0], name='ofinput1')
        of_use1 = Input(shape=(1), name='ofuse1')
        grayinput1 = Input(shape=input_shapes[1], name='grayinput1')
        gray_use1 = Input(shape=(1), name='grayuse1')
        depthinput1 = Input(shape=input_shapes[2], name='depthinput1')
        depth_use1 = Input(shape=(1), name='depthuse1')

        if fActivation == 'relu':
            if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                ofBranch = UWYHNet.buildBranch("ofBranch", input_shapes[0], number_convolutional_layers, filters_size,
                                               filters_numbers,
                                               ndense_units0, weight_decay, dropout, init_branch=init_branches['of'])
            else:
                ofBranch = UWYHSemiNet.build_3Dbranch("ofBranch", ndense_units=ndense_units0,
                                                      init_branch=init_branches['of'])

            if not use3D:
                grayBranch = UWYHNet.buildBranch("grayBranch", input_shapes[1], number_convolutional_layers, filters_size,
                                                 filters_numbers,
                                                 ndense_units0, weight_decay, dropout, init_branch=init_branches['gray'])
            else:
                grayBranch = UWYHSemiNet.build_3Dbranch("grayBranch", ndense_units=ndense_units0,
                                                        init_branch=init_branches['gray'])

            if not use3D:
                depthBranch = UWYHNet.buildBranch("depthBranch", input_shapes[2], number_convolutional_layers, filters_size,
                                                  filters_numbers,
                                                  ndense_units0, weight_decay, dropout, init_branch=init_branches['depth'])
            else:
                depthBranch = UWYHSemiNet.build_3Dbranch("depthBranch", ndense_units=ndense_units0,
                                                         init_branch=init_branches['depth'])
        else:
            if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                if gaitset:
                    ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1,
                                                                input_shape=input_shapes[0], ndense_units=ndense_units0,
                                                                init_branch=init_branches['of'])
                else:
                    ofBranch = UWYHNet.buildBranchLReLU("ofBranch", input_shapes[0], number_convolutional_layers, filters_size,
                                               filters_numbers, ndense_units0, weight_decay, dropout,
                                                    init_branch=init_branches['of'], alpha=alpha)
            else:
                if gaitset:
                    ofBranch = UWYHSemiNet.build_gaitset_branch("ofBranch", input_layer=ofinput1,
                                                                input_shape=input_shapes[0], ndense_units=ndense_units0,
                                                                init_branch=init_branches['of'])
                else:
                    ofBranch = UWYHSemiNet.build_3DbranchLReLU("ofBranch", ndense_units=ndense_units0,
                                                      init_branch=init_branches['of'], alpha=alpha)

            if not use3D:
                if gaitset:
                    grayBranch = UWYHSemiNet.build_gaitset_branch("grayBranch", input_layer=grayinput1,
                                                                  input_shape=input_shapes[1],
                                                                  ndense_units=ndense_units0,
                                                                  init_branch=init_branches['gray'])
                else:
                    grayBranch = UWYHNet.buildBranchLReLU("grayBranch", input_shapes[1], number_convolutional_layers,
                                                 filters_size,
                                                 filters_numbers,
                                                 ndense_units0, weight_decay, dropout,
                                                 init_branch=init_branches['gray'], alpha=alpha)
            else:
                grayBranch = UWYHSemiNet.build_3DbranchLReLU("grayBranch", ndense_units=ndense_units0,
                                                        init_branch=init_branches['gray'], alpha=alpha)

            if not use3D:
                if gaitset:
                    depthBranch = UWYHSemiNet.build_gaitset_branch("depthBranch", input_layer=depthinput1,
                                                                  input_shape=input_shapes[2],
                                                                  ndense_units=ndense_units0,
                                                                  init_branch=init_branches['depth'])
                else:
                    depthBranch = UWYHNet.buildBranchLReLU("depthBranch", input_shapes[2], number_convolutional_layers,
                                                  filters_size,
                                                  filters_numbers,
                                                  ndense_units0, weight_decay, dropout,
                                                  init_branch=init_branches['depth'], alpha=alpha)
            else:
                depthBranch = UWYHSemiNet.build_3DbranchLReLU("depthBranch", ndense_units=ndense_units0,
                                                         init_branch=init_branches['depth'], alpha=alpha)

        if freeze_branches:
            for layer in ofBranch.layers:
                layer.trainable = False

            for layer in grayBranch.layers:
                layer.trainable = False

            for layer in depthBranch.layers:
                layer.trainable = False

        # Process data: branch 1
        # ----------------------
        if gaitset:
            ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofBranch, of_use1])
        else:
            ofout1 = ofBranch(ofinput1)
            if normbfmerge:
                ofout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2of")(ofout1)
            ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofout1, of_use1])

        if gaitset:
            grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayBranch, gray_use1])
        else:
            grayout1 = grayBranch(grayinput1)
            if normbfmerge:
                grayout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2gr")(grayout1)
            grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayout1, gray_use1])

        if gaitset:
            depthout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_depth1")([depthBranch, depth_use1])
        else:
            depthout1 = depthBranch(depthinput1)
            if normbfmerge:
                depthout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2dep")(depthout1)
            depthout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_depth1")([depthout1, depth_use1])

        # Combine layers
        # agg1 = Maximum(name="fusion")([ofout1, grayout1, depthout1])
        agg1 = fMerge(name="fusion")([ofout1, grayout1, depthout1])  # Custom merge can be used
        # L2 normalize embeddings
        agg1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="signature")(agg1)
        outsignature = agg1

        if add_extra_dense:
            if fActivation == 'relu':
                x = layers.Dense(ndense_units[1], activation='relu', activity_regularizer=regularizers.l2(1e-3),
                             name="code")(agg1)
            else:
                x = layers.Dense(ndense_units[1], activation=None, activity_regularizer=regularizers.l2(1e-3),
                                 name="code")(agg1)
                x = LeakyReLU(alpha=alpha)(x)

            agg1 = layers.Dropout(dropout, name="dropcode")(x)

        # Build composed model
        the_inputs = [ofinput1, of_use1, grayinput1, gray_use1, depthinput1, depth_use1]

        useTriplet = loss_weights[0] > 0

        if nclasses > 0:
            if gaitset:
                agg1 = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(agg1)
                agg1 = layers.Flatten()(agg1)
            classprob = layers.Dense(nclasses, activation='softmax', name="classprob")(agg1)
            # mjmarin: uncommenting next lines implies modifying data generator to output only one-hot labels, instead of current two outputs
#            if not useTriplet:
#                outputs = [classprob]
#            else:
            outputs = [outsignature, classprob]

            # Add auxiliar classifiers.
            if aux_losses:
                classprob_of = layers.Dense(nclasses, activation='softmax', name="classprob_of")(ofout1)
                classprob_gray = layers.Dense(nclasses, activation='softmax', name="classprob_gray")(grayout1)
                classprob_depth = layers.Dense(nclasses, activation='softmax', name="classprob_depth")(depthout1)

                outputs.append(classprob_of)
                outputs.append(classprob_gray)
                outputs.append(classprob_depth)

            loss_weights = loss_weights   # ???

            # mjmarin: this can be done much better if data generator is updated
#            if not useTriplet:
#                metrics = ['acc']
#            else:
            metrics = [[], 'acc']

            if smoothlabels == 0:
                if not useTriplet:
                    losses = ['categorical_crossentropy']
                else:
                    losses = [triplet_loss(margin=margin), 'categorical_crossentropy']
                # Add auxiliar losses.
                if aux_losses:
                    losses.append('categorical_crossentropy')
                    losses.append('categorical_crossentropy')
                    losses.append('categorical_crossentropy')
                    metrics.append('acc')
                    metrics.append('acc')
                    metrics.append('acc')
            else:
                if not useTriplet:
                    losses = [tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels)]
                else:
                    losses = [triplet_loss(margin=margin),
                          tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels)]
                # Add auxiliar losses.
                if aux_losses:
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))

            if len(loss_weights) < len(losses):
                import numpy as np
                loss_weights_ = np.zeros(len(losses)) + loss_weights[-1]
                loss_weights_[0:len(loss_weights)] = np.asarray(loss_weights)
                loss_weights = list(loss_weights_)

        else:
            outputs = outsignature
            losses = triplet_loss(margin=margin)
            loss_weights = 1.0
            metrics = []

        print("inputs: ", the_inputs, flush=True)
        print("outputs: ", outputs, flush=True)
        model = Model(inputs=the_inputs, outputs=outputs)

        # Check compatible types
        model_type = model.dtype
        same_type = model_type in str(type(margin))
        #import pdb; pdb.set_trace()
        if not same_type:
            import numpy as np
            if model_type == "float32":
                margin = np.float32(margin)
            else:
                margin = np.float64(margin)

            if nclasses > 0:
                if useTriplet:
                    losses[0].margin = margin
            else:
                losses.margin = margin

        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

        return model

    @staticmethod
    def compile_hard(model, optimizer, loss_weights, margin):
        losses = [tfa.losses.TripletHardLoss(margin=margin), model.loss[1]]
        metrics = [[], 'acc']
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        return model

    @staticmethod
    def build_or_load(input_shapes, number_convolutional_layers, filters_size, filters_numbers, ndense_units=512,
                      weight_decay=1e-4, dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2, nclasses=0,
                      loss_weights=[1.0, 1.0], initnet="", freeze_convs=False, use3D=False, smoothlabels=0,
                      freeze_all=False, postriplet=1, init_branches=None, freeze_branches=False, aux_losses=False,
                      fMerge=Maximum, normbfmerge=False, fActivation='relu', alpha=0.3, gaitset=False):

        if gaitset:
            fActivation = 'leaky'

        if initnet == "":
            model = UWYHSemiNet3Mods.build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
                                           ndense_units, weight_decay, dropout, optimizer, margin, nclasses,
                                           loss_weights, use3D=use3D, smoothlabels=smoothlabels, postriplet=postriplet,
                                           init_branches=init_branches, freeze_branches=freeze_branches,
                                           aux_losses=aux_losses, fMerge=fMerge, normbfmerge=normbfmerge,
                                           fActivation=fActivation, alpha=alpha, gaitset=gaitset)
        else:
            model = None
            model_base = UWYHSemiNet3Mods.loadnet(initnet)
            # Check top layer and change if needed
            # try:
            if True:
                l = model_base.get_layer("classprob")
                if l.units != nclasses:
                    print("Surgery needed: {} vs {}".format(l.units, nclasses))
                    import deepdish as dd

                    fconfig = UWYHSemiNet3Mods.get_netconfig_filename(initnet)
                    netconfig = dd.io.load(fconfig)
                    netconfig["nclasses"] = nclasses
                    netconfig["loss_weights"] = loss_weights
                    netconfig["dropout"] = dropout
                    netconfig["margin"] = np.float32(margin)
                    netconfig["optimizer"] = optimizer
                    netconfig["use3D"] = use3D
                    netconfig["postriplet"] = postriplet
                    netconfig["fMerge"] = fMerge
                    netconfig["normbfmerge"] = normbfmerge
                    netconfig["fActivation"] = fActivation
                    netconfig["alpha"] = alpha

                    model = UWYHSemiNet3Mods.build_by_config(netconfig)

                    filewes = UWYHSemiNet3Mods.get_weights_filename(initnet)

                    model.load_weights(filewes, by_name=True, skip_mismatch=True)  # Load compatible weights
                else:
                    model = model_base

            # Check if freeze some weights
            if freeze_convs or freeze_all:
                if gaitset:
                    model = UWYHSemiNet3Mods.build(input_shapes, number_convolutional_layers, filters_size,
                                                   filters_numbers,
                                                   ndense_units, weight_decay, dropout, optimizer, margin, nclasses,
                                                   loss_weights, use3D=use3D, smoothlabels=smoothlabels,
                                                   postriplet=postriplet,
                                                   init_branches=init_branches, freeze_branches=freeze_branches,
                                                   aux_losses=aux_losses, fMerge=fMerge, normbfmerge=normbfmerge,
                                                   fActivation=fActivation, alpha=alpha, gaitset=gaitset)
                    filewes = UWYHSemiNet.get_weights_filename(initnet)
                    model.load_weights(filewes, by_name=True, skip_mismatch=True)  # Load compatible weights
                    for layer_ix in range(len(model.layers) - 1):
                        if freeze_all:
                            if hasattr(model.layers[layer_ix], 'layer'):
                                model.layers[layer_ix].layer.trainable = False
                            model.layers[layer_ix].trainable = False
                else:
                    seq1 = model.get_layer("ofBranch")
                    seq2 = model.get_layer("grayBranch")
                    seq3 = model.get_layer("depthBranch")

                    for layer in seq1.layers:
                        if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                            layer.trainable = False

                    for layer in seq2.layers:
                        if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                            layer.trainable = False

                    for layer in seq3.layers:
                        if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                            layer.trainable = False

        # except:
        # This is not a classification network, just verification
        print("Alright")

        return model



class UWYHSemiNet3Mods_Fran(UWYHSemiNet):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loadnet(netpath: str):

        model = None

        from tensorflow.keras.models import load_model
        import deepdish as dd

        try:
            model = load_model(netpath, custom_objects={"triplet_loss": triplet_loss(), "MatMul": MatMul()}, compile=False)
        except:
            print("థ_థ")
            bdir = osp.dirname(netpath)
            fconfig = osp.join(bdir, "model-config.hdf5")
            netconfig = dd.io.load(fconfig)

            model = mj_buildnet_by_config(netconfig, UWYHSemiNet3Mods_Fran.build)

            filewes = UWYHSemiNet.get_weights_filename(netpath)

            model.load_weights(filewes, by_name=True)

        return model

    @staticmethod
    def build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
              ndense_units=512, weight_decay=1e-4,
              dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2,
              nclasses=0, loss_weights=[1.0, 1.0], use3D=False, smoothlabels=0,
              postriplet=1, init_branches=None, freeze_branches=False, aux_losses=False, fMerge=Maximum,
              normbfmerge=False, fActivation='relu', alpha=0.3):
        """

        :param input_shapes: tuple ((50,60,60), (25,60,60))
        :param number_convolutional_layers:
        :param filters_size:
        :param filters_numbers:
        :param ndense_units:
        :param weight_decay:
        :param dropout:
        :param optimizer:
        :param margin:
        :param nclasses: if 0, cross-entropy is not used
        :param loss_weights: if any position is 0, that one is not used
        :return: the model
        """

        # TODO implement use of 'postriplet'

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        if type(ndense_units) is list:
            ndense_units0 = ndense_units[0]
            add_extra_dense = len(ndense_units) > 1
        else:
            ndense_units0 = ndense_units
            add_extra_dense = False

        if init_branches is None:
            init_branches = {'of': '', 'gray': '', 'depth': ''}

        if fActivation == 'relu':
            if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                ofBranch = UWYHNet.buildBranch("ofBranch", input_shapes[0], number_convolutional_layers, filters_size,
                                               filters_numbers,
                                               ndense_units0, weight_decay, dropout, init_branch=init_branches['of'])
            else:
                ofBranch = UWYHSemiNet.build_3Dbranch("ofBranch", ndense_units=ndense_units0,
                                                      init_branch=init_branches['of'])

            if not use3D:
                grayBranch = UWYHNet.buildBranch("grayBranch", input_shapes[1], number_convolutional_layers, filters_size,
                                                 filters_numbers,
                                                 ndense_units0, weight_decay, dropout, init_branch=init_branches['gray'])
            else:
                grayBranch = UWYHSemiNet.build_3Dbranch("grayBranch", ndense_units=ndense_units0,
                                                        init_branch=init_branches['gray'])

            if not use3D:
                depthBranch = UWYHNet.buildBranch("depthBranch", input_shapes[2], number_convolutional_layers, filters_size,
                                                  filters_numbers,
                                                  ndense_units0, weight_decay, dropout, init_branch=init_branches['depth'])
            else:
                depthBranch = UWYHSemiNet.build_3Dbranch("depthBranch", ndense_units=ndense_units0,
                                                         init_branch=init_branches['depth'])
        else:
            if input_shapes[0][0] == 50 or not use3D:  # This should be OF
                ofBranch = UWYHNet.buildBranchLReLU("ofBranch", input_shapes[0], number_convolutional_layers, filters_size,
                                               filters_numbers, ndense_units0, weight_decay, dropout,
                                                    init_branch=init_branches['of'], alpha=alpha)
            else:
                ofBranch = UWYHSemiNet.build_3DbranchLReLU("ofBranch", ndense_units=ndense_units0,
                                                      init_branch=init_branches['of'], alpha=alpha)

            if not use3D:
                grayBranch = UWYHNet.buildBranchLReLU("grayBranch", input_shapes[1], number_convolutional_layers,
                                                 filters_size,
                                                 filters_numbers,
                                                 ndense_units0, weight_decay, dropout,
                                                 init_branch=init_branches['gray'], alpha=alpha)
            else:
                grayBranch = UWYHSemiNet.build_3DbranchLReLU("grayBranch", ndense_units=ndense_units0,
                                                        init_branch=init_branches['gray'], alpha=alpha)

            if not use3D:
                depthBranch = UWYHNet.buildBranchLReLU("depthBranch", input_shapes[2], number_convolutional_layers,
                                                  filters_size,
                                                  filters_numbers,
                                                  ndense_units0, weight_decay, dropout,
                                                  init_branch=init_branches['depth'], alpha=alpha)
            else:
                depthBranch = UWYHSemiNet.build_3DbranchLReLU("depthBranch", ndense_units=ndense_units0,
                                                         init_branch=init_branches['depth'], alpha=alpha)

        if freeze_branches:
            for layer in ofBranch.layers:
                layer.trainable = False

            for layer in grayBranch.layers:
                layer.trainable = False

            for layer in depthBranch.layers:
                layer.trainable = False

        # Process data: branch 1
        # ----------------------
        ofinput1 = Input(shape=input_shapes[0], name='ofinput1')
        of_use1 = Input(shape=(1), name='ofuse1')
        grayinput1 = Input(shape=input_shapes[1], name='grayinput1')
        gray_use1 = Input(shape=(1), name='grayuse1')
        depthinput1 = Input(shape=input_shapes[2], name='depthinput1')
        depth_use1 = Input(shape=(1), name='depthuse1')

        ofout1 = ofBranch(ofinput1)
        if normbfmerge:
            ofout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2of")(ofout1)
        ofout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_of1")([ofout1, of_use1])

        grayout1 = grayBranch(grayinput1)
        if normbfmerge:
            grayout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2gr")(grayout1)
        grayout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_gray1")([grayout1, gray_use1])

        depthout1 = depthBranch(depthinput1)
        if normbfmerge:
            depthout1 = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="nrmbfl2dep")(depthout1)
        depthout1 = layers.Lambda(mj_tensor_times_scalar, name="gate_depth1")([depthout1, depth_use1])

        # Combine layers
        # agg1 = Maximum(name="fusion")([ofout1, grayout1, depthout1])
        agg1 = fMerge(name="fusion")([ofout1, grayout1, depthout1])  # Custom merge can be used
        # L2 normalize embeddings
        outsignature = layers.Lambda(lambda x_: tf.math.l2_normalize(x_, axis=1), name="signature")(agg1)

        if add_extra_dense:
            if fActivation == 'relu':
                x = layers.Dense(ndense_units[1], activation='relu', activity_regularizer=regularizers.l2(1e-3),
                             name="code")(agg1)
            else:
                x = layers.Dense(ndense_units[1], activation=None, activity_regularizer=regularizers.l2(1e-3),
                                 name="code")(agg1)
                x = LeakyReLU(alpha=alpha)(x)

            agg1 = layers.Dropout(dropout, name="dropcode")(x)

        # Build composed model
        the_inputs = [ofinput1, of_use1, grayinput1, gray_use1, depthinput1, depth_use1]

        useTriplet = loss_weights[0] > 0

        if nclasses > 0:
            classprob = layers.Dense(nclasses, activation='softmax', name="classprob")(agg1)
            # mjmarin: uncommenting next lines implies modifying data generator to output only one-hot labels, instead of current two outputs
#            if not useTriplet:
#                outputs = [classprob]
#            else:
            outputs = [outsignature, classprob]

            # Add auxiliar classifiers.
            if aux_losses:
                classprob_of = layers.Dense(nclasses, activation='softmax', name="classprob_of")(ofout1)
                classprob_gray = layers.Dense(nclasses, activation='softmax', name="classprob_gray")(grayout1)
                classprob_depth = layers.Dense(nclasses, activation='softmax', name="classprob_depth")(depthout1)

                outputs.append(classprob_of)
                outputs.append(classprob_gray)
                outputs.append(classprob_depth)

            loss_weights = loss_weights   # ???

            # mjmarin: this can be done much better if data generator is updated
#            if not useTriplet:
#                metrics = ['acc']
#            else:
            metrics = [[], 'acc']

            if smoothlabels == 0:
                if not useTriplet:
                    losses = ['categorical_crossentropy']
                else:
                    losses = [triplet_loss(margin=margin), 'categorical_crossentropy']
                # Add auxiliar losses.
                if aux_losses:
                    losses.append('categorical_crossentropy')
                    losses.append('categorical_crossentropy')
                    losses.append('categorical_crossentropy')
                    metrics.append('acc')
                    metrics.append('acc')
                    metrics.append('acc')
            else:
                if not useTriplet:
                    losses = [tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels)]
                else:
                    losses = [triplet_loss(margin=margin),
                          tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels)]
                # Add auxiliar losses.
                if aux_losses:
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))
                    losses.append(tf.losses.CategoricalCrossentropy(label_smoothing=smoothlabels))

            if len(loss_weights) < len(losses):
                import numpy as np
                loss_weights_ = np.zeros(len(losses)) + loss_weights[-1]
                loss_weights_[0:len(loss_weights)] = np.asarray(loss_weights)
                loss_weights = list(loss_weights_)

        else:
            outputs = outsignature
            losses = triplet_loss(margin=margin)
            loss_weights = 1.0
            metrics = []

        model = Model(inputs=the_inputs, outputs=outputs)

        # Check compatible types
        model_type = model.dtype
        same_type = model_type in str(type(margin))
        #import pdb; pdb.set_trace()
        if not same_type:
            import numpy as np
            if model_type == "float32":
                margin = np.float32(margin)
            else:
                margin = np.float64(margin)

            if nclasses > 0:
                if useTriplet:
                    losses[0].margin = margin
            else:
                losses.margin = margin

        print("losses: ", losses, flush=True)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

        return model

    @staticmethod
    def build_or_load(input_shapes, number_convolutional_layers, filters_size, filters_numbers, ndense_units=512,
                      weight_decay=1e-4, dropout=0.4, optimizer=optimizers.SGD(0.001, 0.9), margin=0.2, nclasses=0,
                      loss_weights=[1.0, 1.0], initnet="", freeze_convs=False, use3D=False, smoothlabels=0,
                      freeze_all=False, postriplet=1, init_branches=None, freeze_branches=False, aux_losses=False,
                      fMerge=Maximum, normbfmerge=False, fActivation='relu', alpha=0.3):

        if initnet == "":
            model = UWYHSemiNet3Mods_Fran.build(input_shapes, number_convolutional_layers, filters_size, filters_numbers,
                                           ndense_units, weight_decay, dropout, optimizer, margin, nclasses,
                                           loss_weights, use3D=use3D, smoothlabels=smoothlabels, postriplet=postriplet,
                                           init_branches=init_branches, freeze_branches=freeze_branches,
                                           aux_losses=aux_losses, fMerge=fMerge, normbfmerge=normbfmerge,
                                           fActivation=fActivation, alpha=alpha)
        else:
            model = None
            model_base = UWYHSemiNet3Mods_Fran.loadnet(initnet)
            # Check top layer and change if needed
            # try:
            if True:
                l = model_base.get_layer("classprob")
                if l.units != nclasses:
                    print("Surgery needed: {} vs {}".format(l.units, nclasses))
                    import deepdish as dd

                    fconfig = UWYHSemiNet3Mods_Fran.get_netconfig_filename(initnet)
                    netconfig = dd.io.load(fconfig)
                    netconfig["nclasses"] = nclasses
                    netconfig["loss_weights"] = loss_weights
                    netconfig["dropout"] = dropout
                    netconfig["margin"] = np.float32(margin)
                    netconfig["optimizer"] = optimizer
                    netconfig["use3D"] = use3D
                    netconfig["postriplet"] = postriplet
                    netconfig["fMerge"] = fMerge
                    netconfig["normbfmerge"] = normbfmerge
                    netconfig["fActivation"] = fActivation
                    netconfig["alpha"] = alpha

                    model = UWYHSemiNet3Mods_Fran.build_by_config(netconfig)

                    filewes = UWYHSemiNet3Mods_Fran.get_weights_filename(initnet)

                    model.load_weights(filewes, by_name=True, skip_mismatch=True)  # Load compatible weights
                else:
                    model = model_base

            # Check if freeze some weights
            if freeze_convs or freeze_all:
                seq1 = model.get_layer("ofBranch")
                seq2 = model.get_layer("grayBranch")
                seq3 = model.get_layer("depthBranch")

                for layer in seq1.layers:
                    if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                        layer.trainable = False

                for layer in seq2.layers:
                    if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                        layer.trainable = False

                for layer in seq3.layers:
                    if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D:
                        layer.trainable = False

        # except:
        # This is not a classification network, just verification
        print("Alright")

        return model


# ========================== MAIN ========================
if __name__ == "__main__":

    filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
    filters_numbers = [96, 192, 512, 512]
    weight_decay = 0.00005

    if False:
        ofbranch = UWYHNet.buildBranch("ofBranch", (50, 60, 60), len(filters_size), filters_size=filters_size,
                                       filters_numbers=filters_numbers)
        ofbranch.summary()
    else:
        grayBranch3D = UWYHSemiNet.build_3Dbranch("branch3D", input_shape=(25, 60, 60, 1))

        grayBranch3D.summary()

        # import numpy as np
        # M = np.random.uniform(-1, 1, (4, 25, 60, 60, 1)).astype(np.float32)
        # res = grayBranch3D(M)

    print("=====================================================")
    if False:
        model = UWYHSemiNet.build([(50, 60, 60), (25, 60, 60, 1)], len(filters_size), filters_size=filters_size,
                                  filters_numbers=filters_numbers, nclasses=23, loss_weights=[1.0, 0.1],
                                  dropout=[0, 0.4],
                                  ndense_units=[1024, 256], use3D=True, postriplet=2)
    elif False:
        model = UWYHSemiNet.build((50, 60, 60), len(filters_size), filters_size=filters_size,
                                  filters_numbers=filters_numbers, nclasses=0, loss_weights=None,
                                  dropout=0.4, ndense_units=256)
    else:
        model = UWYHSemiNet3Mods.build([(50, 60, 60), (25, 60, 60, 1), (25, 60, 60, 1)], len(filters_size),
                                       filters_size=filters_size,
                                       filters_numbers=filters_numbers, nclasses=23, loss_weights=[0.0, 0.1],
                                       ndense_units=[1024, 256], use3D=True, fMerge=Average,
                                       normbfmerge=True)

    model.summary()  # ఠ  థ

    ಠ_ಠ = "Cool!"
    print(ಠ_ಠ)
    print("done!")
