# Loss functions

__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'March 2020'

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.layers import Layer

def mj_l2normalize(x, axis=1):
    dnorm = K.l2_normalize(x, axis=axis)
    return dnorm

# Version 1 of L1-smooth
HUBER_DELTA = 0.5
def mj_smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def mj_smoothL1bis(trash, y):
    y_true = y[0]
    y_pred = y[1]
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


# def mj_sim_loss(trash, vinput):
#     v1 = vinput[0]
#     v2 = vinput[1]
#
#     return CosineSimilarity(v1, v2)


class PairLossLayer(Layer):
    """ L1-smooth loss"""

    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        super(PairLossLayer, self).__init__(**kwargs)

    def pair_loss(self, inputs):
        y_true, y_pred = inputs
        x = K.abs(y_true - y_pred)
        x = K.switch(x < self.alpha, 0.5 * x ** 2, self.alpha * (x - 0.5 * self.alpha))
        return K.sum(x)

    def call(self, inputs):
        loss = self.pair_loss(inputs)
        self.add_loss(loss)
        return loss

    # This is needed as we have an extra input param during init
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
        })
        return config


class VerifLossLayer(Layer):
    """ Tukey's loss """

    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        super(VerifLossLayer, self).__init__(**kwargs)

    def pair_loss(self, inputs):
        y_true, y_pred, labels = inputs
        res2 = K.square(y_true - y_pred)
        m = self.alpha

        # Separate between positive and negative samples
        labels = tf.squeeze(labels, -1)

        posIdx = tf.where(K.equal(labels, 1))
        posIdx = tf.squeeze(posIdx)

        negIdx = tf.where(K.equal(labels, 0))
        negIdx = tf.squeeze(negIdx)
                
        rpos = tf.gather(res2, posIdx)
        rneg = tf.gather(res2, negIdx)

        # Partial loss
        xpos = 0.5 * K.sum(rpos)                                              # 0.5 * (L2-distance)^2
        xneg = 0.5 * K.square(K.maximum(0.0, m - K.sqrt(K.sum(rneg)) ))       # L2 with margin

        return xpos + xneg

    def call(self, inputs):
        loss = self.pair_loss(inputs)
        self.add_loss(loss)
        return loss

    # This is needed as we have an extra input param during init
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
        })
        return config


class TripletLossLayer(Layer):
    """ Triplet loss """
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

    # This is needed as we have an extra input param during init
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
        })
        return config