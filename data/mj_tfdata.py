"""
Handles data
(c) MJMJ/2020
"""
__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'March 2020'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def mj_parse_tfr_of_all(example):
   "Parse TFExample records"
   image_feature_description = {
      'height': tf.compat.v1.FixedLenFeature([], tf.int64),
      'width': tf.compat.v1.FixedLenFeature([], tf.int64),
      'depth': tf.compat.v1.FixedLenFeature([], tf.int64),
      'data': tf.compat.v1.FixedLenFeature([], tf.string),
      'labels': tf.compat.v1.FixedLenFeature([], tf.int64),
      'set': tf.compat.v1.FixedLenFeature([], tf.int64),
      'videoId': tf.compat.v1.FixedLenFeature([], tf.int64),
      'compressFactor': tf.compat.v1.FixedLenFeature([], tf.int64),
      'gait': tf.compat.v1.FixedLenFeature([], tf.int64),
      'mirrors': tf.compat.v1.FixedLenFeature([], tf.int64),

   }

   parsed = tf.compat.v1.parse_example(example, image_feature_description)

   parsed['data'] = tf.compat.v1.decode_raw(parsed['data'], tf.int16)
   #parsed['data'] = tf.math.divide(tf.cast(parsed['data'], tf.float32), 100.0)

   parsed['data'] = tf.reshape(parsed['data'], shape=(-1, 50, 60, 60))

   return parsed

def mj_parse_tfr_of_id(example):
    return mj_parse_tfr_of(example, withid=True)

def mj_parse_tfr_of(example, withid=False):
  "Parse TFExample records."
  image_feature_description = {
      'height': tf.compat.v1.FixedLenFeature([], tf.int64),
      'width': tf.compat.v1.FixedLenFeature([], tf.int64),
      'depth': tf.compat.v1.FixedLenFeature([], tf.int64),
      'data': tf.compat.v1.FixedLenFeature([], tf.string),
      'labels': tf.compat.v1.FixedLenFeature([], tf.int64),
      'set': tf.compat.v1.FixedLenFeature([], tf.int64),
      'videoId': tf.compat.v1.FixedLenFeature([], tf.int64),
      'compressFactor': tf.compat.v1.FixedLenFeature([], tf.int64),
      'gait': tf.compat.v1.FixedLenFeature([], tf.int64),

  }

  parsed = tf.compat.v1.parse_example(example, image_feature_description)

  parsed['data'] = tf.compat.v1.decode_raw(parsed['data'], tf.int16)
  parsed['data'] = tf.math.divide(tf.cast(parsed['data'], tf.float32), 100.0)

  parsed['data'] = tf.reshape(parsed['data'], shape=(-1, 50, 60, 60))

  if withid:
    return parsed['data'], parsed["labels"][0], parsed['videoId'][0]
  else:
    return parsed['data'], parsed["labels"][0]


def mj_loadSingleGaitOFTFrecord(filepath, sess, graph=None, withid=False, allinfo=False):
    """
    Loads and parse the OF data stored in a single TF record file. It assumes: 60x60x50
    :param filepath: full path to record file
    :param tfconfig: variable returned by tf.compat.v1.ConfigProto()
    :return: tuple with loaded data (OF, label, videoId)
    """

    data = None
    #with tf.compat.v1.Session(config=tfconfig) as sess:
    #with graph.as_default():
    #with sess.as_default():
    with tf.compat.v1.Session() as sess:
        dataset = tf.data.TFRecordDataset([filepath])
        dataset = dataset.batch(1)
        if not allinfo:
            if withid:
                dataset = dataset.map(map_func=mj_parse_tfr_of_id)
            else:
                dataset = dataset.map(map_func=mj_parse_tfr_of)
        else:
            dataset = dataset.map(map_func=mj_parse_tfr_of_all)

        dataset = tf.data.experimental.get_single_element(dataset)
        data = sess.run(dataset)

        del dataset

    return data


# ============================== MAIN ========================
if __name__ == "__main__":
    filepath = '/home/mjmarin/datatum/matimdbtum_gaid_N150_of25_60x60_lite/0/output002100.record'

    tfconfig = tf.compat.v1.ConfigProto()

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(config=tfconfig, graph=graph)
        # with tf.device('/gpu:' + str(1)):
        with sess.as_default():

            data = mj_loadSingleGaitOFTFrecord(filepath, sess, graph, withid=False)

            print(data[0].shape)
            print(data[1])
            if len(data) > 2:
                print(data[2])

            import matplotlib.pyplot as plt

            import numpy as np
            sample = np.squeeze(data[0])

            pos_idx = 0  # Posicion de la secuencia

            sample_x = np.squeeze(sample[pos_idx * 2,])
            sample_y = np.squeeze(sample[pos_idx * 2 + 1,])

            plt.figure(1)
            plt.imshow(sample_x)

            plt.figure(2)
            plt.imshow(sample_y)

            plt.show()

    print("Done!")
