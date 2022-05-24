import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional


def triplet_loss(margin: FloatTensorLike = 1.0):
	@tf.function
	def loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
		"""Computes the triplet loss with semi-hard negative mining.

		Args:
		  y_true: 1-D integer `Tensor` with shape [batch_size] of
			multiclass integer labels.
		  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
			be l2 normalized.
		  margin: Float, margin term in the loss definition.

		Returns:
		  triplet_loss: float scalar with dtype of y_pred.
		"""
		labels, embeddings = y_true, y_pred

		convert_to_float32 = (
			embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
		)
		precise_embeddings = (
			tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
		)

		#n, m, d = tf.shape(embeddings)
		_shape = tf.shape(embeddings)
		n = _shape[0]
		m = _shape[1]
		d = _shape[2]
		labels = tf.transpose(labels, [1, 0])
		#labels = tf.repeat(tf.expand_dims(labels, axis=0), n, axis=0)
		labels = tf.repeat(labels, n, axis=0)
		hp_mask = tf.reshape(tf.expand_dims(labels, axis=1) == tf.expand_dims(labels, axis=2), shape=[-1])
		hn_mask = tf.reshape(tf.expand_dims(labels, axis=1) != tf.expand_dims(labels, axis=2), shape=[-1])

		dist = batch_dist(embeddings)
		dist = tf.reshape(dist, shape=[-1])

		full_hp_dist = tf.reshape(tf.boolean_mask(dist, hp_mask), [n, m, -1, 1])
		full_hn_dist = tf.reshape(tf.boolean_mask(dist, hn_mask), [n, m, 1, -1])

		full_loss_metric = tf.reshape(tf.math.maximum(margin + tf.subtract(full_hp_dist, full_hn_dist), 0.0), [n, -1])

		full_loss_metric_sum = tf.math.reduce_sum(full_loss_metric, axis=1)
		valid_triplets = tf.cast(tf.greater(full_loss_metric, 0.0), dtype=tf.dtypes.float32)
		full_loss_num = tf.math.reduce_sum(valid_triplets, axis=1)

		full_loss_metric_mean = full_loss_metric_sum/full_loss_num

		zero = tf.constant(0.0, dtype=tf.float32)
		where = tf.not_equal(full_loss_num, zero)
		full_loss_metric_mean = tf.where(where, full_loss_metric_mean, tf.zeros_like(full_loss_metric_mean))

		full_loss_metric_mean = tf.reduce_mean(full_loss_metric_mean, axis=0)

		if convert_to_float32:
			return tf.cast(full_loss_metric_mean, embeddings.dtype)
		else:
			return full_loss_metric_mean
	return loss


def batch_dist(x):
	x2 = tf.math.reduce_sum(tf.math.square(x), axis=2)
	dist = tf.expand_dims(x2, axis=2) + tf.expand_dims(x2, axis=1) - 2.0 * tf.linalg.matmul(x, tf.transpose(x, [0, 2, 1]))
	dist = tf.math.maximum(dist, 0.0)
	error_mask = tf.math.less_equal(dist, 0.0)
	dist = tf.math.sqrt(dist + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16)
	dist = tf.math.multiply(dist, tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),)
	return dist

class TripletBatchAllLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with semi-hard negative mining.
	The loss encourages the positive distances (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance
	among which are at least greater than the positive distance plus the
	margin constant (called semi-hard negative) in the mini-batch.
	If no such negative exists, uses the largest negative distance instead.
	See: https://arxiv.org/abs/1503.03832.
	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.
	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin

	def call(self, y_true, y_pred):
		return triplet_loss(self.margin)

	def get_config(self):
		config = {
			"margin": self.margin,
		}
		base_config = super().get_config()
		return {**base_config, **config}


if __name__ == '__main__':
	import numpy as np
	logits = tf.convert_to_tensor([[1.1, 1.2, 1.4], [1.09, 1.21,1.41], [0.25, 0.45, 0.75], [0.23, 0.43, 0.7], [1.5, 2.5, 3.5], [1.55, 2.75, 3.8]], dtype=tf.dtypes.float32)
	labels = tf.convert_to_tensor(np.array([1, 1, 2, 2, 3, 3]), dtype=tf.dtypes.float32)
	loss = triplet_loss(labels, logits)
	print(loss)

