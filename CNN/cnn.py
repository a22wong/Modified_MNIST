# Convolutional Neural Network for Modified MNIST
# Build using TensorFlow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
from resize_data import *

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# Modified MNIST images are 64x64 pixels, and have one color channel
	input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 64, 64, 1]
	# Output Tensor Shape: [batch_size, 64, 64, 32]
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=32,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 64, 64, 32]
	# Output Tensor Shape: [batch_size, 32, 32, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 32, 32, 32]
	# Output Tensor Shape: [batch_size, 32, 32, 64]
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 32, 32, 64]
	# Output Tensor Shape: [batch_size, 16, 16, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 16, 16, 64]
	# Output Tensor Shape: [batch_size, 16 * 16 * 64]
	pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 16 * 16 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 40]
	logits = tf.layers.dense(inputs=dropout, units=40)

	predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=40)
	loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
					labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def reMapY(y):
	# re-map y to 0-40 classes instead of kaggle classes
	kaggle_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

	output = np.array(y)
	for i, j in enumerate(y):
		output[i] = kaggle_classes.index(j)
	return output


def main(unused_argv):

	# load train data
	train_x = "../Data/train_x_1000.csv"
	train_y = "../Data/train_y_1000.csv"
	train_data = np.loadtxt(train_x, delimiter=",").astype(np.float32)
	train_labels = np.loadtxt(train_y, dtype=int, delimiter=",")

	# load eval data
	eval_x = "../Data/eval_x_100.csv"
	eval_y = "../Data/eval_y_100.csv"
	eval_data = np.loadtxt(eval_x, delimiter=",").astype(np.float32)
	eval_labels = np.loadtxt(eval_y, dtype=int, delimiter=",")

	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
			model_fn=cnn_model_fn, model_dir="/tmp/mnist_cnn64_model")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
			tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": train_data},
			y=train_labels,
			batch_size=100,
			num_epochs=None,
			shuffle=True)
	mnist_classifier.train(input_fn=train_input_fn, steps=5000, hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": eval_data},
			y=eval_labels,
			num_epochs=1,
			shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)


if __name__ == "__main__":
	tf.app.run()
