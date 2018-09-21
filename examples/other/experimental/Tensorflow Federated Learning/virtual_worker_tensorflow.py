# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Import pysyft
import syft as sy

# Helper libraries
import numpy as np
import os

BATCH_SIZE = 32
EPOCHS = 5

x = sy.array([1]) # Why do I need this in order for local_worker to exist?

alice = sy.VirtualWorker(id="alice") # If I get here without calling sy.array first it raises an error
bob = sy.VirtualWorker(id="bob")

workers = [bob, alice]

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

num_workers = len(workers)
train_images = np.split(train_images, num_workers)
train_labels = np.split(train_labels, num_workers)

SHUFFLE_SIZE = train_images[0].shape[0]

train_distributed_dataset = []
for images, labels, worker in zip(train_images, train_labels, workers):
    images = sy.array(images)
    labels = sy.array(labels)
    train_distributed_dataset.append((images.send(worker), labels.send(worker)))

def cast(images, labels):
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int64)
    return images, labels

def train_input_fn(images, labels):
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(SHUFFLE_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(cast)
        return dataset.make_one_shot_iterator().get_next()

def test_input_fn():
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        # For some reason I need this batch so that the map function gets parallelized,
        # it does nothing because it's batching with the size of the whole test set
        dataset = dataset.batch(test_images.shape[0])
        dataset = dataset.map(cast)
        return dataset.make_one_shot_iterator().get_next()

def model_fn(features, labels, mode, params):
    flatten_layer = tf.layers.flatten(features, name='flatten')

    dense_layer = tf.layers.dense(flatten_layer, 128, activation=tf.nn.relu, name='relu')

    logits = tf.layers.dense(dense_layer, params['n_classes'], name='logits')

    # Compute predictions
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='accuracy')
    tf.summary.scalar('accuracy', accuracy[1])
    metrics = {'accuracy': accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Train
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# Create the estimator
classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir='logs_dir', params={'n_classes': 10})

# Train the model
for epoch in range(EPOCHS):
    for count, (images, labels) in enumerate(train_distributed_dataset):
        worker = images.location
        classifier.train(input_fn=lambda: train_input_fn(images, labels))

# Evaluate the model
eval_result = classifier.evaluate(input_fn=test_input_fn)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
