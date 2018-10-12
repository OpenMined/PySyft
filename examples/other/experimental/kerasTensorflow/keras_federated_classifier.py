# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import numpy as np
from time import time
from syft.core.frameworks.tensorflow import federated_averaging_optimizer

flags = tf.app.flags
flags.DEFINE_integer(
    "task_index",
    None,
    "Worker task index, should be >= 0. task_index=0 is "
    "the master worker task the performs the variable "
    "initialization ",
)
flags.DEFINE_integer(
    "train_steps", 1000, "Number of (global) training steps to perform"
)
flags.DEFINE_string(
    "ps_hosts", "localhost:2222", "Comma-separated list of hostname:port pairs"
)
flags.DEFINE_string(
    "worker_hosts",
    "localhost:2223,localhost:2224",
    "Comma-separated list of hostname:port pairs",
)
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

# Steps between averages
INTERVAL_STEPS = 100

# Disable GPU to avoid OOM issues (could enable it for just one of the workers)
# Not necessary if workers are hosted in different machines
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
print("job name = %s" % FLAGS.job_name)
print("task index = %d" % FLAGS.task_index)

# Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")

# Get the number of workers.
num_workers = len(worker_spec)

cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# The server will block here
if FLAGS.job_name == "ps":
    server.join()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

is_chief = FLAGS.task_index == 0

# We are not telling the MonitoredSession who is the chief so we need to
# prevent non-chief workers from saving checkpoints or summaries.
if is_chief:
    checkpoint_dir = "logs_dir/{}".format(time())
else:
    checkpoint_dir = None

worker_device = "/job:worker/task:%d" % FLAGS.task_index

# Place all ops in the local worker by default
with tf.device(worker_device):
    global_step = tf.train.get_or_create_global_step()

    # Define the model
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu, name="relu"),
            keras.layers.Dense(10, activation=tf.nn.softmax, name="softmax"),
        ]
    )

    # Get placeholder for the labels
    y = tf.placeholder(tf.float32, shape=[None], name="labels")

    # Store reference to the output of the model
    predictions = model.output

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, predictions)
        )

    tf.summary.scalar("cross_entropy", loss)

    with tf.name_scope("train"):
        # Define a device setter which will place a global copy of trainable variables
        # in the parameter server.
        device_setter = tf.train.replica_device_setter(
            worker_device=worker_device, cluster=cluster
        )
        # Define our custom optimizer
        optimizer = federated_averaging_optimizer.FederatedAveragingOptimizer(
            tf.train.AdamOptimizer(0.001),
            replicas_to_aggregate=num_workers,
            interval_steps=INTERVAL_STEPS,
            is_chief=is_chief,
            device_setter=device_setter,
        )
        train_op = optimizer.minimize(loss, global_step=global_step)
        # Define the hook which initializes the optimizer
        federated_average_hook = optimizer.make_session_run_hook()

    # ConfiProto for our session
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index],
    )

    # We need to let the MonitoredSession initialize the variables
    keras.backend.manual_variable_initialization(True)
    # Define the training feed
    train_feed = {model.inputs[0]: train_images, y: train_labels}

    # Hook to log training progress
    class _LoggerHook(tf.train.SessionRunHook):
        def before_run(self, run_context):
            return tf.train.SessionRunArgs(global_step)

        def after_run(self, run_context, run_values):
            step = run_values.results
            if step % 100 == 0:
                print(f"Iter {step}/{FLAGS.train_steps}")

    with tf.train.MonitoredTrainingSession(
        master=server.target,
        checkpoint_dir=checkpoint_dir,
        hooks=[
            tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
            _LoggerHook(),
            federated_average_hook,
        ],
        save_checkpoint_steps=100,
        config=sess_config,
    ) as mon_sess:
        keras.backend.set_session(mon_sess)
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=train_feed)
