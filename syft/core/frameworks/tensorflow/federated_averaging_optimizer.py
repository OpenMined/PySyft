# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2018 CoMind.
# ==============================================================================

"""Synchronize replicas for FedAvg training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import session_run_hook

# Please note that the parameters from replicas are averaged so you need to
# increase the learning rate according to the number of replicas. This change is
# introduced to be consistent with how parameters are aggregated within a batch
class FederatedAveragingOptimizer(optimizer.Optimizer):
  """Class to synchronize and aggregate model params.

  In a typical synchronous training environment, gradients will be averaged each
  step and then applied to the variables in one shot, after which replicas can
  fetch the new variables and continue. In a federated average training environment,
  model variables will be averaged every 'interval_steps' steps, and then the
  replicas will fetch the new variables and continue training locally. In the
  interval between two average operations, there is no data transfer, which can
  accelerate training.

  The following accumulators/queue are created:
  <empty line>
  * N `parameter accumulators`, one per variable to train. Local variables are
  pushed to them and the chief worker will wait until enough variables are
  collected and then average them. The accumulator will drop all stale variables
  (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.

  The following local variable is created:
  * `global_step`, one per replica. Updated after every average operation.

  The optimizer adds nodes to the graph to collect local variables and pause
  the trainers until variables are updated.
  For the Parameter Server job:
  <empty line>
  1. An accumulator is created for each variable, and each replica pushes the
     local variables into the accumulators.
  2. Each accumulator averages once enough variables (replicas_to_aggregate)
     have been accumulated.
  3. Apply the averaged variables to global variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes a token in the `token_queue`, once for
     each worker replica. The workers can now fetch the token and start
     the next round.

  For the replicas:
  <empty line>
  1. Start a training block: fetch variables and train for "interval_steps" steps.
  2. Once the training block has been computed, push local variables into variable
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the variables, dequeue a token from the token queue and
     continue training. Note that this is effectively a barrier.
  4. Fetch new variables and start the next block.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Wrap the optimizer with fed_avg_optimizer with 50 replicas: at each
  # step the FederatedAveragingOptimizer collects "replicas_to_aggregate" variables
  # before applying the average. Note that if you want to have 2 backup replicas,
  # you can change total_num_replicas=52 and make sure this number matches how
  # many physical replicas you started in your job.
  opt = fed_avg_optimizer.FederatedAveragingOptimizer(opt,
                                                      replicas_to_aggregate=50,
                                                      is_chief=True,
                                                      interval_steps=100,
                                                      device_setter)

  # Some models have startup_delays to help stabilize the model but when using
  # federated_average training, set it to 0.

  # Now you can call 'minimize() normally'
  # train_op = opt.minimize(loss, global_step=global_step)

  # And also, create the hook which handles initialization.
  fed_avg_hook = opt.make_session_run_hook()
  ```

  In the training program, every worker will run the train_op as if not
  averaged or synchronized. Note that if you want to run other ops like
  test op, you should use common session instead of MonitoredSession:

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target,
      hooks=[fed_avg_hook]) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(training_op)
      sess = mon_sess._tf_sess()
      sess.run(testing_op)
  ```
  """

  def __init__(self,
               opt,
               replicas_to_aggregate,
               interval_steps,
               is_chief=False,
               total_num_replicas=None,
               device_setter=None,
               use_locking=False,
               name="fedAverage"):
    """Construct a fedAverage optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      interval_steps: number of steps between two "average op", which specifies
        how frequent a model synchronization is performed.
      is_chief: whether this worker is chief or not.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple blocks per update to variables.
      device_setter: A replica_device_setter that will be used to place copies
        of the trainable variables in the parameter server.
      use_locking: If True use locks for update operations.
      name: string. Name of the global variables and related operation on ps.
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    super(FederatedAveragingOptimizer, self).__init__(use_locking, name)
    logging.info(
        "FedAvgV4: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)
    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._interval_steps = interval_steps
    self._is_chief = is_chief
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate) - 1
    self._device_setter = device_setter
    self._name = name

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def _generate_shared_variables(self):
    """Generate a global variable placed on ps for each trainable variable.

       This creates a new copy of each user-defined trainable variable and places
       them on ps_device. These variables store the averaged parameters.
    """
    # Only the chief should initialize the variables
    if self._is_chief:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES, "global_model"]
    else:
      collections = ["global_model"]

    # Generate new global variables dependent on trainable variables.
    with ops.device(self._device_setter):
      for v in variables.trainable_variables():
        _ = variable_scope.variable(
            name="%s/%s" % (self._name, v.op.name),
            initial_value=v.initialized_value(), trainable=False,
            collections=collections)

      # Place the global step in the ps so that all the workers can see it
      self._global_step = variables.Variable(0, name="%s_global_step" %
          self._name, trainable=False)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.
    This contains most of the synchronization implementation.

    Args:
      grads_and_vars: List of (local_vars, gradients) pairs.
      global_step: Variable to increment by one after the variables have been
      updated. We need it to check staleness.
      name: Optional name for the returned operation. Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and apply averages to local vars or an op to update vars locally.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      raise ValueError("Global step is required")

    # Generate copy of all trainable variables
    self._generate_shared_variables()

    # Wraps the apply_gradients op of the parent optimizer
    apply_updates = self._opt.apply_gradients(grads_and_vars, global_step)

    # This function will be called whenever the global_step divides interval steps
    def _apply_averages():  # pylint: disable=missing-docstring
      # Collect local and global vars
      local_vars = [v for g, v in grads_and_vars if g is not None]
      global_vars = ops.get_collection_ref("global_model")
      # sync queue, place it in the ps
      with ops.colocate_with(self._global_step):
        sync_queue = data_flow_ops.FIFOQueue(
            -1, [dtypes.bool], shapes=[[]], shared_name="sync_queue")
      train_ops = []
      aggregated_vars = []
      with ops.name_scope(None, self._name + "/global"):
        for var, gvar in zip(local_vars, global_vars):
          # pylint: disable=protected-access
          # Get reference to the tensor, this works with Variable and ResourceVariable
          var = ops.convert_to_tensor(var)
          # Place the accumulator in the same ps as the corresponding global_var
          with ops.device(gvar.device):
            var_accum = data_flow_ops.ConditionalAccumulator(
                var.dtype,
                shape=var.get_shape(),
                shared_name=gvar.name + "/var_accum")
            # Add op to push local_var to accumulator
            train_ops.append(
                var_accum.apply_grad(var, local_step=global_step))
            # Op to average the vars in the accumulator
            aggregated_vars.append(var_accum.take_grad(self._replicas_to_aggregate))
            # Remember accumulator and corresponding device
            self._accumulator_list.append((var_accum, gvar.device))
      # chief worker updates global vars and enqueues tokens to the sync queue
      if self._is_chief:
        update_ops = []
        # Make sure train_ops are run
        with ops.control_dependencies(train_ops):
          # Update global_vars with average values
          for avg_var, gvar in zip(aggregated_vars, global_vars):
            with ops.device(gvar.device):
              update_ops.append(state_ops.assign(gvar, avg_var))
          # Update shared global_step
          with ops.device(global_step.device):
            update_ops.append(state_ops.assign_add(self._global_step, 1))
        # After averaging, push tokens to the queue
        with ops.control_dependencies(update_ops), ops.device(
            global_step.device):
          tokens = array_ops.fill([self._tokens_per_step],
                                  constant_op.constant(False))
          sync_op = sync_queue.enqueue_many(tokens)
      # non chief workers deque a token, they will block here until chief is done
      else:
        # Make sure train_ops are run
        with ops.control_dependencies(train_ops), ops.device(
            global_step.device):
          sync_op = sync_queue.dequeue()

      # All workers pull averaged values
      with ops.control_dependencies([sync_op]):
        local_update_op = self._assign_vars(local_vars, global_vars)
      return local_update_op

    # Check if we should push and average or not
    with ops.control_dependencies([apply_updates]):
      condition = math_ops.equal(
          math_ops.mod(global_step, self._interval_steps), 0)
      conditional_update = control_flow_ops.cond(
          condition, _apply_averages, control_flow_ops.no_op)

    chief_init_ops = []
    # Initialize accumulators, ops placed in ps
    for accum, dev in self._accumulator_list:
      with ops.device(dev):
        chief_init_ops.append(
            accum.set_global_step(global_step, name="SetGlobalStep"))
    self._chief_init_op = control_flow_ops.group(*(chief_init_ops))

    return conditional_update

  def _assign_vars(self, local_vars, global_vars):
    """Utility to refresh local variables.

    Args:
      local_vars: List of local variables.
      global_vars: List of global variables.

    Returns:
      refresh_ops: The ops to assign value of global vars to local vars.
    """
    reassign_ops = []
    for local_var, global_var in zip(local_vars, global_vars):
      reassign_ops.append(state_ops.assign(local_var, global_var))
    refresh_ops = control_flow_ops.group(*(reassign_ops))
    return refresh_ops

  def make_session_run_hook(self):
    """Creates a hook to handle federated average init operations."""
    return _FederatedAverageHook(self)

class _FederatedAverageHook(session_run_hook.SessionRunHook):
  """A SessionRunHook that handles ops related to FederatedAveragingOptimizer."""

  def __init__(self, fed_avg_optimizer):
    """Creates hook to handle FederatedAveragingOptimizer

    Args:
      fed_avg_optimizer: 'FederatedAveragingOptimizer' which this hook will
        initialize.
    """
    self._fed_avg_optimizer = fed_avg_optimizer

  def begin(self):
    local_vars = variables.trainable_variables()
    global_vars = ops.get_collection_ref("global_model")
    self._variable_init_op = self._fed_avg_optimizer._assign_vars(
        local_vars,
        global_vars)

  def after_create_session(self, session, coord):
    # Make sure all models start at the same point
    session.run(self._variable_init_op)
