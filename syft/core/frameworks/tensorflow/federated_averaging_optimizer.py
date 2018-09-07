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

import time

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
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
  * `local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the variables.

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
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next round.

  For the replicas:
  <empty line>
  1. Start a training block: fetch variables and train for "interval_steps" steps.
  2. Once the training block has been computed, push local variables into variable
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the variables, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
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
                                                      interval_steps=100,
                                                      device_setter)

  # Some models have startup_delays to help stabilize the model but when using
  # federated_average training, set it to 0.

  # Now you can call 'minimize() normally'
  # train_op = opt.minimize(loss, global_step=global_step)

  # And also, create the hook which handles initialization, queues and averaging.
  fed_avg_hook = opt.make_session_run_hook(is_chief)
  ```

  In the training program, every worker will run the train_op as if not
  averaged or synchronized. Note that if you want to run other ops like
  test op, you should use common session instead of MonitoredSession:

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target, is_chief=is_chief,
      hooks=[ma_replicas_hook, ma_hook]) as mon_sess:
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
      total_num_replicas: Total number of tasks/workers/replicas, could be
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple blocks per update to variables.
      device_setter: A replica_device_setter that will be used to place copies
        of the trainable variables in the parameter server.
      name: string. Name of the global variables and related operation on ps.
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    super(FederatedAveragingOptimizer, self).__init__(use_locking, name)
    logging.info(
        "FedAvgV3: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)
    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._interval_steps = interval_steps
    self._average_applied = False
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
    self._global_step = None
    self._sync_token_queue = None
    self._device_setter = device_setter
    self._name = name

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def _generate_shared_variables(self):
    """Generate a global variable placed on ps for each trainable variable.

       This creates a new copy of each user-defined trainable variable and places
       them on ps_device. These variables store the averaged parameters.
    """
    # Change all variables to local variables.
    for v in variables.global_variables():
      ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)

    # Clear global_variables list.
    ops.get_default_graph().clear_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    # Generate new global variables dependent on trainable variables.
    with ops.device(self._device_setter):
      for v in variables.trainable_variables():
        _ = variable_scope.get_variable(
            name="%s/%s" % (self._name, v.op.name),
            initializer=v.initialized_value(), trainable=False,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES, "global_model"])

      self._global_step = variables.Variable(0, name="%s_global_step" %
          self._name, trainable=False)

  def minimize(self, loss, global_step=None):
    """Add operations to minimize `loss` by updating `var_list`.
    This simply wraps the minimize() from the real optimizer.
    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.
    """
    self._curr_iter = global_step
    return self._opt.minimize(loss, global_step)

  def _apply_model_average(self, lvars_and_gvars, name=None):
    """Apply local weights to global variables.

    This contains most of the synchronization implementation.

    Args:
      lvars_and_gvars: List of (local_vars, global_vars) pairs.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the lvars_and_gvars is empty.
    """
    if not lvars_and_gvars:
      raise ValueError("Must supply at least one variable")

    train_ops = []
    aggregated_lvars = []

    model_reassign_ops = []

    global_vars = [g for v, g in lvars_and_gvars if v is not None]

    # local_anchor op will be placed on this worker task by default.
    local_anchor = control_flow_ops.no_op()
    # Colocating local_step variable prevents it being placed on the PS.
    with ops.colocate_with(local_anchor):
      self._local_step = variables.Variable(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=self._global_step.dtype.base_dtype,
          name="%s_local_step" % self._name)

    self.local_step_init_op = state_ops.assign(self._local_step,
        self._global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    with ops.name_scope(None, self._name):
      for lvar, gvar in lvars_and_gvars:
        lvar = ops.convert_to_tensor(lvar)
        with ops.device(gvar.device):
          # Dense variables.
          if lvar is None:
            aggregated_lvars.append(None)  # pass-through.
            continue
          elif isinstance(lvar, ops.Tensor):
            lvar_accum = data_flow_ops.ConditionalAccumulator(
                lvar.dtype,
                shape=gvar.get_shape(),
                shared_name=gvar.name + "/lvar_accum")
            train_ops.append(lvar_accum.apply_grad(
                lvar, local_step=self._local_step))
            aggregated_lvars.append(lvar_accum.take_grad(
                self._replicas_to_aggregate))
          else:
            if not isinstance(lvar, ops.IndexedSlices):
              raise ValueError("Unknown model variable type!")
            lvar_accum = data_flow_ops.SparseConditionalAccumulator(
                lvar.dtype,
                shape=(),
                shared_name=gvar.name + "/model_variable_accum")
            train_ops.append(lvar_accum.apply_indexed_slices_grad(
                lvar, local_step=self._local_step))
            aggregated_lvars.append(lvar_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((lvar_accum, gvar.device))

      # sync_op will be assigned to the same device as the global step.
      with ops.device(self._global_step.device), ops.name_scope(""):
        for avg_var, gvar in zip(aggregated_lvars, global_vars):
          model_reassign_ops.append(state_ops.assign(gvar, avg_var))
        model_reassign_ops.append(state_ops.assign_add(self._global_step, 1))
        update_op = control_flow_ops.group(*(model_reassign_ops))

      # Create token queue.
      with ops.device(self._global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    self._global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(self._global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
        train_op = state_ops.assign(self._local_step, token)

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._tokens_per_step], self._global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  self._global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._average_applied = True
      return train_op

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

  def get_chief_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate weights,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_model_average() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_model_average().
    """
    if self._average_applied is False:
      raise ValueError("Should be called after apply_model_average().")

    return self._chief_queue_runner

  def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_model_average().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._average_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_model_average().")

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    """Creates a hook to handle federated average and init operations."""
    return _FederatedAverageHook(self, is_chief, num_tokens)

class _FederatedAverageHook(session_run_hook.SessionRunHook):
  """A SessionRunHook that handles ops related to FederatedAveragingOptimizer."""

  def __init__(self, fed_avg_optimizer, is_chief, num_tokens):
    """Creates hook to handle FederatedAveragingOptimizer output_shapes

    Args:
      fed_avg_optimizer:  'FederatedAveragingOptimizer' which this hook will
        initialize.
      is_chief: 'Bool', whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    """
    self._fed_avg_optimizer = fed_avg_optimizer
    self._is_chief = is_chief
    self._num_tokens = num_tokens

  def begin(self):
    self._fed_avg_optimizer._generate_shared_variables()
    local_vars = variables.trainable_variables()
    global_vars = ops.get_collection_ref("global_model")
    self._refresh_local_vars_op = self._fed_avg_optimizer._assign_vars(
        local_vars,
        global_vars)
    local_and_init_vars = list(zip(local_vars, global_vars))

    self._apply_ma_op = self._fed_avg_optimizer._apply_model_average(
        local_and_init_vars,
        global_vars)

    if self._is_chief:
      self._local_init_op = self._fed_avg_optimizer.chief_init_op
      self._ready_for_local_init_op = (
          self._fed_avg_optimizer.ready_for_local_init_op)
      self._q_runner = self._fed_avg_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._fed_avg_optimizer.get_init_tokens_op(
          self._num_tokens)
    else:
      self._local_init_op = self._fed_avg_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
          self._fed_avg_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    """Runs FederatedAveragingOptimizer initialization ops."""
    local_init_success, msg = session_manager._ready( # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for FederatedAveragingOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for "
          "FederatedAveragingOptimizer local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)
    session.run(self._refresh_local_vars_op)

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs([
        self._fed_avg_optimizer._global_step,
        self._fed_avg_optimizer._curr_iter])

  def after_run(self, run_context, run_values):
    """ FedAvg Distributed Training """
    global_step, curr_iter = run_values.results
    session = run_context.session
    if curr_iter % self._fed_avg_optimizer._interval_steps == 0 \
        and not curr_iter == 0:
      # Apply model_average op before pulling.
      cur_time = time.time()
      session.run(self._apply_ma_op)

      elapsed_ma_time = time.time() - cur_time
      logging.info("FedAvg %s: global_step: %d, _step:%d,"
                   "average time: %.4f sec.", type(self).__name__,
                   global_step, curr_iter, elapsed_ma_time)

      # Pull new model params after federated average op.
      session.run(self._refresh_local_vars_op)
