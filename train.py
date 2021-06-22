# coding=utf-8
# Copyright 2021 ShanghaiTech University Authors.
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
import pdb
import functools
import gc
import time 
from absl import app
from absl import flags
from flax.core.scope import Variable
import numpy as np
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jax import random
from netf.datasets import get_dataset
from netf import utils
from netf import models

FLAGS = flags.FLAGS

utils.define_flags()

def train_step(model, rng, state, batch, lr):
  """One optimization step.
  
  Args:
    model: The NeTF model
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batc of data for training.
    lr: float, real-time learning rate.
  
  Returns:
    new_state: utils.TrainState, new training state.
    stats: list. [(loss)]
    rng: jnp.ndarray, updated random number generator.
  """
  rng, key_0, key_1 = random.split(rng, 3)
  def loss_fn(variables):
    origins = batch['grid']
    radius = batch['radius']
    pred_hist = model.apply(variables, key_0, key_1, origins, radius, randomized=False)
    if len(pred_hist) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    scale = 1e3
    loss = ((pred_hist[0].flatten() - batch['hist']*scale)**2).mean()
    return loss, utils.Stat(loss=loss)

  
  (_, stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target)
  grad = jax.lax.pmean(grad, axis_name='batch')
  stats = jax.lax.pmean(stats, axis_name='batch')

  # Clip the gradient by value.
  if FLAGS.grad_max_val > 0:
    clip_fn = lambda z: jnp.clip(z, -FLAGS.grad_max_val, FLAGS.grad_max_val)
    grad = jax.tree_util.tree_map(clip_fn, grad)

  # Clip the (possibly value-clipped) gradient by norm.
  if FLAGS.grad_max_norm > 0:
    grad_norm = jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grad, initializer=0))
    mult = jnp.minimum(1, FLAGS.grad_max_norm / (1e-7 + grad_norm))
    grad = jax.tree_util.tree_map(lambda z: mult * z, grad)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng

def main(unused_argv):
  rng = random.PRNGKey(20210601)
  np.random.seed(20210601+jax.process_index())
  if FLAGS.config is not None:
      utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
      raise ValueError("Batch siz must be devisible by the number of devices")
  
  dataset = get_dataset(FLAGS)
  rng, key = random.split(rng)
  model, variables = models.construct_netf(key, dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, variables

  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      lr_init=FLAGS.lr_init, 
      lr_final=FLAGS.lr_final, 
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps,
      lr_delay_mult=FLAGS.lr_delay_mult
  )

  if not utils.isdir(FLAGS.cache_dir):
    utils.makedirs(FLAGS.cache_dir)
  
  state = checkpoints.restore_checkpoint(FLAGS.cache_dir, state)
  init_step = state.optimizer.state.step + 1
  
  # train_step(model, rng, state, next(dataset), learning_rate_fn(0))
  # pdb.set_trace()
  state = flax.jax_utils.replicate(state)
  # pdb.set_trace()
  train_pstep = jax.pmap(
                functools.partial(train_step, model), 
                axis_name="batch", 
                in_axes=(0, 0, 0, None), 
                donate_argnums=(2,))
  
  # train = functools.partial(train_step, model)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.cache_dir)
  
  # Disable automatic garbage collection for efficiency
  gc.disable()
  stats_trace = []
  n_local_devices = jax.local_device_count()
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)
  
  reset_timer = True
  # if n_local_devices > 1:
    # raise ValueError("Currently not support gpu>1, please set CUDA_VISIBLE_DEVICES=0.")
  
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = learning_rate_fn(step)
    state, stats, keys = train_pstep(keys, state, batch, lr)
    if jax.process_index() == 0:
      stats_trace.append(stats)
    if step % FLAGS.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.process_index() == 0:
      if step % FLAGS.print_every == 0:
        summary_writer.scalar("train_loss", stats.loss[0], step)
        avg_loss = np.mean(np.stack([s.loss for s in stats_trace]))
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        stats_trace = []
        summary_writer.scalar("learning_rate", lr, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        print("iter[{} / {}] time: {:4f} \t loss: {:6f} ({:6f})\t  lr: {:4f}\t"
              .format(step, FLAGS.max_steps+1, steps_per_sec, stats.loss[0], avg_loss, lr))

        if step % FLAGS.save_every == 0:
          state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
          checkpoints.save_checkpoint(
              FLAGS.cache_dir, state_to_save, int(step), keep=100)


if __name__ == '__main__':
  app.run(main)