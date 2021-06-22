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

def main(unused_argv):
  rng = random.PRNGKey(20210601)
  if FLAGS.config is not None:
      utils.update_flags(FLAGS)

  dataset = get_dataset(FLAGS)
  rng, key = random.split(rng)
  model, init_variables = models.construct_netf(key, dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables
  state = checkpoints.restore_checkpoint(FLAGS.cache_dir, state)
  cx, cy, cz = dataset.volume_position
  length = dataset.volume_size / 2.
  volume_box = (cx, cy, cz, length)
  resolutions = (256, 256, 256)
  utils.render_volume(model, 
                    state.optimizer.target, 
                    volume_box, 
                    resolutions, 
                    prefix='{}_{}'.format(FLAGS.config, state.optimizer.state.step))
 

if __name__ == '__main__':
  app.run(main)