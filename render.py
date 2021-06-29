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
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

utils.define_flags()

def main(unused_argv):
  rng = random.PRNGKey(20210601)
  if FLAGS.config is not None:
      utils.update_flags(FLAGS)

  FLAGS.__dict__['batch_size'] = 200
  FLAGS.__dict__['batching'] = 'single_grid'

  dataset = get_dataset(FLAGS)
  rng, key = random.split(rng)

  model, init_variables = models.construct_netf(key, dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables
  state = checkpoints.restore_checkpoint(FLAGS.cache_dir, state)
  pdb.set_trace()
  for i in range(20):
    batch = next(dataset)
    origins = batch['grid']
    radius = batch['radius']
    variables = state.optimizer.target
    rng = random.PRNGKey(20210601)
    rng, key_0, key_1 = random.split(rng, 3)
    pred_hist = model.apply(variables, key_0, key_1, origins, radius, randomized=False)
    pred_hist = np.array(pred_hist[0].flatten() / 1e3)
    plt.clf()
    plt.plot(np.arange(200), batch['hist'].flatten(), 'b', label='gt_hist')
    plt.plot(np.arange(200), pred_hist, 'r', label='pred_hist')
    plt.legend()
    plt.savefig('tmp/{}_pred_hist{}.png'.format(FLAGS.config, i))
  
  plt.close()
  pdb.set_trace()
  cx, cy, cz = dataset.volume_position
  length = dataset.volume_size * 2
  volume_box = (cx, cy, cz, length)
  resolutions = (256, 256, 256)
  utils.rendering(model, 
                    state.optimizer.target, 
                    volume_box, 
                    resolutions, 
                    prefix='tmp/{}_{}'.format(FLAGS.config, state.optimizer.state.step))
 

if __name__ == '__main__':
  app.run(main)