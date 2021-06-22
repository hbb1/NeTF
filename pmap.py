from functools import partial
import jax
import jax.numpy as jnp
import pdb
from jax import vmap

def square(x):
    return 5

# vmap(square)(np.random.randint(2, 5))
pdb.set_trace()

# out = jax.pmap(lambda x: x **2)(jnp.arange(4))
# print(out)
# pdb.set_trace()
# f = lambda x: x + jax.lax.psum(x, axis_name='i')
# data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
# out = jax.pmap(f, axis_name='i')(data)
# x, y = jnp.arange(2.), 4.
# pdb.set_trace()
# out = jax.pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  
# print(out)
