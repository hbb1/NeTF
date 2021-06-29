A super brief README

### Install

#### Dependencies
install JAX : (following the instructions) https://github.com/google/jax

install flax (NN framework based on JAX)

```bash
pip install flax 
```

Install tensorflow for visualizing the training process (tesorboard)

```bash
pip install tensorflow
```

### Train
Assume we use 2 gpus for training.

```bash
CUDA_VISIVLE_DEVICES=0,1 train.py --config zaragoza_bunny_all_grid
```


### Test
```bash
CUDA_VISIBLE_DEVICES=0, reander.py --config zaragonza_bunny_all_grid
```
