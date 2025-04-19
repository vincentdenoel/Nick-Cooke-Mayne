import numpy as np

def sample_blocks(x, block_size, random=False, circular=True, step=None, func=lambda x: x):
    """
    Vectorized block sampling along the last axis of an n-D array.
    
    Arguments:
    - x: np.ndarray of shape (T, ...)
    - block_size: int, size of each block

    Keyword arguments:
    - random: bool, whether to sample blocks randomly. If False, functions as sliding window.
    - circular: bool, whether to apply circular padding
    - step: int, step size for sliding window. If None, defaults to block_size. If random, same number of blocks are sampled at random as a sliding window with given step.
    - func: function to apply to each block (default is identity function)
    
    Returns:
    - Blocks of shape (..., num_blocks, block_size), or result of applying func

    Example usage:
    - Sliding Block: random=False, step=1
    - Sliding Block Maxima: random=False, step=1, func=lambda x: np.max(x, axis=1)
    - Disjoint Block: random=False, step=None
    - Random Blocks (N blocks): random=True, step=len(x)/N
    """
    x = np.asarray(x)
    T = x.shape[-1]
    
    if T % block_size != 0:
        raise ValueError("Length of data must be divisible by block_size.")
    
    if step is None:
        step = block_size
    step = int(step)

    if circular:
        pad = block_size - 1
        x = np.concatenate([x, x[...,:pad]], axis=-1)
        T += pad

    if random:
        num_blocks = T // step
        start_idxs = np.random.randint(0, T - block_size + 1, size=(num_blocks,))
    else:
        start_idxs = np.arange(0, T - block_size + 1, step)

    # Gather blocks along the last axis using advanced indexing
    blocks = []
    for start_idx in start_idxs:
        blocks.append(func(x[..., start_idx:start_idx + block_size]))
    return np.squeeze(np.array(blocks))

def CBM(x, block_size, k=2, superblock_random=False, circular=True, step=1):
    """
    Compute circular block maxima sample with superblocks.
    
    Parameters:
    - x: 1D iterable of time series data
    - block_size: size of each sliding block (r)
    - k: multiplier to define superblock size (kr = k * r)
    - superblock_random: whether to sample superblocks randomly
    - circular: whether to apply circular padding within superblocks
    - step: step size for sliding window
    
    Returns:
    - List of maxima from circular blocks
    """
    kr = k * block_size
    blocks = sample_blocks(x, kr, random=superblock_random, circular=False, step=None)
    maxima = sample_blocks(blocks, block_size, circular=True, step=step, func=lambda x: np.max(x, axis=-1))

    return maxima