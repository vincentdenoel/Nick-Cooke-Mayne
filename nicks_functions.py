import numpy as np

def sample_blocks(x, block_size, random = False, circular=True, step=None, func=lambda x: x):
    """
    Split data into disjoint blocks of a given size.
    
    Arguments:
    - x: 1D array
    - block_size: int, size of each block

    Keyword arguments:
    - random: bool, whether to sample blocks randomly. If False, functions as sliding window.
    - circular: bool, whether to apply circular padding
    - step: int, step size for sliding window. If None, defaults to block_size. If random, same number of blocks are sampled at random as a sliding window with given step.
    - func: function to apply to each block (default is identity function)
    
    Returns:
    - List of 1D numpy arrays, each of size block_size. If func is not identity, the list is instead the result of applying func to each block.

    Example usage:
    - Sliding Block: random=False, step=1
    - Sliding Block Maxima: random=False, step=1, func=lambda x: np.max(x, axis=1)
    - Disjoint Block: random=False, step=None
    - Random Blocks (N blocks): random=True, step=len(x)/N
    """
    x = np.asarray(x)
    n = len(x)
    if step is None:
        step = block_size
    step = int(step)
    block_size = int(block_size)
    if len(x) % block_size != 0:
        raise ValueError("Length of data must be divisible by block_size.")
    if circular:
        x = np.concatenate([x, x[:block_size-1]])
    else:
        n += 1 - block_size
    if random:
        sample_size = n // step
        indices = np.random.randint(0, n, sample_size)[:, None]
    else:
        indices = np.arange(0, n, step)[:, None]
    indices = indices + np.arange(block_size)
    blocks = x[indices]
    return func(blocks)

def CBM(x, block_size, k=2, superblock_random = False, circular=True, step=1):
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
    maxima = []

    for block in blocks:
        maxima.extend(sample_blocks(block, block_size, circular=True, step=step, func=np.max))


    return np.array(maxima)