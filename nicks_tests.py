import pytest
import numpy as np
import nicks_functions as nf

@pytest.fixture
def setup_data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8])

def test_sliding_blocks_basic(setup_data):
    result = nf.sample_blocks(setup_data, block_size=2)
    expected = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8])]
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))

def test_sliding_blocks_circular(setup_data):
    result = nf.sample_blocks(setup_data, block_size=2, circular=True, step=1)
    assert len(result) == len(setup_data)
    assert np.array_equal(result[-1], np.array([8, 1]))

def test_sliding_blocks_with_step():
    data = np.array([1, 2, 3, 4, 5, 6])
    result = nf.sample_blocks(data, block_size=2, step=1, circular=False)
    assert len(result) == 5
    assert np.array_equal(result[-1], np.array([5, 6]))

def test_sliding_blocks_error():
    with pytest.raises(ValueError):
        nf.sample_blocks([1, 2, 3], block_size=2)

def test_random_blocks_basic(setup_data):
    np.random.seed(0)  # For reproducibility
    result = nf.sample_blocks(setup_data, random=True, block_size=2)
    assert len(result) == 4
    assert all(len(block) == 2 for block in result)

def test_random_blocks_sample_size(setup_data):
    np.random.seed(0)  # For reproducibility
    result = nf.sample_blocks(setup_data, random=True, block_size=2, step=4)
    assert len(result) == 2

def test_random_blocks_error():
    with pytest.raises(ValueError):
        nf.sample_blocks([1, 2, 3], random=True, block_size=2)

def test_CBM_basic(setup_data):
    result = nf.CBM(setup_data, block_size=2, k=2)
    assert isinstance(result, list)
    assert all(isinstance(x, (int, float, np.number)) for x in result)

def test_CBM_different_k(setup_data):
    result1 = nf.CBM(setup_data, block_size=2, k=2)
    result2 = nf.CBM(setup_data, block_size=2, k=4)
    assert result1[3] != result2[3]
    assert len(result1) == len(setup_data)
    assert len(result2) == len(setup_data)

def test_CBM_different_block_size(setup_data):
    result1 = nf.CBM(setup_data, block_size=1, k=1)
    result2 = nf.CBM(setup_data, block_size=8, k=1)
    for i in range(len(result1)):
        assert result1[i] == setup_data[i]
    for i in range(1, len(result2)):
        assert result2[i-1] == result2[i]
    assert len(result1) == len(setup_data)
    assert len(result2) == len(setup_data)

def test_CBM_step_size(setup_data):
    result = nf.CBM(setup_data, block_size=2, step=1)
    assert len(result) > len(nf.CBM(setup_data, block_size=2, step=2))

def test_CBM_circular(setup_data):
    result_circular = nf.CBM(setup_data, block_size=2, circular=True)
    result_noncircular = nf.CBM(setup_data, block_size=2, circular=False)
    assert len(result_circular) >= len(result_noncircular)