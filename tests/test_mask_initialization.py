import numpy as np

from src.constrjson import JSONconstr


def test_base_mask_allows_all_tokens_by_default() -> None:
    mask = JSONconstr.create_base_mask(4)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (4,)
    assert mask.tolist() == [True, True, True, True]
