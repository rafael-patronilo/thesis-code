import pandas as pd
import numpy as np
from pathlib import Path
import logging
import functools
import os
from typing import Callable, Concatenate
from . import assert_type
import torch
root_logger = logging.getLogger()

TENSOR_DEBUG_PATH = Path(os.getenv("TENSOR_DEBUG_PATH") or "debug")
PathLike = str | os.PathLike

def debug(func : Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        if root_logger.level > logging.DEBUG:
            return
        try:
            return func(*args, **kwds)
        except Exception as e:
            logging.getLogger(func.__module__).exception(f"Exception in debug function {func.__name__}: {e}")
    return wrapper

def _get_np(obj) -> np.ndarray:
    if not isinstance(obj, (torch.Tensor, np.ndarray)):
        obj = obj()
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy(force=True)
    assert_type(obj, np.ndarray)
    return obj

@debug
def debug_table(file_name : PathLike, **tensors):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.exists(follow_symlinks=True):
        def expand(key, tensor):
            tensor = _get_np(tensor)
            if len(tensor.shape) == 1:
                return [(key, tensor)]
            elif len(tensor.shape) == 2:
                return ((f"{key}_{i}", _get_np(tensor)[:, i]) for i in range(tensor.shape[1]))
            else:
                raise ValueError(f"Cannot expand tensor of shape {tensor.shape}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({xK: xV for k, v in tensors.items() for xK, xV in expand(k, v)})
        df.to_csv(path)

@debug
def debug_tensor(file_name : PathLike, tensor):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.exists(follow_symlinks=True):
        tensor = _get_np(tensor)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, tensor)

@debug
def debug_tensors(file_name : PathLike, *args, **kwds):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.suffix == ".npz":
        path = path.with_suffix(".npz")
    if not path.exists(follow_symlinks=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        args = [_get_np(a) for a in args]
        kwds = {k: _get_np(v) for k, v in kwds.items()}
        np.savez(path, *args, **kwds)

