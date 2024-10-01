import pandas as pd
import numpy as np
from pathlib import Path
import logging
import functools
import os
from typing import Callable, Concatenate
root_logger = logging.getLogger()

TENSOR_DEBUG_PATH = Path(os.getenv("TENSOR_DEBUG_PATH") or "debug")
PathLike = str | os.PathLike

def log_table(file_name : PathLike, level = logging.DEBUG, **tensors):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.exists(follow_symlinks=True) and level >= root_logger.level:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({k: v.numpy(force=True).flatten() for k, v in tensors.items()})
        df.to_csv(path)

def log_tensor(file_name : PathLike, tensor, level = logging.DEBUG):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.exists(follow_symlinks=True) and level >= root_logger.level:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, tensor.numpy(force=True))

def log_tensors(file_name : PathLike, *args, level = logging.DEBUG, **kwds):
    path = TENSOR_DEBUG_PATH.joinpath(file_name)
    if not path.suffix == ".npz":
        path = path.with_suffix(".npz")
    if not path.exists(follow_symlinks=True) and level >= root_logger.level:
        path.parent.mkdir(parents=True, exist_ok=True)
        args = [a.numpy(force=True) for a in args]
        kwds = {k: v.numpy(force=True) for k, v in kwds.items()}
        np.savez(path, *args, **kwds)

