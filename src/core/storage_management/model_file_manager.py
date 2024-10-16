
from pathlib import Path
from typing import Literal, NamedTuple, Any, Optional, Self
import datetime
import logging
import torch
import os
from .. import ModelDetails
import threading

logger = logging.getLogger(__name__)

MODELS_PATH = os.getenv("MODEL_PATH", "storage/models")
__lock = threading.Lock()
__context_instance : Optional['ModelFileManager'] = None


class ModelFileManager:
    MODEL_FILE_NAME = "model.pth"
    CHECKPOINTS_DIR = "checkpoints"
    METRICS_FORMAT = "metrics_{identifier}.csv"
    METRICS_DIR = ""
    CHECKPOINT_FORMAT = "epoch_{epoch:0>9}.pth"

    def __init__(self,
            model_name : str,
            model_identifier : str = "",
            conflict_strategy : Literal['new', 'error', 'load'] = 'new',
            models_path : Optional[str | os.PathLike] = None
        ) -> None:
        self.models_path = models_path or MODELS_PATH
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.__metrics_streams = {}
        self.__format_paths()
        if self.path.exists():
            match conflict_strategy:
                case 'load':
                    logger.debug(f"Loading existing model path at {self.path}")
                    self.__create_paths(exists_ok=True)
                case 'new':
                    self.model_identifier = datetime.datetime.now().isoformat()
                    logger.debug(f"Model path already exists at {self.path}, changing identifier to {self.model_identifier}.")
                    self.__format_paths()
                    if self.path.exists():
                        raise FileExistsError(f"Failed to automatically solve conflict: New model path also exists at {self.path}")
                    self.__create_paths(exists_ok=False)
                case 'error':
                    raise FileExistsError(f"Model path already exists at {self.path}")
                case _:
                    raise ValueError(f"Invalid conflict strategy: {conflict_strategy}")
        else:
            self.__create_paths(exists_ok=False)
    
    def __format_paths(self):
        self.path = Path(self.models_path).joinpath(self.model_name + '_' + self.model_identifier)
        self.checkpoint_path = self.path.joinpath(self.CHECKPOINTS_DIR)
        self.model_file = self.path.joinpath(self.MODEL_FILE_NAME)
        self.metrics_dest = self.path.joinpath(self.METRICS_DIR)

    def __create_paths(self, exists_ok = False):
        logger.debug(f"Creating paths for model {self.model_name} at {self.path}")
        self.path.mkdir(parents=True, exist_ok=exists_ok)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.metrics_dest.mkdir(parents=True, exist_ok=True)

    def init_metrics_file(self, metrics : list[str], identifier : str):
        self.__assert_context()
        if identifier in self.__metrics_streams:
            return self.__metrics_streams[identifier]
        logger.debug(f"Initializing metrics file for {metrics}")
        metrics_file = self.metrics_dest.joinpath(self.METRICS_FORMAT.format(identifier=identifier))
        file_stream : Any
        def init_file():
            logger.info(f"Creating new metrics file at {metrics_file}")
            file_stream = metrics_file.open('w')
            file_stream.write(",".join(metrics) + "\n")
            file_stream.flush()
            return file_stream
        if metrics_file.exists():
            logger.debug(f"Metrics file already exists at {metrics_file}")
            conflict = False
            with metrics_file.open('r') as f:
                header = f.readline().strip().split(',')
                header = [x.strip() for x in header]
                if header != metrics:
                    conflict = True
                    backup_name = self.METRICS_FORMAT.format(identifier=identifier + datetime.datetime.now().isoformat())
                    backup_path = self.metrics_dest.joinpath('old').joinpath(backup_name)
                    logger.error(
                        f"Existing metrics file has different header: Expected {metrics}, got {header}\n"
                        f"Backing it up to {backup_path} and creating a new one."
                    )
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    with backup_path.open('w') as backup:
                        backup.write(",".join(header) + "\n")
                        backup.write(f.read())
            if conflict:
                file_stream = init_file()
            else:
                file_stream = metrics_file.open('a')
        else:
            file_stream = init_file()
        assert file_stream is not None
        self.__metrics_streams[identifier] = file_stream
        return file_stream

    def write_metrics(self, metrics : list[str], identifier : str, records : list[list[Any]]):
        self.__assert_context()
        metrics_file = self.init_metrics_file(metrics, identifier)
        metrics_file.write("\n".join(",".join(map(str, record)) for record in records) + "\n")
        metrics_file.flush()

    def save_model_details(self, details : ModelDetails):
        self.__assert_context()
        torch.save(details, self.model_file)
    
    def load_model_details(self) -> ModelDetails:
        self.__assert_context()
        return torch.load(self.model_file, weights_only=False)
    
    def save_checkpoint(self, epoch, state_dict):
        self.__assert_context()
        path = self.checkpoint_path.joinpath(self.CHECKPOINT_FORMAT.format(epoch=epoch))
        if path.exists():
            logger.warning(f"Overwriting existing checkpoint at {path}")
        logger.info(f"Saving checkpoint at {path}")
        torch.save(state_dict, path)

    def load_last_checkpoint(self):
        self.__assert_context()
        checkpoint_files = self.checkpoint_path.glob("*")
        checkpoint_files = [(int(file.with_suffix("").name.split('_')[-1]), file) for file in checkpoint_files]
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > 0:
            return torch.load(checkpoint_files[-1][1], weights_only=False)
        else:
            return None
    
    def __assert_context(self):
        global __context_instance
        if __context_instance is not self:
                    raise RuntimeError("ModelFileManager must be used as a context manager")

    def __enter__(self):
        global __lock, __context_instance
        if __lock.acquire(blocking=False):
            __context_instance = self
            return self
        elif __context_instance is self:
            return self
        else:
            raise RuntimeError("Conflicting context manager instances")
    
    def __exit__(self, exc_type, exc_value, traceback):
        global __lock, __context_instance
        for stream in self.__metrics_streams.values():
            stream.close()
        self.__metrics_streams = {}
        __context_instance = None
        __lock.release()

    @classmethod
    def get_context_instance(cls) -> 'ModelFileManager':
        if __context_instance is None:
            raise RuntimeError("No ModelFileManager context available")
        return __context_instance