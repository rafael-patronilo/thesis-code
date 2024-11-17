
from collections import OrderedDict
from pathlib import Path
from typing import Literal, NamedTuple, Any, Optional, Self, assert_never, TYPE_CHECKING
import datetime
import logging
import torch
import os
if TYPE_CHECKING:
    from ..trainer import TrainerConfig
    from core.util.typing import PathLike
import threading
import json
import importlib

logger = logging.getLogger(__name__)

MODELS_PATH = os.getenv("MODEL_PATH", "storage/models")
METRICS_BUFFER_LIMIT = 10

class _MetricsBufferer:
        def __init__(self, stream):
            self.stream = stream
            self.buffer : list[OrderedDict[str, Any]] = []
        
        def flush(self):
            self.stream.write("\n".join(",".join(map(str, record.values())) for record in self.buffer) + "\n")
            self.stream.flush()
            self.buffer.clear()

        def add(self, record : OrderedDict[str, Any]):
            self.buffer.append(record)
            if len(self.buffer) >= METRICS_BUFFER_LIMIT:
                self.flush()

class ModelFileManager:
    MODEL_FILE_NAME = "model.pth"
    CONFIG_FILE_NAME = "config.json"
    CHECKPOINTS_DIR = "checkpoints"
    RESULTS_DIR = "results"
    LAST_CHECKPOINT = "last_checkpoint.pth"
    METRICS_FORMAT = "metrics_{identifier}.csv"
    METRICS_DIR = ""
    CHECKPOINT_FORMAT = "epoch_{epoch:0>9}.pth"
    

    def __init__(self,
            path : 'PathLike',
            models_path : Optional['PathLike'] = None
        ) -> None:
        self.models_path = Path(models_path or MODELS_PATH)
        self.__metrics_bufferers : dict[str, _MetricsBufferer]= {}
        self.path = self.models_path.joinpath(path)
        self.model_name = self.path.name
        self.__is_context = False
        self.__format_paths()
    
    def __format_paths(self):
        self.checkpoint_path = self.path.joinpath(self.CHECKPOINTS_DIR)
        self.model_file = self.path.joinpath(self.MODEL_FILE_NAME)
        self.config_file = self.path.joinpath(self.CONFIG_FILE_NAME)
        self.metrics_dest = self.path.joinpath(self.METRICS_DIR)
        self.results_dest = self.path.joinpath(self.RESULTS_DIR)
        self.last_checkpoint = self.path.joinpath(self.LAST_CHECKPOINT)

    def __create_paths(self, exists_ok = False):
        logger.debug(f"Creating paths for model {self.model_name} at {self.path}")
        self.path.mkdir(parents=True, exist_ok=exists_ok)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.metrics_dest.mkdir(parents=True, exist_ok=True)

    def init_directory(self):
        self.__create_paths(exists_ok=True)

    def init_metrics_file(self, identifier : str, metrics : list[str]):
        self.__assert_context()
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
        self.__metrics_bufferers[identifier] = _MetricsBufferer(file_stream)
        return file_stream

    def write_metrics(self, identifier : str, records : OrderedDict[str, Any]):
        self.__assert_context()
        try:
            bufferer = self.__metrics_bufferers[identifier]
        except KeyError:
            logger.error(f"Metrics file for {identifier} not initialized")
        bufferer.add(records)
        #TODO tensorboard logging?

    def flush(self):
        self.__assert_context()
        for bufferer in self.__metrics_bufferers.values():
            bufferer.flush()

    def save_config(self, config : 'TrainerConfig', conflict_strategy : Literal['error', 'ignore', 'compare'] = 'compare'):
        self.__assert_context()
        if self.config_file.exists():
            match conflict_strategy:
                case 'ignore':
                    logger.warning("Ignoring existing config file")
                    return
                case 'error':
                    raise FileExistsError(f"Config file already exists at {self.config_file}")
                case 'compare':
                    existing_config = self.load_config()
                    post_json_config = json.loads(json.dumps(config)) # Remove any non json objects (eg. tuples)
                    if existing_config != post_json_config:
                        raise FileExistsError(
                            f"Config file already exists and is different from provided config. "
                            f"Existing: {existing_config}, Provided: {config}")
                    else:
                        logger.warning("Config file already exists and is the same, not writing")
                        return
                case never:
                    assert_never(never)
        with self.config_file.open('w') as f:
            json.dump(config, f, indent=4)

    def load_config(self) -> 'TrainerConfig':
        self.__assert_context()
        if self.config_file.exists():
            with self.config_file.open('r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found at {self.config_file}")
    
    def save_checkpoint(self, epoch, state_dict, abrupt):
        self.__assert_context()
        path = self.checkpoint_path.joinpath(self.CHECKPOINT_FORMAT.format(epoch=epoch))
        if path.exists():
            logger.warning(f"Overwriting existing checkpoint at {path}")
        if not abrupt:
            logger.info(f"Overwriting last checkpoint at {self.last_checkpoint}")
            torch.save(state_dict, self.last_checkpoint)
        else:
            logger.warning("Abrupt checkpoint, not overwriting last checkpoint")
        logger.info(f"Saving checkpoint at {path}")
        torch.save(state_dict, path)
        

    def load_last_checkpoint(self):
        self.__assert_context()
        if self.last_checkpoint.exists():
            # TODO load weights only (currently not working)
            return torch.load(self.last_checkpoint, weights_only=False)
        else:
            logger.warning(f"No checkpoint found at {self.last_checkpoint}, searching latest checkpoint")
        checkpoint_files = self.checkpoint_path.glob("*")
        checkpoint_files = [(int(file.with_suffix("").name.split('_')[-1]), file) for file in checkpoint_files]
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > 0:
            return torch.load(checkpoint_files[-1][1], weights_only=False)
        else:
            return None
    
    def __assert_context(self):
        if not self.__is_context:
            raise RuntimeError("ModelFileManager must be used as a context manager")

    def __enter__(self) -> Self:
        self.__is_context = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for bufferer in self.__metrics_bufferers.values():
            bufferer.stream.close()
        self.__metrics_bufferers.clear()
        self.__is_context = False