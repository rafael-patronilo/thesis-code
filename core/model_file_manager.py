
from pathlib import Path
from typing import Literal, NamedTuple, Any, Optional
import datetime
import logging
import torch
import os

logger = logging.getLogger(__name__)

MODELS_PATH = os.getenv("MODEL_PATH") or "storage/models"

ModelDetails = NamedTuple(
    "ModelDetails",
    [
        ("architecture", Any),
        ("optimizer", str),
        ("loss_fn", str),
        ("dataset", str),
        ("metrics", list[str]),
        ("batch_size", int),
        ("train_metrics", Optional[list[str]]),
    ]
)

class ModelFileManager:
    MODEL_FILE_NAME = "model.pt"
    CHECKPOINTS_DIR = "checkpoints"
    METRICS_FORMAT = "metrics{identifier}.csv"
    METRICS_DIR = ""
    CHECKPOINT_FORMAT = "epoch_{epoch:0>9}.pt"
    
    def __init__(self,
            model_name : str,
            model_identifier : str = "",
            conflict_strategy : Literal['new', 'error', 'ignore'] = 'new',
        ) -> None:
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.__format_paths()
        if self.path.exists():
            match conflict_strategy:
                case 'ignore':
                    pass
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
    
    def __format_paths(self):
        self.path = Path(MODELS_PATH).joinpath(self.model_name + '_' + self.model_identifier)
        self.checkpoint_path = self.path.joinpath(self.CHECKPOINTS_DIR)
        self.model_file = self.path.joinpath(self.MODEL_FILE_NAME)
        self.metrics_dest = self.path.joinpath(self.METRICS_DIR)
        self.__metrics_stream = None

    def __create_paths(self, exists_ok = False):
        self.path.mkdir(parents=True, exist_ok=exists_ok)
        self.checkpoint_path.mkdir(exist_ok=exists_ok)

    def init_metrics_file(self, metrics : list[str], identifier : str = ""):
        logger.debug(f"Initializing metrics file for {metrics}")
        metrics_file = self.metrics_dest.joinpath(self.METRICS_FORMAT.format(identifier=identifier))
        file_stream : Any
        def init_file():
            logger.info(f"Creating new metrics file at {metrics_file}")
            file_stream = metrics_file.open('w+')
            file_stream.write(",".join(metrics) + "\n")
        if metrics_file.exists():
            logger.debug(f"Metrics file already exists at {metrics_file}")
            conflict = False
            with metrics_file.open('r') as f:
                header = f.readline().strip().split(',')
                if header != metrics:
                    conflict = True
                    backup_name = self.METRICS_FORMAT.format(identifier=identifier + datetime.datetime.now().isoformat())
                    backup_path = self.metrics_dest.joinpath('old').joinpath(backup_name)
                    logger.error(
                        f"Existing metrics file has different header: Expected {metrics}, got {header}\n"
                        f"Backing up to {backup_path}"
                    )
                    with backup_path.open('w+') as backup:
                        backup.write(",".join(header) + "\n")
                        backup.write(f.read())
            if conflict:
                init_file()
            else:
                file_stream = metrics_file.open('a')
        else:
            init_file()
        assert file_stream is not None
        return file_stream

    def save_model_details(self, details : ModelDetails):
        torch.save(details, self.model_file)
    
    def load_model_details(self) -> ModelDetails:
        return torch.load(self.model_file)
    
    def save_checkpoint(self, epoch, state_dict):
        path = self.checkpoint_path.joinpath(self.CHECKPOINT_FORMAT.format(epoch=epoch))
        if path.exists():
            logger.warn(f"Overwriting existing checkpoint at {path}")
        logger.info(f"Saving checkpoint at {path}")
        torch.save(state_dict, path)

    def load_last_checkpoint(self):
        checkpoint_files = self.checkpoint_path.glob("*")
        checkpoint_files = [(int(file.with_suffix("").name.split('_')[-1]), file) for file in checkpoint_files]
        checkpoint_files.sort(key=lambda x: x[0])
        if len(checkpoint_files) > 0:
            return torch.load(checkpoint_files[-1][1])
        else:
            raise FileNotFoundError(f"No checkpoint found for model {self.model_name} with identifier {self.model_identifier}")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__metrics_stream is not None:
            self.__metrics_stream.close()