
import os
from typing import Optional, Literal, Any, assert_never
from .model_file_manager import ModelFileManager
from pathlib import Path
import datetime
import json
import logging
logger = logging.getLogger(__name__)

STUDIES_PATH = os.getenv("STUDIES_PATH", "storage/studies")

class StudyFileManager:
    RESULTS_FILE = "results.json"
    def __init__(
            self, 
            study_name : str, 
            studies_path : Optional[str] = None,
            conflict_strategy : Literal['new', 'error'] = 'new'
        ) -> None:
        self.studies_path = studies_path or STUDIES_PATH
        self.study_name = study_name
        self.path = Path(self.studies_path).joinpath(self.study_name)
        match conflict_strategy:
            case 'new':
                if self.path.exists():
                    self.study_name = self.study_name + datetime.datetime.now().isoformat()
                    logger.debug(f"Study path already exists at {self.path}, changing identifier to {self.study_name}.")
                    self.path = Path(self.studies_path).joinpath(self.study_name)
                    if self.path.exists():
                        raise FileExistsError(f"Failed to automatically solve conflict: New study path also exists at {self.path}")
            case 'error':
                if self.path.exists():
                    raise FileExistsError(f"Study path already exists at {self.path}")
            case never:
                assert_never(never)
        self.path.mkdir(parents=True, exist_ok=False)

    def new_experiment(self, experiment_name : str) -> ModelFileManager:
        return ModelFileManager(experiment_name, models_path=self.path)

    def save_results(self, results : dict[str, Any], best : str):
        complete_results = {
            "best_experiment" : {
                "name" : best,
                "results" : results[best]
            },
            "experiments" : results
        }
        with open(os.path.join(self.path, self.RESULTS_FILE), 'w') as f:
            json.dump(complete_results, f, indent=4)