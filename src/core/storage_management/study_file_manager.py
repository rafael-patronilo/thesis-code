
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
            best_experiment_link_name : Optional[str] = 'best'
        ) -> None:
        self.studies_path = studies_path or STUDIES_PATH
        self.study_name = study_name
        self.path = Path(self.studies_path).joinpath(self.study_name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.best_experiment_link_path = None
        if best_experiment_link_name is not None:
            self.best_experiment_link_path = self.path.joinpath(best_experiment_link_name)

    def new_experiment(self, experiment_name : str) -> ModelFileManager:
        return ModelFileManager(experiment_name, models_path=self.path)

    def read_json(self, file_name : str) -> dict[str, Any]:
        path = self.path.joinpath(file_name)
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def set_best_link(self, best : str):
        if self.best_experiment_link_path is None:
            logger.warning("Best experiment link path is None, not setting link")
            return
        if self.best_experiment_link_path.exists():
            if self.best_experiment_link_path.is_symlink():
                self.best_experiment_link_path.unlink()
            else:
                raise FileExistsError(f"Best experiment link {self.best_experiment_link_path} exists and is not a symlink")
        self.best_experiment_link_path.symlink_to(self.path.joinpath(best),True)

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
        self.set_best_link(best)