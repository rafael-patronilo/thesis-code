import main
from core import Trainer, ModelFileManager
import sys

# Resume training from the last checkpoint
model_path = sys.argv[1]
parts = model_path.split("_")
model_name = "_".join(parts[:-1])
model_identifier = parts[-1]

with ModelFileManager(
    model_name=model_name, 
    model_identifier=model_identifier, 
    conflict_strategy='load') as file_manager:
    # Load trainer
    trainer = Trainer.load_checkpoint(file_manager, device=main.device)
