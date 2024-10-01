#!/usr/bin/env python
import script_base
import torch
from core import prepare_new_model
import logging
logger = logging.getLogger(__name__)

@script_base.main_wrapper
def main():
    model = torch.nn.Sequential(
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(3),
        torch.nn.Sigmoid()
    )
    prepare_new_model(
        model_name="test",
        model_identifier="v1",
        model=model,
        dataset='xtrains_ontology_simplified_comp_all',
        optimizer="Adam",
        loss_fn="bce",
        metrics=["epoch_elapsed", "f1_score", "accuracy"],
        train_metrics=None,
        batch_size=32
    )
    logger.info("Model created successfully")

if __name__ == '__main__':
    main()