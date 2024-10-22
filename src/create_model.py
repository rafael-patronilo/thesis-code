#!/usr/bin/env python
import script_base
import torch
from core import prepare_new_model
import logging
logger = logging.getLogger(__name__)

@script_base.main_wrapper
def main():
    model = torch.nn.Sequential(
        torch.nn.LazyLinear(16),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(4),
        torch.nn.Sigmoid()
    )
    prepare_new_model(
        model_name="xtrains_rn_simp_with_inv",
        model_identifier="v1",
        model=model,
        dataset_name='xtrains_ontology_simplified_comp_inv_all',
        optimizer="Adam",
        loss_fn="bce",
        val_metrics=["loss", "epoch_elapsed", "f1_score", "accuracy"],
        batch_size=32
    )
    logger.info("Model created successfully")

if __name__ == '__main__':
    main()