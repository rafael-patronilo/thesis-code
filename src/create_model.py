import main
import torch
from core import prepare_new_model

@main.main_thread_wrapper
def main_function():
    model = torch.nn.Sequential(
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(3),
        torch.nn.Sigmoid()
    )
    prepare_new_model(
        model_name="xtrains_rn_simp",
        model_identifier="v1",
        model=model,
        dataset='xtrains_ontology_simplified_comp_all',
        optimizer="Adam",
        loss_fn="bce",
        metrics=["epoch_elapsed", "f1_score", "accuracy"],
        train_metrics=None,
        batch_size=32
    )

if __name__ == '__main__':
    main_function()