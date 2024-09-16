import main
import torch
from core import prepare_new_model

@main.main_thread_wrapper
def main_function():
    model = torch.nn.Sequential(
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(32),
        torch.nn.LazyLinear(1),
        torch.nn.Sigmoid()
    )
    prepare_new_model(
        model_name="test_model",
        model_identifier="v1",
        model=model,
        dataset='xtrains_concepts_test_1',
        optimizer="Adam",
        loss_fn="bce",
        metrics=["accuracy"],
        train_metrics=["accuracy"],
        batch_size=32
    )

if __name__ == '__main__':
    main_function()