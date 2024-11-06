from core import StudyManager, ModelDetails, datasets
from core.storage_management import StudyFileManager
from torch import nn

def main():
    # Load study manager
    file_manager = StudyFileManager("rn_xtrains")
    dataset = datasets.get_dataset("xtrains_ontology_simplified_comp_all")
    study_manager = StudyManager(
        file_manager,
        compare_strategy="max",
        metric_key=("val", "accuracy"),
        num_epochs=50
    )
    architectures = [
        ('L16',nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
        ('L32', nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
        ('L16x2',nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
        ('L32x2',nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
        ('L16x3', nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
        ('L32x3', nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Sigmoid()
        )),
    ]
    base_details = ModelDetails(
        architecture=None,
        optimizer="Adam",
        loss_fn="bce",
        dataset=dataset,
        metrics=[],
        batch_size=32
    )
    experiments = [(name, base_details._replace(architecture=arch)) for name, arch in architectures]
    #study_manager.run(experiments)
    raise NotImplementedError("This script is not yet implemented")

if __name__ == '__main__':
    main()