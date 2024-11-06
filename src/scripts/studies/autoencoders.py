from core import StudyManager, ModelDetails, datasets
from core.storage_management import StudyFileManager
from torch import nn
import sys



def make_model(input_shape, linear_sizes, encoding_size=9):
    image_area = input_shape[1] * input_shape[2]
    encoder_conv = [32, 32, ('pool', 2)] + [64, ('pool', 2)] * 2 + [128, ('pool', 2)] * 2
    
    encoder_layers = []
    in_channels = input_shape[0]
    conv = nn.Conv2d(in_channels, encoder_conv[0], 3, padding=1)
    
    

def main():
    # Load study manager
    identifier = sys.argv[1]
    file_manager = StudyFileManager(f"autoencoders_{identifier}")
    dataset = datasets.get_dataset("") #TODO select correct dataset
    study_manager = StudyManager(
        file_manager,
        compare_strategy="min",
        metric_key=("train", "loss"),
        num_epochs=100
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