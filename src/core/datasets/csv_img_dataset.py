from .csv_dataset import CSVDataset
from typing import Optional, Callable
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from torchvision.transforms.v2 import Transform
from pathlib import Path

_IDENTITY_TRANSFORM : Transform = transforms.Lambda(lambda x : x)


class CSVImageDataset(CSVDataset):
    def __init__(
        self,
        csv_path: str,
        images_path: str,
        image_collumns : list[str | tuple[str, Callable[[str], str]]],
        target : str | list[str],
        features: Optional[list[str]] = None,
        dtype : Optional[torch.dtype] = torch.float32,
        global_transform : Optional[Transform] = None,
        collumn_transforms : Optional[dict[str, Transform]] = None,
        shuffle: bool = True,
        random_state = None
    ):
        super().__init__(
            csv_path,
            target,
            features,
            shuffle=shuffle,
            random_state=random_state
        )
        self.images_path = Path(images_path)
        self.dtype = dtype
        self.global_transform : Transform = global_transform or _IDENTITY_TRANSFORM
        self.collumn_transforms = collumn_transforms or {}
        for collumn in image_collumns:
            if isinstance(collumn, tuple):
                col_name, path_getter = collumn
            else:
                col_name = collumn
                path_getter = lambda x: x
            def image_getter(x):
                path : Path = self.images_path.joinpath(path_getter(x))
                image : torch.Tensor = torchvision.io.decode_image(path) # type: ignore (documentation claims method supports Path)
                if self.dtype is not None:
                    image = transforms.functional.to_dtype(image, dtype = self.dtype, scale=True)
                image = self.global_transform(image)
                collumn_transform = self.collumn_transforms.get(col_name, _IDENTITY_TRANSFORM)
                image = collumn_transform(image)
                return image
            self.add_collumn_preprocessor(col_name, image_getter, is_scalar=False)

