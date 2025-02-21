from .csv_dataset import CSVDataset
from typing import Optional, Callable, TYPE_CHECKING
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from torchvision.transforms.v2 import Transform
from pathlib import Path
if TYPE_CHECKING:
    import pandas as pd

_IDENTITY_TRANSFORM : Transform = transforms.Lambda(lambda x : x)


class CSVImageDataset(CSVDataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_path: str | Path,
        image_columns : list[str | tuple[str, Callable[[str], str]]],
        target : str | list[str],
        features: Optional[list[str]] = None,
        dtype : Optional[torch.dtype] = torch.float32,
        global_transform : Optional[Transform] = None,
        column_transforms : Optional[dict[str, Transform]] = None,
        shuffle: bool = True,
        random_state = None,
        splits: float | tuple[float, float] = (0.7, 0.15),
        filter : Optional[Callable[['pd.Series'], bool]] = None
    ):
        super().__init__(
            csv_path,
            target,
            features,
            shuffle=shuffle,
            random_state=random_state,
            splits=splits,
            filter=filter
        )
        self.images_path = Path(images_path)
        self.dtype = dtype
        self.global_transform : Transform = global_transform or _IDENTITY_TRANSFORM
        self.column_transforms = column_transforms or {}
        self.skip_image_loading = False
        for column in image_columns:
            if isinstance(column, tuple):
                col_name, path_getter = column
            else:
                col_name = column
                path_getter = lambda x: x
            def image_getter(x):
                path : Path = self.images_path.joinpath(path_getter(x))
                if self.skip_image_loading:
                    return path.relative_to(self.images_path)
                image : torch.Tensor = torchvision.io.decode_image(path) # type: ignore # (documentation claims method supports Path)
                if self.dtype is not None:
                    image = transforms.functional.to_dtype(image, dtype = self.dtype, scale=True)
                image = self.global_transform(image)
                column_transform = self.column_transforms.get(col_name, _IDENTITY_TRANSFORM)
                image = column_transform(image)
                return image
            self.add_column_preprocessor(col_name, image_getter, is_scalar=False)

