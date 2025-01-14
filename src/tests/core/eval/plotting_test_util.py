from pathlib import Path
import filecmp
import os

def create_paths(module_path : str, test_name : str) -> tuple[Path, Path]:
    path = Path(module_path).parent.joinpath(test_name)
    path.mkdir(exist_ok=True)
    expected_path = path.joinpath("expected")
    result_path = path.joinpath("results")
    if result_path.exists():
        raise FileExistsError(f"Result path {result_path} already exists, delete manually to run test")
    return expected_path, result_path

def assert_same(expected_path : Path, result_path : Path):
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected path {expected_path} does not exist")
    if filecmp.cmp(result_path, expected_path):
        os.remove(result_path)
        return
    else:
        raise AssertionError(f'{result_path} does not match {expected_path}')