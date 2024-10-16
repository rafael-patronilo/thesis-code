from core.util.typing import PathLike
from pathlib import Path

class FileSearcher:
    VALID_ARCHIVES = {
        '.zip' : None,
    }
    def __init__(
            self, 
            search_path: PathLike | list[PathLike], 
            recursive: bool = True, 
            open_archives : bool = True
        ):
        self._is_open = False
        self.recursive = recursive
        self.open_archives = open_archives
        self.search_path : list[Path]
        if not isinstance(search_path, list):
            self.search_path = [Path(search_path)]
        else:
            self.search_path = [Path(x) for x in search_path]

    def _is_valid_path(self, path : Path):
        return (
            path.exists() and
            (path.is_dir() or self._is_valid_archive(path))
        )
    
    def _is_valid_archive(self, path : Path):
        return (
            path.is_file() and
            path.suffix in self.VALID_ARCHIVES and
            self.open_archives
        )

    def open(self):
        if self._is_open:
            return
        self.paths_set = set()
        self.paths = []
        self.archives = []
        for x in self.search_path:
            if not self._is_valid_path(x):
                raise ValueError(f"Invalid search path: {x}")
            x = x.resolve()
            if x.is_dir():
                self._walk(x)
            elif self._is_valid_path(x):
                self._open_archive(x)
        self._is_open = True
        del self.paths_set

    def _walk(self, path : Path):
        path = path.resolve()
        assert path.is_dir()
        if path in self.paths_set:
            return
        self.paths_set.add(path)
        self.paths.append(path)
        for subpath in path.iterdir():
            if subpath.is_dir() and self.recursive:
                self._walk(subpath)
            elif self._is_valid_archive(subpath):
                self._open_archive(subpath)
    
    def _open_archive(self, path : Path):
        if path in self.paths_set:
            return
        self.paths_set.add(path)
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _search(self):
        raise NotImplementedError()

    def search(self):
        if not self._is_open:
            with self:
                self._search()
        else:
            self._search()
    
    