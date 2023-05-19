from typing import IO
import zipfile
from pathlib import PurePath as ppth
from pathlib import Path as pth
import tempfile

import gdown


class Fonts:
    def __init__(self):
        self.zip_path = pth(tempfile.mkdtemp())

        self.zip_file = None
        self.font_list = []
        self.font_path_iter = iter([])

    def download_fonts(self, *gdrive_IDs: str):
        for k in gdrive_IDs:
            print(f"Downloading {k}")
            path = self.zip_path / f"{k}.zip"
            gdown.download(id=k, output=str(path), quiet=False)

            self.zip_file = zipfile.ZipFile(path, "r")
            self.font_list = [ppth(k) for k in self.zip_file.namelist() if k.lower().endswith((".otf", ".ttf"))]

    def get_font_by_path(self, path: pth) -> IO[bytes] or None:
        return self.zip_file.open(path.as_posix())