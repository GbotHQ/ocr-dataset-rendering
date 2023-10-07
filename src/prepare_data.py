from typing import Union
from random import randint
from pathlib import Path as pth
import zipfile
import itertools

import gdown
from mega import Mega

from datasets import load_dataset
from shuffle_iter import ShuffleIterator, DatasetShuffleIterator


zip_path = pth("../assets/")
mega = Mega()
m = mega.login()


class DownloadError(IOError):
    def __init__(self, *args):
        super().__init__(*args)
    
    def __str__(self):
        return f"Failed to download {super().__str__()}"


class MultipleDatasetShuffleIterator:
    def __init__(self, datasets: list):
        self.dataset_shuffle_iterators = [DatasetShuffleIterator(k) for k in datasets]

    def __next__(self):
        return next(
            self.dataset_shuffle_iterators[
                randint(0, len(self.dataset_shuffle_iterators) - 1)
            ]
        )


def get_text_dataset():
    datasets = [
        load_dataset(k, split="train", streaming=True)
        for k in (
            "ChristophSchuhmann/wikipedia-en-nov22-1-sentence-level",
            "ChristophSchuhmann/1-sentence-level-gutenberg-en_arxiv_pubmed_soda",
        )
    ]
    shuffled_dataset_iters = MultipleDatasetShuffleIterator(datasets)

    return datasets, shuffled_dataset_iters


def _download_and_extract(download_fn, path: Union[str, pth]) -> pth:
    path = pth(path)
    zip_path = path.with_suffix(".zip")

    if not zip_path.is_file():
        download_fn(zip_path)

    if not path.is_dir():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)

    return path

def gdrive_download_and_extract(path: Union[str, pth], file_id: str) -> pth:
    def download_fn(zip_path: pth) -> None:
        gdown.download(id=file_id, output=str(zip_path.resolve()), quiet=False)
    
    return _download_and_extract(download_fn, path)

def mega_download_and_extract(path: Union[str, pth], file_id_and_key: str) -> pth:
    def download_fn(zip_path: pth) -> None:
        m.download_url(f"https://mega.nz/file/{file_id_and_key}", str(zip_path.parent.resolve()), str(zip_path.name))
    
    return _download_and_extract(download_fn, path)


def get_hdris(hdris_dir: Union[str, pth]) -> ShuffleIterator:
    hdris_dir = pth(hdris_dir)

    hdris = list(hdris_dir.glob("*.exr"))
    return ShuffleIterator(hdris)


def get_materials(materials_dir: Union[str, pth]) -> ShuffleIterator:
    materials_dir = pth(materials_dir)

    texture_types = ["albedo", "roughness", "displacement"]

    materials = {}
    for file in materials_dir.glob("*.jpg"):
        for k in texture_types:
            if k not in file.name:
                continue
            name = file.name.split(k, 1)[0]
            material = materials.get(name, dict.fromkeys(texture_types))
            material[k] = file
            materials[name] = material

    # remove materials with missing textures
    materials = {k: v for k, v in materials.items() if None not in v.values()}

    return ShuffleIterator(list(materials.values()))


def get_fonts(fonts_dir: Union[str, pth]) -> ShuffleIterator:
    fonts_dir = pth(fonts_dir)

    fonts = list(
        itertools.chain.from_iterable(
            (fonts_dir / "fontcollection").glob(f"*.{k}") for k in ("ttf", "otf")
        )
    )
    return ShuffleIterator(fonts)
