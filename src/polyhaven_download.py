from typing import Union, List
import requests
from pathlib import Path as pth

from tqdm import tqdm


polyhaven_api_endpoint = "https://api.polyhaven.com"

polyhaven_texture_type_map = {
    "Diffuse": "albedo",
    "Rough": "roughness",
    "Displacement": "displacement",
}


def download_texture(
    name: str,
    texture_type: str,
    save_directory: Union[str, pth],
    resolution: str,
    file_extension: str,
) -> pth:
    """
    Downloads a single texture from Polyhaven.
    """
    save_directory = pth(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    mapped_texture_type = polyhaven_texture_type_map.get(texture_type, texture_type)

    file_path = (
        save_directory / f"{name}_{mapped_texture_type}_{resolution}.{file_extension}"
    )
    if file_path.is_file():
        return file_path

    file_url = requests.get(f"{polyhaven_api_endpoint}/files/{name}").json()
    file_url = file_url[texture_type][resolution][file_extension]["url"]

    response = requests.get(file_url, allow_redirects=True)

    file_path.write_bytes(response.content)

    return file_path


def download_all_hdris(
    save_directory: Union[str, pth] = "hdris",
    resolution: str = "2k",
    file_extension: str = "exr",
) -> List[pth]:
    """
    Downloads all HDRIs from Polyhaven.
    """
    names = requests.get("https://api.polyhaven.com/assets?t=hdris").json().keys()

    file_paths = []
    for name in tqdm(names):
        try:
            file_paths.append(
                download_texture(
                    name, "hdri", save_directory, resolution, file_extension
                )
            )
        except Exception as e:
            tqdm.write(f"Failed to download {name}: {e}")

    return file_paths


def download_all_materials(
    save_directory: Union[str, pth] = "materials",
    resolution: str = "1k",
    file_extension: str = "jpg",
    texture_types: List[str] = None,
) -> List[pth]:
    """
    Downloads all materials from Polyhaven.
    """
    if texture_types is None:
        texture_types = ["Diffuse", "Rough", "Displacement"]

    names = requests.get("https://api.polyhaven.com/assets?t=textures").json().keys()

    file_paths = []
    for name in tqdm(names):
        newly_downloaded = []
        try:
            newly_downloaded.append(
                {
                    k: download_texture(
                        name, k, save_directory, resolution, file_extension
                    )
                    for k in texture_types
                }
            )
        except Exception as e:
            tqdm.write(f"Failed to download {name}: {e}")
            print(newly_downloaded)
            for item in newly_downloaded:
                for k in item.values():
                    print(f"deleting {k}")
                    # k.close()

        file_paths += newly_downloaded

    return file_paths
