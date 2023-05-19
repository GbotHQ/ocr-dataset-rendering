from typing import List, Dict, Any
import requests
from pathlib import Path as pth
import zipfile

from tqdm import tqdm


def find_string_containing_substring_in_list(strings: List[str], substring: str) -> str:
    return next((s for s in strings if substring in s), None)


def is_list_of_substrings_in_list_of_strings(
    strings: List[str], substrings: List[str]
) -> bool:
    return all(
        bool(find_string_containing_substring_in_list(strings, k)) for k in substrings
    )


def download_url(url: str, out_path: pth) -> None:
    with requests.Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


class AmbientCGDownloader:
    def __init__(self, extension: str, resolution: str, texture_types: List[str]):
        self.extension: str = extension
        self.resolution: str = resolution
        self.texture_types: List[str] = texture_types
        self.texture_type_to_filename_map: Dict[str, str] = {
            "Color": "albedo",
            "Roughness": "roughness",
            "Displacement": "displacement",
        }
        self.api_endpoint: str = "https://ambientcg.com/api/v2/full_json"

    def _get_material_assets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with requests.Session() as session:
            response = session.get(self.api_endpoint, params=params)

            response.raise_for_status()
            data = response.json()
        return data

    def _get_material_assets_with_pagination(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []
        offset = 0
        while offset < params["total"]:
            params["offset"] = offset
            data_list.append(self._get_material_assets(params))
            offset += params["limit"]
        return data_list

    def _has_required_texture_types(self, file_info: Dict[str, Any]) -> bool:
        texture_type_substrings = [
            f"{filename}.{self.extension}" for filename in self.texture_types
        ]
        return is_list_of_substrings_in_list_of_strings(
            file_info["zipContent"], texture_type_substrings
        )

    def _is_matching_resolution_and_type(self, file_info: Dict[str, Any]) -> bool:
        return (
            file_info["attribute"]
            == f"{self.resolution.upper()}-{self.extension.upper()}"
        )

    def _get_matching_files(self, asset: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            file_info
            for file_info in asset["downloadFolders"]["default"][
                "downloadFiletypeCategories"
            ]["zip"]["downloads"]
            if self._is_matching_resolution_and_type(file_info)
            and self._has_required_texture_types(file_info)
        ]

    def _map_asset_to_dict(
        self, asset: Dict[str, Any], file_info: Dict[str, Any]
    ) -> Dict[str, str]:
        return {
            "filename": file_info["fileName"],
            "id": asset["assetId"],
            "url": file_info["downloadLink"],
        }

    def _get_matching_assets(
        self, assets: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        file_dicts = []
        for asset in assets:
            matching_files = self._get_matching_files(asset)
            file_dicts.extend(
                self._map_asset_to_dict(asset, file_info)
                for file_info in matching_files
            )
        return file_dicts

    def _download_material_zip(self, asset: Dict[str, str], out_dir: str) -> pth:
        out_dir = pth(out_dir)
        zip_path = out_dir / asset["filename"]
        download_url(asset["url"], zip_path)
        return zip_path

    def _extract_texture_from_zip(
        self, zip_file: zipfile.ZipFile, texture_type: str, out_dir: pth
    ) -> pth:
        try:
            filename = find_string_containing_substring_in_list(
                zip_file.namelist(), f"{texture_type}.{self.extension}"
            )
            if not filename:
                return

            out_path = out_dir / filename
            zip_file.extract(filename, out_dir)
            return out_path
        except Exception as e:
            print(f"Error extracting texture from zip file: {e}")
            return

    def _extract_textures_from_zip(self, zip_path: pth, out_dir: pth) -> Dict[str, pth]:
        out_dir = pth(out_dir)
        textures: Dict[str, pth] = {}
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                for texture_type in self.texture_types:
                    mapped_texture_type = self.texture_type_to_filename_map[
                        texture_type
                    ]
                    filename = f"{zip_path.stem}_{mapped_texture_type}_{self.resolution}.{self.extension}"
                    out_path = out_dir / filename
                    if original_out_path := self._extract_texture_from_zip(
                        zip_file, texture_type, out_dir
                    ):
                        original_out_path.rename(out_path)
                        textures[mapped_texture_type] = out_path
        except Exception as e:
            print(f"Error extracting textures from zip file: {e}")

        return textures

    def _get_and_filter_material_assets(self) -> List[Dict[str, str]]:
        params: Dict[str, Any] = {
            "method": "",
            "type": "Material",
            "sort": "Alphabet",
            "include": "downloadData",
            "limit": 250,
        }

        response = self._get_material_assets(params)
        total_count = response["numberOfResults"]
        params["total"] = total_count

        data_list = self._get_material_assets_with_pagination(params)
        return self._get_matching_assets(
            [asset for data in data_list for asset in data["foundAssets"]]
        )

    def download_and_extract_materials(self, out_dir: str) -> List[Dict[str, Any]]:
        out_dir = pth(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        assets = self._get_and_filter_material_assets()

        output_assets: List[Dict[str, Any]] = []
        for asset in tqdm(assets):
            try:
                zip_path = self._download_material_zip(asset, out_dir)
                textures = self._extract_textures_from_zip(zip_path, out_dir)
                asset["textures"] = textures
                output_assets.append(asset)
                zip_path.unlink()
            except Exception as e:
                print(f"Error downloading or extracting material: {e}")

        return output_assets
